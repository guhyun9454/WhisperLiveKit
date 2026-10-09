import io
import logging
import math
import sys
from typing import List

import numpy as np
import soundfile as sf

from whisperlivekit.model_paths import detect_model_format, resolve_model_path
from whisperlivekit.timed_objects import ASRToken
from whisperlivekit.whisper.transcribe import transcribe as whisper_transcribe

logger = logging.getLogger(__name__)
class ASRBase:
    sep = " "  # join transcribe words with this character (" " for whisper_timestamped,
              # "" for faster-whisper because it emits the spaces when needed)

    def __init__(self, lan, model_size=None, cache_dir=None, model_dir=None, lora_path=None, logfile=sys.stderr):
        self.logfile = logfile
        self.transcribe_kargs = {}
        self.lora_path = lora_path
        if lan == "auto":
            self.original_language = None
        else:
            self.original_language = lan
        self.model = self.load_model(model_size, cache_dir, model_dir)

    def load_model(self, model_size, cache_dir, model_dir):
        raise NotImplementedError("must be implemented in the child class")

    def transcribe(self, audio, init_prompt=""):
        raise NotImplementedError("must be implemented in the child class")

    def use_vad(self):
        raise NotImplementedError("must be implemented in the child class")


class WhisperASR(ASRBase):
    """Uses WhisperLiveKit's built-in Whisper implementation."""
    sep = " "

    def load_model(self, model_size=None, cache_dir=None, model_dir=None):
        from whisperlivekit.whisper import load_model as load_whisper_model

        if model_dir is not None:
            resolved_path = resolve_model_path(model_dir)
            if resolved_path.is_dir():
                model_info = detect_model_format(resolved_path)
                if not model_info.has_pytorch:
                    raise FileNotFoundError(
                        f"No supported PyTorch checkpoint found under {resolved_path}"
                    )
            logger.debug(f"Loading Whisper model from custom path {resolved_path}")
            return load_whisper_model(str(resolved_path), lora_path=self.lora_path)

        if model_size is None:
            raise ValueError("Either model_size or model_dir must be set for WhisperASR")

        return load_whisper_model(model_size, download_root=cache_dir, lora_path=self.lora_path)

    def transcribe(self, audio, init_prompt=""):
        options = dict(self.transcribe_kargs)
        options.pop("vad", None)
        options.pop("vad_filter", None)
        language = self.original_language if self.original_language else None

        result = whisper_transcribe(
            self.model,
            audio,
            language=language,
            initial_prompt=init_prompt,
            condition_on_previous_text=True,
            word_timestamps=True,
            **options,
        )
        return result

    def ts_words(self, r) -> List[ASRToken]:
        """
        Converts the Whisper result to a list of ASRToken objects.
        """
        tokens = []
        for segment in r["segments"]:
            for word in segment["words"]:
                token = ASRToken(
                    word["start"],
                    word["end"],
                    word["word"],
                    probability=word.get("probability"),
                )
                tokens.append(token)
        return tokens

    def segments_end_ts(self, res) -> List[float]:
        return [segment["end"] for segment in res["segments"]]

    def use_vad(self):
        logger.warning("VAD is not currently supported for WhisperASR backend and will be ignored.")

class FasterWhisperASR(ASRBase):
    """Uses faster-whisper as the backend."""
    sep = ""

    def load_model(self, model_size=None, cache_dir=None, model_dir=None):
        from faster_whisper import WhisperModel

        if model_dir is not None:
            resolved_path = resolve_model_path(model_dir)
            logger.debug(f"Loading faster-whisper model from {resolved_path}. "
                         f"model_size and cache_dir parameters are not used.")
            model_size_or_path = str(resolved_path)
        elif model_size is not None:
            model_size_or_path = model_size
        else:
            raise ValueError("Either model_size or model_dir must be set")
        device = "auto" # Allow CTranslate2 to decide available device
        compute_type = "auto" # Allow CTranslate2 to decide faster compute type


        model = WhisperModel(
            model_size_or_path,
            device=device,
            compute_type=compute_type,
            download_root=cache_dir,
        )
        return model

    def transcribe(self, audio: np.ndarray, init_prompt: str = "") -> list:
        segments, info = self.model.transcribe(
            audio,
            language=self.original_language,
            initial_prompt=init_prompt,
            beam_size=5,
            word_timestamps=True,
            condition_on_previous_text=True,
            **self.transcribe_kargs,
        )
        return list(segments)

    def ts_words(self, segments) -> List[ASRToken]:
        tokens = []
        for segment in segments:
            if segment.no_speech_prob > 0.9:
                continue
            for word in segment.words:
                token = ASRToken(word.start, word.end, word.word, probability=word.probability)
                tokens.append(token)
        return tokens

    def segments_end_ts(self, segments) -> List[float]:
        return [segment.end for segment in segments]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

class _ReusedEncoder:
    """mlx-whisper runs the encoder once for decoding and again for word timestamps
    (find_alignment) on the same mel; reuse the last result. ~35% less time per call."""

    def __init__(self, encoder):
        self.encoder, self.key, self.out = encoder, None, None

    def __call__(self, mel):
        import hashlib

        import mlx.core as mx
        a = np.asarray(mel.astype(mx.float16))
        key = (a.shape, hashlib.blake2b(a.tobytes(), digest_size=16).digest())
        if key != self.key:
            self.out = self.encoder(mel)
            mx.eval(self.out)
            self.key = key
        return self.out

    def __getattr__(self, name):
        return getattr(self.encoder, name)


class MLXWhisper(ASRBase):
    """
    Uses MLX Whisper optimized for Apple Silicon.
    """
    sep = ""

    def load_model(self, model_size=None, cache_dir=None, model_dir=None):
        import mlx.core as mx
        from mlx_whisper.transcribe import ModelHolder, transcribe

        if model_dir is not None:
            resolved_path = resolve_model_path(model_dir)
            logger.debug(f"Loading MLX Whisper model from {resolved_path}. model_size parameter is not used.")
            model_size_or_path = str(resolved_path)
        elif model_size is not None:
            model_size_or_path = self.translate_model_name(model_size)
            logger.debug(f"Loading whisper model {model_size}. You use mlx whisper, so {model_size_or_path} will be used.")
        else:
            raise ValueError("Either model_size or model_dir must be set")

        self.model_size_or_path = model_size_or_path
        dtype = mx.float16
        model = ModelHolder.get_model(model_size_or_path, dtype)
        if not isinstance(model.encoder, _ReusedEncoder):
            model.encoder = _ReusedEncoder(model.encoder)
        return transcribe

    def translate_model_name(self, model_name):
        from whisperlivekit.model_mapping import MLX_MODEL_MAPPING
        mlx_model_path = MLX_MODEL_MAPPING.get(model_name)
        if mlx_model_path:
            return mlx_model_path
        else:
            raise ValueError(f"Model name '{model_name}' is not recognized or not supported.")

    def _transcribe_window(self, audio, init_prompt=""):
        """Exactly one encoder pass, one greedy decode and one alignment for <= 30 s of audio.

        mlx_whisper.transcribe() seeks inside the window: after timestamp-only outputs and
        after every word-timestamp pass it jumps back to the last word and decodes again,
        which made single 10 s calls take 26-46 s on an M2 with large-v3-turbo.
        """
        import mlx.core as mx
        from mlx_whisper.audio import HOP_LENGTH, N_FRAMES, N_SAMPLES, SAMPLE_RATE, log_mel_spectrogram, pad_or_trim
        from mlx_whisper.decoding import DecodingOptions, decode
        from mlx_whisper.timing import add_word_timestamps
        from mlx_whisper.tokenizer import get_tokenizer
        from mlx_whisper.transcribe import ModelHolder

        model = ModelHolder.get_model(self.model_size_or_path, mx.float16)
        mel = log_mel_spectrogram(audio, n_mels=model.dims.n_mels, padding=N_SAMPLES)
        num_frames = mel.shape[-2] - N_FRAMES
        mel = pad_or_trim(mel, N_FRAMES, axis=-2).astype(mx.float16)
        result = decode(model, mel, DecodingOptions(
            task="transcribe",
            language=self.original_language,
            temperature=0.0,
            without_timestamps=True,
            prompt=init_prompt or None,
            # A repetition loop otherwise runs to the 224-token limit; speech stays well
            # under ~20 tokens/s.
            sample_len=min(224, 32 + int(20 * len(audio) / SAMPLE_RATE)),
            fp16=True,
        ))
        tokenizer = get_tokenizer(model.is_multilingual, num_languages=model.num_languages,
                                  language=result.language, task="transcribe")
        segment = {
            "seek": 0, "start": 0.0, "end": num_frames * HOP_LENGTH / SAMPLE_RATE,
            "text": result.text, "tokens": [t for t in result.tokens if t < tokenizer.eot],
            "no_speech_prob": result.no_speech_prob,
        }
        add_word_timestamps(segments=[segment], model=model, tokenizer=tokenizer, mel=mel,
                            num_frames=num_frames, last_speech_timestamp=0.0)
        return [segment]

    def transcribe(self, audio, init_prompt=""):
        if self.transcribe_kargs:
            logger.warning("Transcribe kwargs (vad, task) are not compatible with MLX Whisper and will be ignored.")
        if len(audio) <= 30 * 16000:
            return self._transcribe_window(audio, init_prompt)
        segments = self.model(
            audio,
            language=self.original_language,
            initial_prompt=init_prompt,
            word_timestamps=True,
            condition_on_previous_text=True,
            path_or_hf_repo=self.model_size_or_path,
            # No temperature fallback: on a repetition loop mlx-whisper re-decodes at
            # T=0.2..1.0, which took 28-39 s per call on an M2; LocalAgreement already
            # discards unstable output, so one greedy pass is enough.
            temperature=0.0,
            # A prompt-induced repetition loop ran to the 224-token limit (16-31 s per call);
            # real speech stays well under ~20 tokens/s.
            sample_len=min(224, 32 + int(20 * len(audio) / 16000)),
        )
        return segments.get("segments", [])

    def ts_words(self, segments) -> List[ASRToken]:
        tokens = []
        for segment in segments:
            if segment.get("no_speech_prob", 0) > 0.9:
                continue
            for word in segment.get("words", []):
                token = ASRToken(word["start"], word["end"], word["word"])
                tokens.append(token)
        return tokens

    def segments_end_ts(self, res) -> List[float]:
        return [s["end"] for s in res]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True


class OpenaiApiASR(ASRBase):
    """Uses OpenAI's Whisper API for transcription."""
    def __init__(self, lan=None, temperature=0, logfile=sys.stderr):
        self.logfile = logfile
        self.transcribe_kargs = {}
        self.modelname = "whisper-1"
        self.original_language = None if lan == "auto" else lan
        self.response_format = "verbose_json"
        self.temperature = temperature
        self.load_model()
        self.use_vad_opt = False
        self.direct_english_translation = False
        self.task = "transcribe"

    def load_model(self, *args, **kwargs):
        from openai import OpenAI
        self.client = OpenAI()
        self.transcribed_seconds = 0

    def ts_words(self, segments) -> List[ASRToken]:
        """
        Converts OpenAI API response words into ASRToken objects while
        optionally skipping words that fall into no-speech segments.
        Translation responses have no word-level timestamps, so we fall
        back to segment-level tokens in that case.
        """
        no_speech_segments = []
        if self.use_vad_opt:
            for segment in segments.segments:
                if segment.no_speech_prob > 0.8:
                    no_speech_segments.append((segment.start, segment.end))
        words = getattr(segments, "words", None)
        if words is None:
            return [
                ASRToken(segment.start, segment.end, segment.text)
                for segment in segments.segments
                if not self.use_vad_opt or segment.no_speech_prob <= 0.8
            ]
        tokens = []
        for word in words:
            start = word.start
            end = word.end
            if any(s[0] <= start <= s[1] for s in no_speech_segments):
                continue
            tokens.append(ASRToken(start, end, word.word))
        return tokens

    def segments_end_ts(self, res) -> List[float]:
        words = getattr(res, "words", None)
        if words is None:
            return [s.end for s in res.segments]
        return [s.end for s in words]

    def transcribe(self, audio_data, prompt=None, *args, **kwargs):
        prompt = prompt or kwargs.get("init_prompt")
        buffer = io.BytesIO()
        buffer.name = "temp.wav"
        sf.write(buffer, audio_data, samplerate=16000, format="WAV", subtype="PCM_16")
        buffer.seek(0)
        self.transcribed_seconds += math.ceil(len(audio_data) / 16000)
        task = self.transcribe_kargs.get("task", self.task)
        params = {
            "model": self.modelname,
            "file": buffer,
            "response_format": self.response_format,
            "temperature": self.temperature,
        }
        if task != "translate":
            params["timestamp_granularities"] = ["word", "segment"]
            if not self.direct_english_translation and self.original_language:
                params["language"] = self.original_language
        if prompt:
            params["prompt"] = prompt
        proc = self.client.audio.translations if task == "translate" else self.client.audio.transcriptions
        transcript = proc.create(**params)
        logger.debug(f"OpenAI API processed accumulated {self.transcribed_seconds} seconds")
        return transcript

    def use_vad(self):
        self.use_vad_opt = True
