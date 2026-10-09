"""Utterance policy: each pause-delimited utterance is transcribed once, with correct absolute times."""
import numpy as np

from whisperlivekit.local_agreement.online_asr import UtteranceASRProcessor
from whisperlivekit.timed_objects import ASRToken

SR = 16000


class FakeASR:
    """One word per second of non-silent audio, named by its level."""
    sep = " "
    tokenizer = None
    confidence_validation = False
    buffer_trimming = "segment"
    buffer_trimming_sec = 15

    def __init__(self):
        self.calls = []

    def transcribe(self, audio, init_prompt=""):
        self.calls.append(len(audio) / SR)
        return [(i, round(float(audio[i * SR]), 2)) for i in range(len(audio) // SR) if audio[i * SR] > 0]

    def ts_words(self, res):
        return [ASRToken(i, i + 0.8, f"w{level}-{i}") for i, level in res]


def speech(seconds, level=0.5):
    return np.full(int(seconds * SR), level, dtype=np.float32)


def test_short_pauses_batch_until_min_length_then_one_call():
    asr = FakeASR()
    p = UtteranceASRProcessor(asr)
    p.insert_audio_chunk(speech(5))
    assert p.start_silence() == ([], 0.0)          # 5 s < MIN_S: keep collecting
    p.end_silence(0.5, 0)
    p.insert_audio_chunk(speech(6))
    tokens, upto = p.start_silence()                # 11.5 s buffered: transcribe once
    assert asr.calls == [11.5] and upto == 11.5
    assert [t.start for t in tokens] == [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]  # 5.0-5.5 s is the inserted pause


def test_long_pause_keeps_audio_and_shifts_time():
    asr = FakeASR()
    p = UtteranceASRProcessor(asr)
    p.insert_audio_chunk(speech(2))
    p.start_silence()
    p.end_silence(10, 0)                            # parent would drop these 2 s
    p.insert_audio_chunk(speech(2, 0.7))
    tokens, _ = p.process_iter()
    assert [t.start for t in tokens] == [0, 1]
    tokens, upto = p.finish()
    assert [(t.start, t.text) for t in tokens] == [(12, "w0.7-0"), (13, "w0.7-1")] and upto == 14


def test_monologue_is_cut_at_max_length_without_losing_words():
    asr = FakeASR()
    p = UtteranceASRProcessor(asr)
    p.insert_audio_chunk(speech(21))
    tokens, upto = p.process_iter()
    assert [t.start for t in tokens] == list(range(19))   # last 1.5 s held back
    assert upto == tokens[-1].end
    p.insert_audio_chunk(speech(2))
    rest, _ = p.finish()
    assert [t.start for t in rest][0] >= 19 - 0.2 and len(asr.calls) == 2


def test_decoding_loops_are_cut():
    from whisperlivekit.local_agreement.online_asr import _cut_repetition
    toks = lambda ws: [ASRToken(i, i + 1, " " + w) for i, w in enumerate(ws)]
    loop = toks("QPR은 작아지는데 QR에서 QR에서 QR에서 QR에서 QR에서".split())
    assert [t.text for t in _cut_repetition(loop)] == [" QPR은", " 작아지는데", " QR에서", " QR에서"]
    two = toks("안쓰 아름다운 거 아름다운 거 아름다운 거 아름다운 거".split())
    assert len(_cut_repetition(two)) == 5
    fine = toks("네 네 그렇습니다 그걸로 학습을".split())
    assert _cut_repetition(fine) == fine
