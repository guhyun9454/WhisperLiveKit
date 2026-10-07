"""Persistent speaker identity on top of streaming Sortformer.

Sortformer keeps "who spoke when" inside one session (slot 0..3). This module names those slots
across sessions: frames where exactly one slot is confident are cut into short pieces, embedded with
a speaker-verification model (TitaNet by default), and the running centroid per slot is matched
against enrolled profiles. A name is accepted only if

    speech >= min_speech_s  AND  best_score > threshold  AND  best_score - second_score > margin

Every new piece updates the slot's running centroid and the decision is re-evaluated, so a slot
whose first seconds were polluted by another voice (diarization merges) can still settle on the
right person. A name sticks until a different name passes the rule, and one name is held by at
most one slot (the higher-scoring slot wins). Until then the slot stays unknown (name None).

Profiles live in a directory, one JSON file per person:
    {"name": "...", "embeddings": [[...], ...], "model": "titanet_large"}

Enroll from audio:
    python -m whisperlivekit.diarization.speaker_identity enroll PROFILE_DIR NAME a.wav [b.wav ...]
"""
from __future__ import annotations

import json
import logging
import pathlib
import re
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


def _norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


class SpeakerEmbeddingProvider(Protocol):
    name: str

    def embed(self, audio: np.ndarray) -> np.ndarray:
        """16 kHz mono float32 audio -> 1-D speaker embedding."""


class TitaNetEmbeddingProvider:
    name = "titanet_large"

    def __init__(self, model_name: str = "nvidia/speakerverification_en_titanet_large", device: Optional[str] = None):
        import torch
        from nemo.collections.asr.models import EncDecSpeakerLabelModel

        self._torch = torch
        self.model = EncDecSpeakerLabelModel.from_pretrained(model_name).eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self._lock = threading.Lock()  # one model shared by all sessions

    def embed(self, audio: np.ndarray) -> np.ndarray:
        torch = self._torch
        with self._lock, torch.inference_mode():
            x = torch.tensor(audio, dtype=torch.float32, device=self.device)[None]
            _, emb = self.model.forward(input_signal=x, input_signal_length=torch.tensor([x.shape[1]], device=self.device))
        return emb[0].float().cpu().numpy()


class SpeakerProfiles:
    """Enrolled people, one JSON file each. Matching uses the normalized mean of each person's embeddings."""

    def __init__(self, directory: str | pathlib.Path, model: str = TitaNetEmbeddingProvider.name):
        self.dir = pathlib.Path(directory)
        self.model = model
        self.embeddings: Dict[str, np.ndarray] = {}
        self.reload()

    def reload(self) -> None:
        self.embeddings = {}
        if not self.dir.exists():
            return
        for f in sorted(self.dir.glob("*.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            if d.get("model", self.model) != self.model:
                logger.warning("Skipping profile %s: built with %s, not %s", f.name, d.get("model"), self.model)
                continue
            self.embeddings[d["name"]] = _norm(np.asarray(d["embeddings"], dtype=np.float32))
        logger.info("Loaded %d speaker profiles from %s", len(self.embeddings), self.dir)

    @property
    def names(self) -> List[str]:
        return sorted(self.embeddings)

    def centroids(self, candidates: Optional[List[str]] = None) -> tuple[List[str], np.ndarray]:
        names = [n for n in self.names if candidates is None or n in candidates]
        if not names:
            return [], np.zeros((0, 0), dtype=np.float32)
        return names, _norm(np.stack([_norm(self.embeddings[n].mean(0)) for n in names]))

    def add(self, name: str, embeddings: np.ndarray) -> pathlib.Path:
        """Append embeddings to a person's profile (creates it if new)."""
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / (re.sub(r"[^\w.-]+", "_", name) + ".json")
        old = json.loads(path.read_text(encoding="utf-8"))["embeddings"] if path.exists() else []
        new = old + _norm(np.atleast_2d(embeddings)).tolist()
        path.write_text(json.dumps({"name": name, "model": self.model, "embeddings": new}, ensure_ascii=False))
        self.embeddings[name] = np.asarray(new, dtype=np.float32)
        return path


@dataclass
class SlotState:
    piece: List[np.ndarray] = field(default_factory=list)   # audio frames of the piece being collected
    emb_sum: Optional[np.ndarray] = None
    n_pieces: int = 0
    speech_s: float = 0.0
    name: Optional[str] = None       # accepted name (sticky until another name passes the rule)
    score: Optional[float] = None    # latest best score
    best: Optional[str] = None       # latest best-scoring profile


class SessionSpeakerIdentifier:
    """Per-session naming of Sortformer slots. Feed it every diarized chunk (audio + per-frame probs)."""

    def __init__(
        self,
        embedder: SpeakerEmbeddingProvider,
        profiles: SpeakerProfiles,
        threshold: float = 0.70,
        margin: float = 0.10,
        on_prob: float = 0.7,
        off_prob: float = 0.3,
        min_piece_s: float = 1.5,
        max_piece_s: float = 6.0,
        min_speech_s: float = 10.0,
        candidates: Optional[List[str]] = None,
    ):
        self.embedder, self.profiles = embedder, profiles
        self.threshold, self.margin = threshold, margin
        self.on_prob, self.off_prob = on_prob, off_prob
        self.min_piece_s, self.max_piece_s = min_piece_s, max_piece_s
        self.min_speech_s = min_speech_s
        self.names, self.P = profiles.centroids(candidates)
        self.slots: Dict[int, SlotState] = {}
        self._lock = threading.Lock()

    def process_chunk(self, audio: np.ndarray, preds: np.ndarray) -> None:
        """audio: chunk PCM (16 kHz); preds: (frames, n_slots) Sortformer probabilities for that chunk."""
        if not len(preds) or not self.names:
            return
        n_frames = len(preds)
        hop = len(audio) / n_frames
        for f in range(n_frames):
            frame = audio[int(f * hop): int((f + 1) * hop)]
            for k in range(preds.shape[1]):
                st = self.slots.setdefault(k, SlotState())
                others = np.delete(preds[f], k)
                if preds[f, k] > self.on_prob and (others.max() if len(others) else 0) < self.off_prob:
                    st.piece.append(frame)
                    if sum(map(len, st.piece)) >= self.max_piece_s * SAMPLE_RATE:
                        self._close_piece(k, st)
                elif st.piece:
                    self._close_piece(k, st)

    def _close_piece(self, k: int, st: SlotState) -> None:
        audio = np.concatenate(st.piece)
        st.piece = []
        if len(audio) < self.min_piece_s * SAMPLE_RATE:
            return
        e = _norm(self.embedder.embed(audio))
        with self._lock:
            st.emb_sum = e if st.emb_sum is None else st.emb_sum + e
            st.n_pieces += 1
            st.speech_s += len(audio) / SAMPLE_RATE
            sc = self.P @ _norm(st.emb_sum)
            order = np.argsort(-sc)
            best = float(sc[order[0]])
            gap = best - (float(sc[order[1]]) if len(order) > 1 else -1.0)
            st.score, st.best = best, self.names[order[0]]
            if st.speech_s >= self.min_speech_s and best > self.threshold and gap > self.margin \
                    and st.best != st.name:
                self._assign(k, st, st.best, gap)

    def _assign(self, k: int, st: SlotState, name: str, gap: float) -> None:
        """Give `name` to slot k unless another slot holds it with a higher score."""
        holder = next((j for j, s in self.slots.items() if j != k and s.name == name), None)
        if holder is not None:
            if (self.slots[holder].score or 0) >= (st.score or 0):
                return
            logger.info("Speaker slot %d loses %s to slot %d", holder, name, k)
            self.slots[holder].name = None
        if st.name:
            logger.info("Speaker slot %d renamed %s -> %s", k, st.name, name)
        st.name = name
        logger.info("Speaker slot %d identified as %s (score %.3f, margin %.3f, %.1fs speech)",
                    k, name, st.score, gap, st.speech_s)

    def identity(self, slot: int) -> tuple[Optional[str], Optional[float]]:
        """(name or None, confidence) for a 0-based Sortformer slot."""
        with self._lock:
            st = self.slots.get(slot)
            if st is None:
                return None, None
            return st.name, (round(st.score, 3) if st.score is not None else None)

    def session_embeddings(self) -> Dict[int, np.ndarray]:
        """Mean embedding per slot, e.g. to enroll an unknown speaker after the meeting."""
        with self._lock:
            return {k: _norm(s.emb_sum) for k, s in self.slots.items() if s.emb_sum is not None}


class SpeakerIdentityModel:
    """Shared across sessions: one embedding model + the profile directory."""

    def __init__(self, profile_dir: str, threshold: float = 0.70, margin: float = 0.10,
                 candidates: Optional[List[str]] = None, embedder: Optional[SpeakerEmbeddingProvider] = None):
        self.embedder = embedder or TitaNetEmbeddingProvider()
        self.profiles = SpeakerProfiles(profile_dir, model=self.embedder.name)
        self.threshold, self.margin, self.candidates = threshold, margin, candidates

    def new_session(self) -> SessionSpeakerIdentifier:
        self.profiles.reload()  # pick up people enrolled since the last session
        return SessionSpeakerIdentifier(self.embedder, self.profiles, self.threshold, self.margin,
                                        candidates=self.candidates)


def _enroll_cli(argv: List[str]) -> None:
    import argparse

    import soundfile as sf

    p = argparse.ArgumentParser(prog="python -m whisperlivekit.diarization.speaker_identity enroll",
                                description="Add a person's voice to a profile directory.")
    p.add_argument("profile_dir")
    p.add_argument("name")
    p.add_argument("audio", nargs="+", help="16 kHz-resamplable audio files with only this person speaking")
    p.add_argument("--piece", type=float, default=6.0, help="seconds per embedding piece")
    a = p.parse_args(argv)
    emb = TitaNetEmbeddingProvider()
    out = []
    for path in a.audio:
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        x = x.mean(1)
        if sr != SAMPLE_RATE:
            import librosa
            x = librosa.resample(x, orig_sr=sr, target_sr=SAMPLE_RATE)
        step = int(a.piece * SAMPLE_RATE)
        for i in range(0, len(x), step):
            if len(x[i:i + step]) >= 1.5 * SAMPLE_RATE:
                out.append(emb.embed(x[i:i + step]))
    path = SpeakerProfiles(a.profile_dir, model=emb.name).add(a.name, np.stack(out))
    print(f"enrolled {a.name}: {len(out)} pieces -> {path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "enroll":
        _enroll_cli(sys.argv[2:])
    else:
        print(__doc__)
