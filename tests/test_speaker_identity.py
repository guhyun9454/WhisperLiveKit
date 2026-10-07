"""Persistent speaker identity: naming rule, session lock, Unknown handling, and output fields.

A fake embedder stands in for TitaNet so the decision logic is tested without model downloads:
each "voice" is a constant audio level that maps to a fixed embedding direction.
"""

import numpy as np
import pytest

from whisperlivekit.diarization.speaker_identity import SessionSpeakerIdentifier, SpeakerProfiles
from whisperlivekit.timed_objects import Segment

SR = 16000
DIM = 8


def voice(i, mix=None):
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    if mix is not None:
        v[mix] = 1.0
    return v


class FakeEmbedder:
    name = "fake"

    def __init__(self, table):
        self.table = table  # audio level -> embedding

    def embed(self, audio):
        return self.table[round(float(audio.mean()), 1)]


@pytest.fixture
def profiles(tmp_path):
    p = SpeakerProfiles(tmp_path, model="fake")
    p.add("Alice", np.stack([voice(0), voice(0)]))
    p.add("Bob", np.stack([voice(1)]))
    p.add("Carol", np.stack([voice(2)]))
    return SpeakerProfiles(tmp_path, model="fake")  # reload from disk


def feed(ident, slot, level, seconds, n_slots=4):
    """Stream `seconds` of audio in 0.8 s chunks where only `slot` is active."""
    frames = 10
    for _ in range(int(seconds / 0.8)):
        audio = np.full(int(0.8 * SR), level, dtype=np.float32)
        preds = np.zeros((frames, n_slots), dtype=np.float32)
        preds[:, slot] = 0.95
        ident.process_chunk(audio, preds)
    # a silent chunk closes the open piece
    ident.process_chunk(np.zeros(int(0.8 * SR), dtype=np.float32), np.zeros((frames, n_slots), dtype=np.float32))


def test_known_speaker_is_named_after_enough_speech(profiles):
    emb = FakeEmbedder({0.1: voice(0), 0.2: voice(1)})
    ident = SessionSpeakerIdentifier(emb, profiles)
    assert ident.identity(0) == (None, None)
    feed(ident, slot=0, level=0.1, seconds=4)
    assert ident.identity(0)[0] is None  # below min_speech_s
    feed(ident, slot=0, level=0.1, seconds=8)
    assert ident.identity(0)[0] == "Alice"
    # a few seconds of another voice merged into the slot do not rename it
    feed(ident, slot=0, level=0.2, seconds=4)
    assert ident.identity(0)[0] == "Alice"


def test_merged_start_is_corrected_by_later_speech(profiles):
    # diarization put Bob's voice first into this slot, then the real owner (Alice) talks
    emb = FakeEmbedder({0.1: voice(0), 0.2: voice(1)})
    ident = SessionSpeakerIdentifier(emb, profiles)
    feed(ident, slot=1, level=0.2, seconds=11)
    assert ident.identity(1)[0] == "Bob"
    feed(ident, slot=1, level=0.1, seconds=40)
    assert ident.identity(1)[0] == "Alice"


def test_one_name_per_slot(profiles):
    emb = FakeEmbedder({0.1: voice(0), 0.5: voice(0, mix=3)})
    ident = SessionSpeakerIdentifier(emb, profiles, threshold=0.6)
    feed(ident, slot=0, level=0.5, seconds=12)   # Alice-ish (0.71)
    assert ident.identity(0)[0] == "Alice"
    feed(ident, slot=1, level=0.1, seconds=12)   # clearly Alice (1.0) takes the name
    assert ident.identity(1)[0] == "Alice"
    assert ident.identity(0)[0] is None


def test_ambiguous_match_stays_unknown(profiles):
    # equally close to Alice and Bob (0.71 each, above this threshold) -> margin rule refuses to pick
    emb = FakeEmbedder({0.3: voice(0, mix=1)})
    ident = SessionSpeakerIdentifier(emb, profiles, threshold=0.6)
    feed(ident, slot=1, level=0.3, seconds=8)
    name, conf = ident.identity(1)
    assert name is None and conf is not None


def test_unenrolled_person_stays_unknown(profiles):
    emb = FakeEmbedder({0.4: voice(5)})
    ident = SessionSpeakerIdentifier(emb, profiles)
    feed(ident, slot=2, level=0.4, seconds=8)
    assert ident.identity(2)[0] is None


def test_overlap_and_short_pieces_are_ignored(profiles):
    emb = FakeEmbedder({0.1: voice(0)})
    ident = SessionSpeakerIdentifier(emb, profiles)
    # two slots active at once: no frame qualifies
    audio = np.full(int(0.8 * SR), 0.1, dtype=np.float32)
    preds = np.full((10, 4), 0.0, dtype=np.float32)
    preds[:, 0] = preds[:, 1] = 0.9
    for _ in range(10):
        ident.process_chunk(audio, preds)
    assert ident.identity(0) == (None, None)
    # 0.8 s alone is shorter than the minimum piece
    feed(ident, slot=0, level=0.1, seconds=0.8)
    assert ident.identity(0) == (None, None)


def test_candidates_restrict_matching(profiles):
    emb = FakeEmbedder({0.1: voice(0)})
    ident = SessionSpeakerIdentifier(emb, profiles, candidates=["Bob", "Carol"])
    feed(ident, slot=0, level=0.1, seconds=8)
    assert ident.identity(0)[0] is None


def test_output_fields_only_when_enabled():
    line = Segment(start=0.0, end=1.0, text="hi", speaker=2)
    assert "speaker_name" not in line.to_dict()
    line.identity = {"speaker_name": "Alice", "speaker_confidence": 0.91}
    d = line.to_dict()
    assert d["speaker"] == 2 and d["speaker_name"] == "Alice" and d["speaker_confidence"] == 0.91
