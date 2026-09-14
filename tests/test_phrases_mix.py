import math

import numpy as np
import soundfile as sf

from tmg.music import phrases_mix as pm
from tmg.music.synth import SAMPLE_RATE

TIMELINE = {
    "bpm": 140.0, "bar_s": 240 / 140, "duration_s": 120.0,
    "sections": [
        {"type": "intro", "start_bar": 0, "bars": 8, "start_s": 0.0, "end_s": 8 * 240 / 140},
        {"type": "build", "start_bar": 8, "bars": 16, "start_s": 8 * 240 / 140, "end_s": 24 * 240 / 140},
        {"type": "drop", "start_bar": 24, "bars": 16, "start_s": 24 * 240 / 140, "end_s": 40 * 240 / 140},
        {"type": "breakdown", "start_bar": 40, "bars": 16, "start_s": 40 * 240 / 140, "end_s": 56 * 240 / 140},
        {"type": "drop", "start_bar": 56, "bars": 8, "start_s": 56 * 240 / 140, "end_s": 64 * 240 / 140},
        {"type": "outro", "start_bar": 64, "bars": 8, "start_s": 64 * 240 / 140, "end_s": 72 * 240 / 140},
    ],
}


def test_fit_ratio():
    beat = 60 / 140
    assert pm.fit_ratio(4 * beat, beat) == 1.0
    r = pm.fit_ratio(4.2 * beat, beat)
    assert abs(r - 1.05) < 1e-6                  # 5 % faster to land on 4 beats
    assert abs(pm.fit_ratio(4.5 * beat, beat) - 1.125) < 1e-6   # 4.5 beats rounds to 4 -> 12.5 % faster, within the limit
    assert pm.fit_ratio(1.3 * beat, beat) == 1.0 # 30 % away from a beat -> untouched
    assert pm.fit_ratio(0.0, beat) == 1.0


def test_candidate_slots_and_choice():
    slots = pm.candidate_slots(TIMELINE, "both")
    kinds = sorted(s["slot"] for s in slots)
    assert kinds == ["before-drop", "breakdown"]
    bd = next(s for s in slots if s["slot"] == "breakdown")
    assert abs(bd["start_s"] - (40 + 1) * 240 / 140) < 1e-6
    pre = next(s for s in slots if s["slot"] == "before-drop")
    assert pre["end_s"] < 24 * 240 / 140
    assert [s["slot"] for s in pm.candidate_slots(TIMELINE, "breakdown")] == ["breakdown"]
    assert len(pm.candidate_slots(TIMELINE, "both", allow_intro=True)) == 3
    phrases = [{"id": "a", "name": "A", "path": "/x/a.wav", "duration_s": 2.0}, {"id": "b", "name": "B", "path": "/x/b.wav", "duration_s": 3.0}]
    rng = np.random.default_rng(1)
    placed = pm.choose_placements(TIMELINE, phrases, count=2, where="both", allow_intro=False, rng=rng)
    assert len(placed) == 2 and {p["phrase"]["id"] for p in placed} == {"a", "b"}
    assert placed[0].get("end_s", placed[0].get("start_s")) <= placed[1].get("start_s", placed[1].get("end_s"))
    long_phrase = [{"id": "l", "name": "L", "path": "/x/l.wav", "duration_s": 60.0}]
    assert pm.choose_placements(TIMELINE, long_phrase, count=2, where="both", allow_intro=False, rng=rng) == []
    assert pm.choose_placements(TIMELINE, phrases, count=0, where="both", allow_intro=False, rng=rng) == []


def test_shape_and_place(tmp_path):
    sr = SAMPLE_RATE
    t = np.arange(int(1.0 * sr)) / sr
    voice = (0.5 * np.sin(2 * math.pi * 300 * t) * np.sin(2 * math.pi * 3 * t)).astype(np.float32)
    path = tmp_path / "voice.wav"
    sf.write(str(path), np.stack([voice, voice], 1), 48000)      # a different rate on purpose
    y = pm.load_phrase(str(path))
    assert y.shape[0] == 2 and abs(y.shape[1] / sr - 1.0 * 48000 / 48000) < 0.2
    shaped, ratio = pm.shape(y, beat_s=60 / 140, level_db=-6.0, telephone=True, echo_repeats=2, rng=np.random.default_rng(0))
    assert shaped.shape[0] == 2 and shaped.shape[1] > y.shape[1]
    assert abs(20 * math.log10(np.abs(shaped).max()) - (-6.0)) < 0.2
    assert 0.85 <= ratio <= 1.15

    # a quiet track: the phrase must show up at the breakdown slot and the music must duck there
    total = int(TIMELINE["duration_s"] * sr)
    music = (0.2 * np.sin(2 * math.pi * 220 * np.arange(total) / sr)).astype(np.float32)
    audio = np.stack([music, music], 1)
    phrase_row = {"id": "v", "name": "Voice", "path": str(path), "duration_s": 1.0, "source": "capture"}
    rng = np.random.default_rng(3)
    placements = pm.choose_placements(TIMELINE, [phrase_row], count=1, where="breakdown", allow_intro=False, rng=rng)
    out, done = pm.place(audio, TIMELINE, placements, level_db=-3.0, telephone=False, echo_repeats=0, fit_to_beat=False, rng=rng)
    assert out.shape == audio.shape and len(done) == 1
    d = done[0]
    assert d["slot"] == "breakdown" and abs(d["start_s"] - 41 * 240 / 140) < 0.01 and d["stretch"] == 1.0
    a, b = int(d["start_s"] * sr), int(d["end_s"] * sr)
    inside = np.abs(out[a:b, 0]).max()
    before = np.abs(out[a - sr: a - sr // 2, 0]).max()
    assert inside > before * 1.5                                   # the voice is audible on top
    assert np.abs(out).max() <= 0.9
    missing = pm.place(audio, TIMELINE, [{**placements[0], "phrase": {**phrase_row, "path": "/nope.wav"}}],
                       level_db=-3.0, telephone=False, echo_repeats=0, fit_to_beat=True, rng=rng)
    assert missing[1] == []
