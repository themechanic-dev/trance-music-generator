import json
import os

import numpy as np
import pytest

from tmg import db as dbmod
from tmg.music import composer, render
from tmg.music import plan as planmod
from tmg.music.plan import DEFAULT_PROFILE, build_plan, hits, semitones_to_degrees

PROFILE = {
    "n_tracks": 22,
    "tempo": {"hist": {"138": 4, "140": 6, "142": 5, "146": 2}, "median": 140.6},
    "duration": {"hist_minutes": {"6": 5, "7": 8, "8": 4}},
    "keys": {"A minor": 6, "F minor": 4, "C major": 3},
    "sections": {
        "sequences": {"intro>build>drop>breakdown>drop>outro": 6, "build>drop>breakdown>drop>outro": 2},
        "lengths": {"intro": {"8": 5, "16": 3}, "build": {"8": 4, "16": 6}, "drop": {"32": 6, "48": 3, "64": 2},
                    "breakdown": {"8": 6, "16": 5, "32": 3}, "outro": {"8": 5, "16": 3}},
        "transitions": {"intro": {"build": 0.5, "drop": 0.5}, "build": {"drop": 1.0},
                        "drop": {"breakdown": 0.6, "outro": 0.3, "build": 0.1}, "breakdown": {"drop": 0.6, "build": 0.4}},
    },
    "patterns": {
        "kick": {"1000100010001000": 0.6, "1000100010001010": 0.3, "0000000000000000": 0.1},
        "snare": {"0000100000001000": 0.5, "0000000000000000": 0.5},
        "hat": {"0010001000100010": 0.5, "0101010101010101": 0.5},
        "bass": {"0111011101110111": 0.5, "0010001000100010": 0.5},
    },
    "bass_movement": {"0,0,0,0": 0.7, "0,8,3,10": 0.2, "0,5,8,10": 0.1},
}


def test_plan_is_deterministic_and_drawn_from_the_profile():
    a, b = build_plan(PROFILE, 42), build_plan(PROFILE, 42)
    assert a.to_dict() == b.to_dict()
    assert build_plan(PROFILE, 43).to_dict() != a.to_dict()
    for seed in range(1, 30):
        p = build_plan(PROFILE, seed)
        assert 138.0 <= p.bpm < 148.0                                  # inside the histogram
        assert p.key_name in ("A minor", "F minor", "C major")
        types = [s.type for s in p.sections]
        assert types[0] in ("intro", "build") and types[-1] == "outro"
        assert "drop" in types and "outro" not in types[:-1]
        assert all(t1 != t2 for t1, t2 in zip(types, types[1:]))       # no section repeats itself
        assert types[-2] != "build"                                    # a build always arrives somewhere
        for s in p.sections:
            assert s.bars % 4 == 0 and 8 <= s.bars <= 64
            assert hits(s.kick_pattern) >= 2 and len(s.kick_pattern) == 16
            assert s.bass_pattern in PROFILE["patterns"]["bass"]
        assert abs(p.duration_s / 60 - p.target_minutes) <= 1.0, (seed, p.duration_s / 60, p.target_minutes)
        assert p.profile_tracks == 22


def test_plan_fixed_length_and_flavor():
    p = build_plan(PROFILE, 7, minutes=4.0, flavor="psy")
    assert p.flavor.name == "psy" and p.flavor.acid_lead
    assert 3.2 <= p.duration_s / 60 <= 5.0
    p2 = build_plan(None, 7)                                           # empty library -> built-in profile
    assert p2.profile_tracks == 0 and 134 <= p2.bpm < 146


def test_semitones_to_degrees():
    assert semitones_to_degrees([0, 8, 3, 10], planmod.MINOR) == (0, 5, 2, 6)
    assert semitones_to_degrees([0, 5, 7, 0], planmod.MAJOR) == (0, 3, 4, 0)
    assert semitones_to_degrees([1], planmod.MINOR) in ((0,), (1,))


def test_default_profile_is_well_formed():
    assert DEFAULT_PROFILE["patterns"]["kick"]
    p = build_plan(DEFAULT_PROFILE, 1)
    assert p.total_bars >= 48


def _tiny_plan(seed=3):
    p = build_plan(PROFILE, seed, minutes=3.0)
    # shrink to a few bars so the render is fast, keeping one of each section
    kept = []
    for kind in ("intro", "build", "drop", "breakdown", "outro"):
        s = next((x for x in p.sections if x.type == kind), None)
        if s is not None:
            s.bars = 4
            kept.append(s)
    p.sections = kept
    return p


def test_render_audio_and_timeline():
    p = _tiny_plan()
    progress = []
    audio, tl = composer.render(p, progress=lambda f, m: progress.append((f, m)))
    assert audio.shape[1] == 2 and audio.dtype == np.float32
    assert abs(audio.shape[0] / 44100 - (p.duration_s + 1.0)) < 0.05
    peak = float(np.abs(audio).max())
    assert 0.85 <= peak <= 0.9                                          # normalised to 0.89
    d = tl.to_dict()
    assert [s["type"] for s in d["sections"]] == [s.type for s in p.sections]
    assert len(d["bars"]) == p.total_bars and len(d["beats"]) == p.total_bars * 4
    kick_sections = [s for s in p.sections if s.kick]
    expected_kicks = sum(hits(s.kick_pattern) * s.bars for s in kick_sections)
    assert 0 < len(d["kicks"]) <= expected_kicks                        # fills drop the last beat's kicks
    assert d["sections"][-1]["end_s"] == pytest.approx(p.duration_s, abs=1e-3)
    assert progress and progress[-1][0] >= 0.8
    json.dumps(d)


def test_write_outputs_and_productions(tmp_path):
    p = _tiny_plan(5)
    audio, tl = composer.render(p)
    files = render.write_outputs(audio, p, tl.to_dict(), formats=("mp3", "wav"), bitrate_k=128, out_dir=tmp_path)
    assert files["title"] == render.name_for(5) and render.name_for(5) == render.name_for(5)
    assert os.path.getsize(files["mp3"]) > 10_000 and os.path.getsize(files["wav"]) > 100_000
    assert json.load(open(files["timeline"]))["bpm"] == p.bpm
    assert json.load(open(files["plan"]))["seed"] == 5
    files2 = render.write_outputs(audio, p, tl.to_dict(), formats=("mp3",), out_dir=tmp_path / "b")
    assert files2["wav"] is None and not os.path.exists(str(tmp_path / "b" / (os.path.basename(files2["mp3"])[:-4] + ".wav")))
    db = dbmod.Database(tmp_path / "p.sqlite")
    pid = db.add_production(seed=5, title=files["title"], bpm=p.bpm, key_name=p.key_name, duration_s=p.duration_s, flavor=p.flavor.name,
                            structure=p.structure, mp3_path=files["mp3"], wav_path=files["wav"], timeline_path=files["timeline"],
                            plan=p.to_dict(), profile_tracks=22)
    rows = db.list_productions()
    assert rows[0]["id"] == pid and rows[0]["structure"] == p.structure
    assert db.get_production(pid)["plan"]["seed"] == 5
    db.delete_production(pid)
    assert db.list_productions() == []
    db.close()
