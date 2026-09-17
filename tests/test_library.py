import json
import math
import os

import numpy as np
import pytest
from conftest import make_wav

from tmg import db as dbmod
from tmg.library import analysis, profile, scan

# ---- scan ---------------------------------------------------------------------------------

def _touch(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")


def test_scan_modes(tmp_path):
    root = tmp_path / "collection"
    _touch(root / "CD1" / "01 a.mp3")
    _touch(root / "CD1" / "02 b.flac")
    _touch(root / "CD1" / "cover.jpg")
    _touch(root / "CD2" / "01 c.wav")
    _touch(root / "CD2" / "sub" / "deep.ogg")
    _touch(root / "$RECYCLE.BIN" / "junk.mp3")
    _touch(root / "loose.m4a")
    assert scan.scan_track(str(root / "CD1" / "01 a.mp3")) == [str(root / "CD1" / "01 a.mp3")]
    assert scan.scan_track(str(root / "CD1" / "cover.jpg")) == []
    assert [os.path.basename(p) for p in scan.scan_cd(str(root / "CD1"))] == ["01 a.mp3", "02 b.flac"]
    assert [os.path.basename(p) for p in scan.scan_cd(str(root))] == ["loose.m4a"]          # no recursion
    col = [os.path.relpath(p, root) for p in scan.scan_collection(str(root))]
    assert col == ["loose.m4a", "CD1/01 a.mp3", "CD1/02 b.flac", "CD2/01 c.wav", "CD2/sub/deep.ogg"]  # recycle bin skipped
    assert scan.guess_kind_for_drop(str(root)) == "collection"
    assert scan.guess_kind_for_drop(str(root / "CD1")) == "cd"
    assert scan.guess_kind_for_drop(str(root / "loose.m4a")) == "track"
    with pytest.raises(ValueError):
        scan.scan("album", str(root))


def test_probe_reads_wav(tmp_path):
    path = make_wav(tmp_path / "Some Track.wav", seconds=1.5)
    info = scan.probe(path)
    assert abs(info["duration_s"] - 1.5) < 0.05
    assert info["sample_rate"] == 48000 and info["channels"] == 2
    assert info["title"] == "Some Track" and info["album"] == tmp_path.name


# ---- analysis building blocks --------------------------------------------------------------

def test_fold_bpm():
    assert analysis.fold_bpm(69) == 138
    assert analysis.fold_bpm(276) == 138
    assert analysis.fold_bpm(140) == 140


def test_slot_pattern_four_on_the_floor():
    fps = analysis.ASR / analysis.HOP
    bar_dur = 240 / 140
    env = np.zeros(int(fps * bar_dur * 2) + 10, dtype=np.float32)
    for beat in range(4):
        env[int(round(beat * bar_dur / 4 * fps))] = 1.0
    env[int(round(2 * bar_dur / 16 * fps))] = 0.2   # a weak ghost hit is ignored
    assert analysis.slot_pattern(env, 0.0, bar_dur) == "1000100010001000"
    assert analysis.slot_pattern(np.zeros(100, dtype=np.float32), 0.0, bar_dur) == "0" * 16


def test_root_midi_finds_the_bass_note():
    sr = analysis.ASR
    t = np.arange(int(0.22 * sr)) / sr
    a1 = np.sin(2 * math.pi * 55.0 * t).astype(np.float32)          # A1
    assert analysis.root_midi(a1, sr) == 9                            # A
    e1 = 0.6 * np.sin(2 * math.pi * 41.2 * t).astype(np.float32)     # E1
    assert analysis.root_midi(e1, sr) == 4                            # E
    assert analysis.root_midi(np.zeros(int(0.22 * sr), dtype=np.float32), sr) is None


def test_estimate_key():
    chroma = np.zeros(12)
    for pc in (9, 11, 0, 2, 4, 5, 7):     # A natural minor scale
        chroma[pc] = 1.0
    chroma[9] += 1.5
    chroma[4] += 0.5
    key, mode, conf = analysis.estimate_key(chroma)
    assert (key, mode) == ("A", "minor") and conf > 0
    assert analysis.estimate_key(np.zeros(12)) == ("?", "?", 0.0)


def test_segment_sections():
    kick = np.array([False] * 16 + [True] * 48 + [False] * 32 + [True] * 32 + [False] * 8)
    energy = np.array([-30.0] * 16 + [-20.0] * 16 + [-14.0] * 32 + [-26.0] * 32 + [-14.0] * 32 + [-30.0] * 8)
    secs = analysis.segment_sections(kick, energy)
    types = [s["type"] for s in secs]
    assert types == ["intro", "build", "drop", "breakdown", "drop", "outro"]
    assert secs[1]["bars"] == 16 and secs[2]["bars"] == 32 and secs[3]["bars"] == 32
    assert sum(s["bars"] for s in secs) == len(kick)
    assert analysis.segment_sections(np.array([]), np.array([])) == []
    # a 2-bar gap in the kick must not split one drop into 'drop > drop'
    kick2 = np.array([True] * 20 + [False] * 2 + [True] * 20)
    energy2 = np.full(len(kick2), -14.0)
    assert [s["type"] for s in analysis.segment_sections(kick2, energy2)] == ["drop"]


def test_bar_grid_and_downbeat():
    beats = np.arange(0, 40) * 0.5           # 120 BPM, 40 beats
    kick_env = np.zeros(int(40 * 0.5 * analysis.ASR / analysis.HOP) + 5, dtype=np.float32)
    for b in range(2, 40, 4):                # the strong kick sits on beats 2, 6, 10, ... -> phase 2
        kick_env[int(round(beats[b] * analysis.ASR / analysis.HOP))] = 1.0
    assert analysis.downbeat_phase(kick_env, beats) == 2
    bars = analysis.bar_grid(beats, 2)
    assert bars[0] == 1.0 and abs(bars[1] - 3.0) < 1e-9 and len(bars) == 9


def _syllables(sr: int, seconds: float, rate_hz: float = 4.0, f0: float = 140.0) -> np.ndarray:
    """A voice-like signal: harmonic 'vowels' shaped by formants around 600 and 1500 Hz, real gaps between the
    syllables, and a whisper of band-limited noise at each syllable onset (the consonant)."""
    from scipy.signal import butter, sosfilt

    rng = np.random.default_rng(1)
    n = int(sr * seconds)
    t = np.arange(n) / sr
    harm = np.zeros(n)
    for k in range(1, 30):
        f = f0 * k
        formant = math.exp(-((f - 600) / 250) ** 2) + 0.7 * math.exp(-((f - 1500) / 400) ** 2)
        harm += formant * np.sin(2 * math.pi * f * t + 0.3 * k)
    harm /= np.abs(harm).max()
    phase = (t * rate_hz) % 1.0
    env = np.clip(np.sin(math.pi * np.minimum(phase, 0.55) / 0.55), 0.0, None) ** 0.7   # 55 % vowel, 45 % gap
    env[phase >= 0.55] = 0.0
    onset = np.where(phase < 0.14, 1.0, 0.0)                                             # the consonant
    sos = butter(4, [2000, 5000], btype="band", fs=sr, output="sos")
    noise = sosfilt(sos, rng.standard_normal(n)) * onset * 0.5
    return (0.3 * harm * env + 0.3 * noise).astype(np.float32)


def test_speech_score_prefers_talk_over_sustained_notes():
    sr = analysis.ASR
    talk = _syllables(sr, 3.0)
    t = np.arange(sr * 3) / sr
    note = (0.3 * np.sin(2 * math.pi * 220 * t)).astype(np.float32)
    s_talk = analysis.speech_score(analysis.speech_features(talk, sr))
    s_note = analysis.speech_score(analysis.speech_features(note, sr))
    assert s_talk > 0.5 and s_note < 0.2                                     # 0.5 is the bank's admission line
    locked = analysis.speech_score(analysis.speech_features(talk, sr), bpm=240.0)   # 4 Hz syllables == the beat
    assert locked < s_talk


def test_vocal_segments_keeps_the_talk_and_drops_the_rest():
    sr = analysis.ASR
    y = np.zeros(sr * 12, dtype=np.float32)
    y[sr * 1: sr * 4] = _syllables(sr, 3.0)                                        # 3 s of "talk" from 1 s
    t = np.arange(sr * 3) / sr
    y[sr * 6: sr * 9] = (0.3 * np.sin(2 * math.pi * 220 * t)).astype(np.float32)   # 3 s sustained note
    y[int(sr * 10.5): int(sr * 10.8)] = 0.3                                          # 0.3 s blip -> too short
    segs = analysis.vocal_segments(y, sr, max_count=3)
    assert len(segs) == 1
    assert abs(segs[0]["start"] - 1.0) < 0.15 and abs(segs[0]["end"] - 4.0) < 0.15
    assert segs[0]["score"] >= analysis.SPEECH_MIN_SCORE


def test_analyze_synthetic_track_without_demucs(tmp_path):
    """A 64-second '138 BPM' track: kicks on every beat, bass on A, 8 kick-less bars in the middle."""
    sr, bpm = 44100, 138.0
    beat = 60 / bpm
    n = int(sr * 64)
    t = np.arange(n) / sr
    y = np.zeros(n, dtype=np.float32)
    bar_dur = 4 * beat
    for k in range(int(64 / beat)):
        s = int(k * beat * sr)
        bar = int(k * beat / bar_dur)
        if 12 <= bar < 20:            # breakdown: no kick
            continue
        tt = np.arange(min(n - s, int(0.12 * sr))) / sr
        y[s:s + len(tt)] += (np.sin(2 * math.pi * 55 * tt) * np.exp(-tt * 30)).astype(np.float32)   # kick
    y += 0.15 * np.sin(2 * math.pi * 55.0 * t).astype(np.float32)                                    # bass A1
    y += 0.05 * np.sin(2 * math.pi * 440.0 * t).astype(np.float32)                                   # a tone
    import soundfile as sf

    path = tmp_path / "synthetic.wav"
    sf.write(str(path), np.stack([y, y], 1), sr)
    result, stems = analysis.analyze(str(path), demucs_model=None, vocal_opts=None)
    assert stems is None
    assert abs(result["bpm"] - bpm) < 2.0, result["bpm"]
    assert result["bars"] >= 20
    kick_top = next(iter(result["kick_patterns"]))
    assert kick_top == "1000100010001000", result["kick_patterns"]
    types = [s["type"] for s in result["sections"]]
    assert "breakdown" in types and "drop" in types, types
    assert result["key"] == "A"
    rel = [r for r in result["bass_roots_rel"] if r is not None]
    assert rel and max(set(rel), key=rel.count) == 0        # bass on the tonic
    json.dumps(result)                                     # serialisable


# ---- profile --------------------------------------------------------------------------------

def _fake_analysis(bpm, key="A", mode="minor", sections=None, kick="1000100010001000", roots=(0, 0, 0, 0)):
    return {
        "bpm": bpm, "bars": 200, "duration_s": 360.0, "key": key, "mode": mode, "demucs": True,
        "sections": sections or [
            {"type": "intro", "start_bar": 0, "bars": 16, "energy_db": -30},
            {"type": "build", "start_bar": 16, "bars": 16, "energy_db": -20},
            {"type": "drop", "start_bar": 32, "bars": 32, "energy_db": -14},
            {"type": "breakdown", "start_bar": 64, "bars": 30, "energy_db": -26},
            {"type": "drop", "start_bar": 94, "bars": 32, "energy_db": -14},
            {"type": "outro", "start_bar": 126, "bars": 16, "energy_db": -30},
        ],
        "kick_patterns": {kick: 150, "1010101010101010": 10}, "snare_patterns": {"0000100000001000": 100},
        "hat_patterns": {"0010001000100010": 100}, "bass_patterns": {"1010101010101010": 100},
        "bass_roots_rel": list(roots) * 20, "vocal_segments": [],
    }


def test_build_profile():
    analyses = [_fake_analysis(138.0), _fake_analysis(140.2), _fake_analysis(145.0, key="D"), _fake_analysis(132.0, roots=(0, 10, 8, 7))]
    p = profile.build_profile(analyses)
    assert p["n_tracks"] == 4
    assert p["tempo"]["min"] == 132.0 and p["tempo"]["max"] == 145.0
    assert sum(p["tempo"]["hist"].values()) == 4
    assert list(p["sections"]["sequences"])[0] == "intro>build>drop>breakdown>drop>outro"
    assert p["sections"]["lengths"]["breakdown"] == {32: 4}          # 30 rounds to 32
    assert p["sections"]["transitions"]["breakdown"] == {"drop": 1.0}
    assert abs(p["sections"]["drop_minus_breakdown_db"] - 12.0) < 1e-6
    assert next(iter(p["patterns"]["kick"])) == "1000100010001000"
    assert "0,0,0,0" in p["bass_movement"] and "0,10,8,7" in p["bass_movement"]
    assert p["keys"]["A minor"] == 3
    assert "138" in profile.describe(p) or "Tempo" in profile.describe(p)
    assert profile.build_profile([]) == {"version": profile.PROFILE_VERSION, "n_tracks": 0}


# ---- db -------------------------------------------------------------------------------------

def test_db_library(tmp_path):
    db = dbmod.Database(tmp_path / "lib.sqlite")
    imp = db.add_import("cd", "/music/CD1")
    t1 = db.add_track("/music/CD1/a.mp3", imp, "CD1", "A", "X", 300.0, 1000, 1.0)
    assert db.add_track("/music/CD1/a.mp3", imp, "CD1", "A", "X", 300.0, 1000, 1.0) is None     # duplicate path
    t2 = db.add_track("/music/CD1/b.mp3", imp, "CD1", "B", "X", 310.0, 1000, 1.0)
    assert db.track_counts()["queued"] == 2
    assert [t["id"] for t in db.tracks_to_analyze()] == [t1, t2]
    db.update_track(t1, status="done", analysis={"bpm": 138.0, "bars": 100}, bpm=138.0, key_name="A minor")
    db.update_track(t2, status="failed", error="boom")
    assert db.get_track(t1)["analysis"] == {"bpm": 138.0, "bars": 100}
    assert db.analyses()[0]["track_id"] == t1
    assert db.list_tracks(search="B")[0]["id"] == t2
    assert db.requeue_failed() == 1 and db.track_counts()["queued"] == 1
    db.save_profile({"n_tracks": 1, "tempo": {}})
    assert db.get_profile()["n_tracks"] == 1 and db.get_profile()["built_utc"]
    db.add_phrase_full("lib-1-1-abc", "A · vocal 1", "/x.wav", "library", "ready", 2.0, 44100, 2, -1.0, t1)
    assert db.count_phrases_for_track(t1) == 1
    assert db.list_phrases(source="library")[0]["track_id"] == t1
    assert db.list_phrases(source="capture") == []
    db.delete_track(t2)
    assert db.get_track(t2) is None
    db.close()


def test_reextract_skips_only_tracks_done_with_the_same_settings():
    from tmg.jobs.kinds import library_phrases as lp

    rule = lp.rule_of({"threshold_db": -40.0, "min_len": 1.2, "max_len": 8.0, "max_count": 3, "min_score": 0.5})
    assert rule == lp.DEFAULT_RULE
    assert lp.needs_redo({"version": 1}, rule)                                    # old loudness picks
    assert not lp.needs_redo({"version": 2}, rule)                                # re-extracted before the rule was stored
    assert not lp.needs_redo({"version": 2, "phrase_rule": dict(rule)}, rule)
    stricter = dict(rule, min_score=0.65)
    assert lp.needs_redo({"version": 2}, stricter)                                # the user raised the score: redo
    assert lp.needs_redo({"version": 2, "phrase_rule": dict(rule)}, stricter)
