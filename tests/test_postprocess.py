import os

import numpy as np
import soundfile as sf
from conftest import make_wav

from tmg.jobs.kinds import phrase_postprocess as pp


def test_trim_and_normalise(tmp_path):
    path = make_wav(tmp_path / "p.wav", seconds=1.0, amplitude=0.2, silence_before=0.5, silence_after=0.7)
    raw = str(tmp_path / "p.raw.wav")
    r = pp.run({"path": path, "raw_path": raw, "threshold_db": -45, "pad_ms": 100, "normalize": True, "peak_db": -1.0})
    assert os.path.exists(raw) and abs(r["raw_duration_s"] - 2.2) < 0.01
    assert abs(r["duration_s"] - 1.2) < 0.02          # 1.0 s tone + 2 x 100 ms padding
    assert abs(r["peak_db"] - (-1.0)) < 0.1
    y, rate = sf.read(path)
    assert rate == 48000 and abs(np.abs(y).max() - 10 ** (-1 / 20)) < 0.01


def test_all_silence_keeps_everything(tmp_path):
    path = make_wav(tmp_path / "s.wav", seconds=0.8, amplitude=0.0)
    r = pp.run({"path": path, "trim": True, "normalize": True})
    assert abs(r["duration_s"] - 0.8) < 0.01 and r["peak_db"] == -100.0


def test_no_trim_no_normalise(tmp_path):
    path = make_wav(tmp_path / "n.wav", seconds=0.5, amplitude=0.3, silence_before=0.3)
    r = pp.run({"path": path, "trim": False, "normalize": False})
    assert abs(r["duration_s"] - 0.8) < 0.01
    assert abs(r["peak_db"] - 20 * np.log10(0.3)) < 0.1
