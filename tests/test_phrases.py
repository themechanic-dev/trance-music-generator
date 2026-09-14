import os
import re

from conftest import make_wav

from tmg import paths
from tmg.capture import phrases


def test_ids_and_names():
    pid = phrases.new_phrase_id()
    assert re.fullmatch(r"\d{8}-\d{6}-[0-9a-z]{3}", pid)
    assert phrases.safe_name("  hello/world\x00 ") == "hello world"
    assert phrases.safe_name("", "fallback") == "fallback"


def test_wav_info_and_trim(tmp_path):
    src = make_wav(tmp_path / "s.wav", seconds=2.0)
    info = phrases.wav_info(src)
    assert info["duration_s"] == 2.0 and info["channels"] == 2 and info["rate"] == 48000
    out = phrases.trim_wav(src, str(tmp_path / "t.wav"), 0.5, 1.0)
    assert abs(out["duration_s"] - 0.5) < 1e-6
    # in-place trim through the temp file
    out2 = phrases.trim_wav(src, src, 0.0, 0.25)
    assert abs(out2["duration_s"] - 0.25) < 1e-6
    assert abs(phrases.wav_info(src)["duration_s"] - 0.25) < 1e-6


def test_waveform_bins(tmp_path):
    src = make_wav(tmp_path / "w.wav", seconds=1.0, amplitude=0.8)
    bins = phrases.waveform_bins(src, 100)
    assert len(bins) == 100
    assert all(-1.0 <= lo <= hi <= 1.0 for lo, hi in bins)
    assert max(hi for _, hi in bins) > 0.7


def test_slice_and_trash(tmp_data):
    src = make_wav(paths.PHRASES / "p.wav", seconds=1.0)
    tmp = phrases.slice_to_temp(src, 0.2, 0.4)
    assert tmp.startswith(str(paths.TMP)) and abs(phrases.wav_info(tmp)["duration_s"] - 0.2) < 1e-6
    phrases.move_to_trash(src, "does-not-exist.wav")
    assert not os.path.exists(src)
    assert (paths.TRASH / "p.wav").exists()
