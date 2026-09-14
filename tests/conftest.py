import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["TMG_ROOT"] = str(ROOT)

from tmg import paths  # noqa: E402


@pytest.fixture
def tmp_data(tmp_path, monkeypatch):
    """Point every data folder at a temporary directory so tests never touch data/."""
    for name in ("DATA", "PHRASES", "TRASH", "LOGS", "TMP", "OUTPUT"):
        d = tmp_path / name.lower()
        d.mkdir()
        monkeypatch.setattr(paths, name, d)
    monkeypatch.setattr(paths, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(paths, "DB_FILE", tmp_path / "tmg.sqlite")
    return tmp_path


def make_wav(path, seconds=1.0, rate=48000, channels=2, amplitude=0.5, freq=440.0, silence_before=0.0, silence_after=0.0):
    """Synthetic 16-bit WAV: silence + sine + silence."""
    import math
    import struct
    import wave

    frames = bytearray()
    n_before, n_tone, n_after = int(silence_before * rate), int(seconds * rate), int(silence_after * rate)
    for i in range(n_before + n_tone + n_after):
        if n_before <= i < n_before + n_tone:
            v = int(amplitude * 32767 * math.sin(2 * math.pi * freq * (i - n_before) / rate))
        else:
            v = 0
        frames += struct.pack("<" + "h" * channels, *([v] * channels))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))
    return str(path)
