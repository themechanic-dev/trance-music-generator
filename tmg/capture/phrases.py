"""The phrase bank: WAV files in data/phrases - trimming, waveform data, playback.

Standard library only (numpy is optional, for a faster waveform) - runs in the GUI process.
"""

from __future__ import annotations

import os
import random
import re
import shutil
import subprocess
import threading
import time
import wave
from array import array
from collections.abc import Callable
from datetime import datetime

from tmg import log, paths

_log = log.get("phrases")
_ALNUM = "0123456789abcdefghijklmnopqrstuvwxyz"


def new_phrase_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return stamp + "-" + "".join(random.choice(_ALNUM) for _ in range(3))


def safe_name(name: str, fallback: str = "phrase") -> str:
    name = re.sub(r"[\x00-\x1f/\\]", " ", name or "").strip()
    return name[:80] or fallback


def wav_info(path: str) -> dict:
    with wave.open(path, "rb") as w:
        frames, rate, ch, width = w.getnframes(), w.getframerate(), w.getnchannels(), w.getsampwidth()
    return {"frames": frames, "rate": rate, "channels": ch, "width": width, "duration_s": frames / rate if rate else 0.0}


def trim_wav(src: str, dst: str, start_s: float, end_s: float) -> dict:
    """Write [start_s, end_s] of src into dst. dst == src is allowed (goes through a temp file)."""
    with wave.open(src, "rb") as w:
        rate, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        total = w.getnframes()
        a = max(0, min(total, int(round(start_s * rate))))
        b = max(a + 1, min(total, int(round(end_s * rate))))
        w.setpos(a)
        data = w.readframes(b - a)
    tmp = dst + ".tmp"
    with wave.open(tmp, "wb") as out:
        out.setnchannels(ch)
        out.setsampwidth(width)
        out.setframerate(rate)
        out.writeframes(data)
    os.replace(tmp, dst)
    return {"frames": b - a, "rate": rate, "channels": ch, "duration_s": (b - a) / rate}


def slice_to_temp(src: str, start_s: float, end_s: float) -> str:
    paths.TMP.mkdir(parents=True, exist_ok=True)
    dst = str(paths.TMP / f"play-{int(time.time() * 1000)}.wav")
    trim_wav(src, dst, start_s, end_s)
    return dst


def waveform_bins(path: str, bins: int = 1200) -> list[tuple[float, float]]:
    """(min, max) per bin of a mono mix, in [-1, 1]. For drawing, not for analysis."""
    with wave.open(path, "rb") as w:
        ch, width, total = w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(total)
    if width != 2 or total == 0:
        return []
    try:
        import numpy as np  # the system Python has numpy; if it is missing we fall back to the slow path

        s = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if ch > 1:
            s = s.reshape(-1, ch).mean(axis=1)
        n = len(s)
        bins = min(bins, n)
        edges = np.linspace(0, n, bins + 1, dtype=np.int64)
        return [(float(s[edges[i]:edges[i + 1]].min()), float(s[edges[i]:edges[i + 1]].max())) for i in range(bins)]
    except ImportError:
        pass
    samples = array("h")
    samples.frombytes(raw)
    n = total
    bins = min(bins, n)
    step = n / bins
    stride = max(1, int(step // 64))  # subsample for speed
    out = []
    for i in range(bins):
        lo, hi = 1.0, -1.0
        for f in range(int(i * step), int((i + 1) * step), stride):
            v = 0.0
            for c in range(ch):
                v += samples[f * ch + c]
            v /= ch * 32768.0
            lo, hi = min(lo, v), max(hi, v)
        out.append((lo if lo <= hi else 0.0, hi if hi >= lo else 0.0))
    return out


def move_to_trash(*files: str) -> None:
    paths.TRASH.mkdir(parents=True, exist_ok=True)
    for f in files:
        if f and os.path.exists(f):
            shutil.move(f, str(paths.TRASH / os.path.basename(f)))


class Player:
    """Playback through `pw-play` (PipeWire, default output)."""

    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    @property
    def playing(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def play(self, path: str, on_finished: Callable[[], None] | None = None) -> None:
        self.stop()
        with self._lock:
            self.proc = subprocess.Popen(["pw-play", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            proc = self.proc

        def watch() -> None:
            proc.wait()
            if on_finished:
                on_finished()

        threading.Thread(target=watch, daemon=True).start()

    def stop(self) -> None:
        with self._lock:
            proc, self.proc = self.proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                proc.kill()
