"""After Stop: cut leading/trailing silence + normalise the peak. The untouched original stays as .raw.wav."""

from __future__ import annotations

import os
import shutil

import numpy as np
import soundfile as sf

from tmg.jobs import protocol


def find_bounds(x: np.ndarray, rate: int, threshold_db: float, pad_ms: float) -> tuple[int, int]:
    """First/last sample above the threshold, padded by pad_ms on both sides."""
    mono = np.abs(x).max(axis=1) if x.ndim == 2 else np.abs(x)
    thr = 10 ** (threshold_db / 20.0)
    idx = np.flatnonzero(mono > thr)
    if idx.size == 0:
        return 0, len(mono)
    pad = int(rate * pad_ms / 1000.0)
    return max(0, int(idx[0]) - pad), min(len(mono), int(idx[-1]) + pad + 1)


def run(params: dict) -> dict:
    path = params["path"]
    raw_path = params.get("raw_path") or (os.path.splitext(path)[0] + ".raw.wav")
    threshold_db = float(params.get("threshold_db", -45.0))
    pad_ms = float(params.get("pad_ms", 120))
    do_trim = bool(params.get("trim", True))
    do_norm = bool(params.get("normalize", True))
    peak_db = float(params.get("peak_db", -1.0))

    protocol.progress(0.1, "reading")
    if not os.path.exists(raw_path):
        shutil.copy2(path, raw_path)  # the untouched original, for "Restore original"
    x, rate = sf.read(raw_path, dtype="float32", always_2d=True)
    total = len(x)
    if total == 0:
        raise RuntimeError("empty recording - no audio was written (was anything playing on that output?)")

    protocol.progress(0.4, "trimming silence")
    a, b = (0, total)
    if do_trim:
        a, b = find_bounds(x, rate, threshold_db, pad_ms)
        if b - a < int(0.05 * rate):  # all silence: keep everything rather than produce nothing
            a, b = 0, total
    y = x[a:b]

    raw_peak = float(np.abs(y).max()) if len(y) else 0.0
    protocol.progress(0.7, "normalising")
    if do_norm and raw_peak > 1e-6:
        y = y * (10 ** (peak_db / 20.0) / raw_peak)
    peak = float(np.abs(y).max()) if len(y) else 0.0

    protocol.progress(0.9, "writing")
    tmp = path + ".tmp"
    sf.write(tmp, y, rate, format="WAV", subtype="PCM_16")
    os.replace(tmp, path)
    return {
        "path": path, "raw_path": raw_path, "sample_rate": int(rate), "channels": int(y.shape[1]),
        "duration_s": round(len(y) / rate, 3), "raw_duration_s": round(total / rate, 3),
        "trimmed_start_s": round(a / rate, 3), "trimmed_end_s": round((total - b) / rate, 3),
        "raw_peak_db": round(20 * np.log10(raw_peak), 2) if raw_peak > 1e-6 else -100.0,
        "peak_db": round(20 * np.log10(peak), 2) if peak > 1e-6 else -100.0,
    }
