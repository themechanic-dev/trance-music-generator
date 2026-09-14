"""Real audio features per frame: a 64-band spectrum and RMS, so the visuals can react to the actual sound
(the timeline says where the kick is; the spectrum says how the lead screams)."""

from __future__ import annotations

import os
import subprocess

import numpy as np

BANDS = 64
ASR = 22050


def decode_mono(path: str, sr: int = ASR, ffmpeg: str | None = None) -> np.ndarray:
    ff = ffmpeg or os.environ.get("TMG_FFMPEG", "ffmpeg")
    raw = subprocess.run([ff, "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def spectrum_frames(y: np.ndarray, sr: int, fps: float, n_frames: int, bands: int = BANDS, n_fft: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """(n_frames, bands) log-spaced band energies in 0..1, and (n_frames,) RMS in 0..1."""
    hop = sr / fps
    window = np.hanning(n_fft).astype(np.float32)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    edges = np.geomspace(40.0, min(sr / 2, 16000.0), bands + 1)
    idx = np.searchsorted(freqs, edges)
    out = np.zeros((n_frames, bands), dtype=np.float32)
    rms = np.zeros(n_frames, dtype=np.float32)
    for f in range(n_frames):
        c = int(f * hop)
        a = c - n_fft // 2
        seg = np.zeros(n_fft, dtype=np.float32)
        lo, hi = max(0, a), min(len(y), a + n_fft)
        if hi > lo:
            seg[lo - a:hi - a] = y[lo:hi]
        rms[f] = float(np.sqrt(np.mean(seg**2)))
        mag = np.abs(np.fft.rfft(seg * window))
        for b in range(bands):
            i0, i1 = idx[b], max(idx[b] + 1, idx[b + 1])
            out[f, b] = float(mag[i0:i1].mean()) if i1 > i0 else 0.0
    # to dB, then normalise each band by its own loud moments, then smooth a little over time
    db = 20 * np.log10(np.maximum(out, 1e-6))
    ref = np.percentile(db, 98, axis=0)
    norm = np.clip((db - (ref - 40.0)) / 40.0, 0.0, 1.0).astype(np.float32)
    k = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    for b in range(bands):
        norm[:, b] = np.convolve(np.pad(norm[:, b], 1, mode="edge"), k, mode="valid")
    rms_n = np.clip(rms / max(float(np.percentile(rms, 99)), 1e-6), 0.0, 1.0).astype(np.float32)
    return norm, rms_n


def features_for(path: str, fps: float, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    y = decode_mono(path)
    return spectrum_frames(y, ASR, fps, n_frames)


__all__ = ["BANDS", "decode_mono", "features_for", "spectrum_frames"]
