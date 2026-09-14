"""Per-frame control signals for the visuals, derived from the timeline (no audio analysis needed).

The composer wrote down every kick, bar, section, riser and phrase, so the picture can follow the
music exactly: a pulse on each kick, a hit on each drop, calm in the breakdowns. Pure numpy.
"""

from __future__ import annotations

import math

import numpy as np

SECTION_ENERGY = {"intro": 0.25, "build": 0.6, "drop": 1.0, "breakdown": 0.2, "outro": 0.3}
SECTION_MOOD = {"intro": "calm", "build": "medium", "drop": "intense", "breakdown": "calm", "outro": "calm"}


def _decay_env(times: list[float], n_frames: int, fps: float, decay_s: float, hard: bool = True) -> np.ndarray:
    """1 at each event, exponential decay afterwards (max of overlapping events)."""
    env = np.zeros(n_frames, dtype=np.float32)
    if not times:
        return env
    t = np.arange(n_frames) / fps
    for ev in times:
        i0 = int(math.floor(ev * fps))
        if i0 >= n_frames:
            continue
        i0 = max(0, i0)
        seg = np.exp(-np.maximum(t[i0:] - ev, 0.0) / max(decay_s, 1e-3))   # never above 1, even on the event's own frame
        np.maximum(env[i0:], seg.astype(np.float32), out=env[i0:])
    return env


def build_signals(timeline: dict, fps: float = 30.0) -> dict[str, np.ndarray]:
    """Everything a shader wants per frame, as arrays of length n_frames."""
    duration = float(timeline["duration_s"])
    n = int(math.ceil(duration * fps))
    t = np.arange(n) / fps
    bar_s = float(timeline["bar_s"])
    beat_s = bar_s / 4.0
    sections = timeline.get("sections") or []

    energy = np.full(n, 0.3, dtype=np.float32)
    drop = np.zeros(n, dtype=np.float32)
    section_index = np.zeros(n, dtype=np.int32)
    section_progress = np.zeros(n, dtype=np.float32)
    for i, s in enumerate(sections):
        a, b = int(float(s["start_s"]) * fps), int(float(s["end_s"]) * fps)
        a, b = max(0, a), min(n, max(a + 1, b))
        kind = s["type"]
        energy[a:b] = SECTION_ENERGY.get(kind, 0.5)
        section_index[a:b] = i
        section_progress[a:b] = np.linspace(0.0, 1.0, b - a, endpoint=False)
        if kind == "drop":
            drop[a:b] = 1.0
        elif kind == "build":
            drop[a:b] = np.linspace(0.3, 0.9, b - a, endpoint=False)
    # builds climb towards the drop; risers do the same over their span
    for r in timeline.get("risers") or []:
        a, b = int(float(r[0]) * fps), int(float(r[1]) * fps)
        a, b = max(0, a), min(n, max(a + 1, b))
        energy[a:b] = np.maximum(energy[a:b], np.linspace(energy[a], 1.0, b - a, endpoint=False).astype(np.float32))

    kick = _decay_env([float(k) for k in timeline.get("kicks") or []], n, fps, decay_s=0.16)
    drop_starts = [float(s["start_s"]) for s in sections if s["type"] == "drop"]
    impact = _decay_env(drop_starts + [float(c) for c in timeline.get("crashes") or []], n, fps, decay_s=0.9)
    fill = _decay_env([float(f) for f in timeline.get("fills") or []], n, fps, decay_s=0.45)

    phrase = np.zeros(n, dtype=np.float32)
    for p in timeline.get("phrases") or []:
        a, b = int(float(p["start_s"]) * fps), int(float(p["end_s"]) * fps)
        phrase[max(0, a):min(n, max(a + 1, b))] = 1.0

    beat_phase = np.mod(t, beat_s) / beat_s
    bar_phase = np.mod(t, bar_s) / bar_s
    bars = t / bar_s
    # smooth energy so the picture never jumps between sections except on a drop hit
    smooth = _smooth(energy, int(fps * 1.5))
    smooth = np.maximum(smooth, drop * 0.85 * (impact > 0.02))
    return {
        "t": t.astype(np.float32), "energy": smooth.astype(np.float32), "energy_raw": energy, "drop": drop,
        "kick": kick, "impact": impact, "fill": fill, "phrase": phrase,
        "beat_phase": beat_phase.astype(np.float32), "bar_phase": bar_phase.astype(np.float32), "bars": bars.astype(np.float32),
        "section_index": section_index, "section_progress": section_progress,
    }


def _smooth(x: np.ndarray, width: int) -> np.ndarray:
    if width <= 1 or len(x) < 3:
        return x.copy()
    k = np.ones(width, dtype=np.float32) / width
    padded = np.pad(x, (width // 2, width - width // 2 - 1), mode="edge")
    return np.convolve(padded, k, mode="valid").astype(np.float32)


__all__ = ["SECTION_ENERGY", "SECTION_MOOD", "build_signals"]
