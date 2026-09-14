"""Phrases inside the music: pick spoken bits from the bank, shape them, drop them where the timeline says.

Runs in the worker after the composer has rendered the track. The timeline tells us where every
breakdown and build is, so a phrase can open a breakdown or land right before a drop without any
audio analysis. Every placement is written back into the timeline for the visuals (phase 5).
"""

from __future__ import annotations

import math
import os

import numpy as np

from tmg.music import synth
from tmg.music.synth import SAMPLE_RATE

DUCK_DB = -4.0          # how much the music steps back under a phrase
DUCK_RAMP_S = 0.06
TAIL_S = 0.35           # echo / reverb tail kept after the phrase


def load_phrase(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Any WAV -> float32 (2, T) at `sr`."""
    import soundfile as sf
    from scipy.signal import resample_poly

    y, rate = sf.read(path, dtype="float32", always_2d=True)
    y = y.T
    if y.shape[0] == 1:
        y = np.vstack([y, y])
    if rate != sr:
        g = math.gcd(int(rate), sr)
        y = resample_poly(y, sr // g, int(rate) // g, axis=1).astype(np.float32)
    return np.ascontiguousarray(y[:2])


def fit_ratio(length_s: float, beat_s: float, limit: float = 0.15) -> float:
    """Stretch ratio that makes the phrase end on a beat, if that needs less than `limit` change; else 1."""
    if length_s <= 0 or beat_s <= 0:
        return 1.0
    beats = max(1, round(length_s / beat_s))
    ratio = length_s / (beats * beat_s)      # >1 = speed up, <1 = slow down
    return ratio if abs(ratio - 1.0) <= limit else 1.0


def stretch(y: np.ndarray, ratio: float) -> np.ndarray:
    if abs(ratio - 1.0) < 1e-3:
        return y
    import librosa

    out = [librosa.effects.time_stretch(ch.astype(np.float32), rate=ratio) for ch in y]
    n = min(len(o) for o in out)
    return np.vstack([o[:n] for o in out]).astype(np.float32)


def shape(y: np.ndarray, *, beat_s: float, level_db: float = -6.0, telephone: bool = False, echo_repeats: int = 2,
          reverb_mix: float = 0.25, fit_to_beat: bool = True, rng: np.random.Generator | None = None) -> tuple[np.ndarray, float]:
    """The processing chain for one phrase. Returns (stereo audio, stretch ratio used)."""
    rng = rng or np.random.default_rng(0)
    peak = float(np.abs(y).max()) if y.size else 0.0
    if peak < 1e-6:
        return y, 1.0
    y = y / peak
    ratio = fit_ratio(y.shape[1] / SAMPLE_RATE, beat_s) if fit_to_beat else 1.0
    y = stretch(y, ratio)
    if telephone:
        y = np.vstack([synth.bandpass(ch.astype(np.float64), 300.0, 3400.0) for ch in y]).astype(np.float32)
        y = np.tanh(y * 1.8) / math.tanh(1.8)
    dry = y
    tail = int(TAIL_S * SAMPLE_RATE) + int(echo_repeats * beat_s * 0.75 * SAMPLE_RATE)
    out = np.zeros((2, y.shape[1] + tail + int(1.4 * SAMPLE_RATE)), dtype=np.float32)
    out[:, : y.shape[1]] += dry
    if echo_repeats > 0:
        for i, ch in enumerate(dry):
            echo = synth.delay_line(ch.astype(np.float64), time_s=beat_s * 0.75, feedback=0.45, repeats=echo_repeats)
            out[i, : len(echo)] += (echo * 0.5).astype(np.float32)
    if reverb_mix > 0:
        for i in range(2):
            wet = synth.reverb(out[i].astype(np.float64), rng=rng, seconds=1.4)
            out[i] += (wet * reverb_mix).astype(np.float32)
    peak = float(np.abs(out).max()) or 1.0
    out *= 10 ** (level_db / 20.0) / peak
    return out, ratio


def candidate_slots(timeline: dict, where: str = "both", allow_intro: bool = False) -> list[dict]:
    """Where a phrase may go: opening a breakdown, or right before a drop at the end of a build."""
    slots: list[dict] = []
    bar_s = float(timeline["bar_s"])
    beat_s = bar_s / 4.0
    sections = timeline.get("sections") or []
    for i, s in enumerate(sections):
        kind, start, end, bars = s["type"], float(s["start_s"]), float(s["end_s"]), int(s["bars"])
        nxt = sections[i + 1]["type"] if i + 1 < len(sections) else None
        if kind == "breakdown" and bars >= 4 and where in ("both", "breakdown"):
            slots.append({"slot": "breakdown", "section": i, "start_s": start + bar_s, "max_len_s": max(beat_s, (bars - 2) * bar_s), "anchor": "start"})
        if kind == "build" and bars >= 8 and nxt == "drop" and where in ("both", "build"):
            slots.append({"slot": "before-drop", "section": i, "end_s": end - beat_s * 0.5, "max_len_s": min(8 * bar_s, (bars - 1) * bar_s), "anchor": "end"})
        if kind == "intro" and allow_intro and bars >= 8:
            slots.append({"slot": "intro", "section": i, "start_s": start + 2 * bar_s, "max_len_s": (bars - 3) * bar_s, "anchor": "start"})
    return slots


def choose_placements(timeline: dict, phrases: list[dict], *, count: int, where: str, allow_intro: bool,
                      rng: np.random.Generator) -> list[dict]:
    """Pair phrases with slots: one phrase per slot, spread over the track, longest slots first when needed."""
    slots = candidate_slots(timeline, where, allow_intro)
    if not slots or not phrases or count <= 0:
        return []
    rng.shuffle(slots)
    pool = list(phrases)
    rng.shuffle(pool)
    placements: list[dict] = []
    used_sections: set[int] = set()
    for slot in slots:
        if len(placements) >= count:
            break
        if slot["section"] in used_sections:
            continue
        fitting = [p for p in pool if float(p.get("duration_s") or 0) <= slot["max_len_s"]]
        if not fitting:
            continue
        phrase = fitting[0]
        pool.remove(phrase)
        if not pool:
            pool = list(phrases)
            rng.shuffle(pool)
        used_sections.add(slot["section"])
        placements.append({**slot, "phrase": phrase})
    placements.sort(key=lambda p: p.get("start_s", p.get("end_s", 0.0)))
    return placements


def place(audio: np.ndarray, timeline: dict, placements: list[dict], *, level_db: float, telephone: bool,
          echo_repeats: int, fit_to_beat: bool, rng: np.random.Generator) -> tuple[np.ndarray, list[dict]]:
    """Mix the chosen phrases into the rendered track. audio is (samples, 2) float32 from the composer."""
    if not placements:
        return audio, []
    mix = audio.T.astype(np.float32).copy()        # (2, T)
    total = mix.shape[1]
    beat_s = float(timeline["bar_s"]) / 4.0
    done: list[dict] = []
    for p in placements:
        phrase = p["phrase"]
        path = phrase.get("path")
        if not path or not os.path.exists(path):
            continue
        try:
            y = load_phrase(path)
        except Exception:  # noqa: BLE001 - a broken file must not kill the track
            continue
        shaped, ratio = shape(y, beat_s=beat_s, level_db=level_db, telephone=telephone, echo_repeats=echo_repeats,
                              fit_to_beat=fit_to_beat, rng=rng)
        spoken_len = int(y.shape[1] / max(ratio, 1e-6))   # samples after stretching, before the tail
        if p["anchor"] == "end":
            start = int((p["end_s"]) * SAMPLE_RATE) - spoken_len
        else:
            start = int(p["start_s"] * SAMPLE_RATE)
        start = max(0, min(total - 1, start))
        end = min(total, start + shaped.shape[1])
        n = end - start
        if n <= 0:
            continue
        # the music steps back a little while the words are spoken
        duck_len = min(total - start, spoken_len + int(TAIL_S * SAMPLE_RATE))
        ramp = int(DUCK_RAMP_S * SAMPLE_RATE)
        env = np.full(duck_len, 10 ** (DUCK_DB / 20.0), dtype=np.float32)
        r = min(ramp, duck_len // 2)
        if r > 0:
            env[:r] = np.linspace(1.0, env[0], r)
            env[-r:] = np.linspace(env[0], 1.0, r)
        mix[:, start:start + duck_len] *= env
        mix[:, start:end] += shaped[:, :n]
        done.append({
            "id": phrase.get("id"), "name": phrase.get("name"), "source": phrase.get("source"),
            "slot": p["slot"], "section": p["section"],
            "start_s": round(start / SAMPLE_RATE, 3), "end_s": round((start + spoken_len) / SAMPLE_RATE, 3),
            "stretch": round(ratio, 3),
        })
    # keep the master where the composer left it
    np.tanh(mix * 1.15, out=mix)
    mix /= math.tanh(1.15)
    loudest = max(float(np.abs(mix).max()), synth.SILENCE)
    mix *= 0.89 / loudest
    return mix.T.copy(), done


__all__ = ["candidate_slots", "choose_placements", "fit_ratio", "load_phrase", "place", "shape"]
