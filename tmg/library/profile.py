"""The style profile: distributions and transitions learned from the analysed tracks.

Not averages - distributions. 'Tempo 128-148 peaking at 140', 'after a 32-bar breakdown: 70 % build of 16,
30 % build of 8', 'kick 4/4 in 90 % of bars'. The composer (phase 3) draws from these; a different
library gives a different profile. Pure Python + numpy, runs in the worker.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

PROFILE_VERSION = 1
SECTION_TYPES = ("intro", "build", "drop", "breakdown", "outro")


def _round_bars(n: int) -> int:
    """Section lengths land on 4/8/16/32...: round to the nearest multiple of 4 (min 4)."""
    return max(4, int(round(n / 4.0)) * 4)


def _share(counter: Counter, top: int) -> dict:
    total = sum(counter.values()) or 1
    return {k: round(v / total, 4) for k, v in counter.most_common(top)}


def build_profile(analyses: list[dict]) -> dict:
    tempos, keys, durations = [], Counter(), []
    seq_counter: Counter = Counter()
    length_hist: dict[str, Counter] = {t: Counter() for t in SECTION_TYPES}
    transitions: dict[str, Counter] = {t: Counter() for t in SECTION_TYPES}
    patterns = {k: Counter() for k in ("kick", "snare", "hat", "bass")}
    bass_moves: Counter = Counter()
    drop_minus_break, vocal_tracks, demucs_tracks = [], 0, 0
    bars_per_track = []

    for a in analyses:
        if not a.get("bars"):
            continue
        tempos.append(float(a["bpm"]))
        durations.append(float(a.get("duration_s", 0)))
        bars_per_track.append(int(a["bars"]))
        if a.get("key") and a.get("key") != "?":
            keys[f"{a['key']} {a['mode']}"] += 1
        secs = a.get("sections") or []
        if secs:
            seq_counter[">".join(s["type"] for s in secs)] += 1
            for s in secs:
                length_hist[s["type"]][_round_bars(s["bars"])] += 1
            for s1, s2 in zip(secs, secs[1:]):
                transitions[s1["type"]][s2["type"]] += 1
            drops = [s["energy_db"] for s in secs if s["type"] == "drop"]
            breaks = [s["energy_db"] for s in secs if s["type"] == "breakdown"]
            if drops and breaks:
                drop_minus_break.append(float(np.mean(drops) - np.mean(breaks)))
        for k in patterns:
            for pat, n in (a.get(f"{k}_patterns") or {}).items():
                patterns[k][pat] += int(n)
        roots = a.get("bass_roots_rel") or []
        for i in range(0, len(roots) - 3, 4):
            chunk = roots[i:i + 4]
            if all(r is not None for r in chunk):
                bass_moves[",".join(str(r) for r in chunk)] += 1
        if a.get("vocal_segments"):
            vocal_tracks += 1
        if a.get("demucs"):
            demucs_tracks += 1

    n = len(tempos)
    if n == 0:
        return {"version": PROFILE_VERSION, "n_tracks": 0}

    t = np.asarray(tempos)
    hist_edges = np.arange(int(t.min() // 2) * 2, int(t.max() // 2) * 2 + 3, 2)
    counts, _ = np.histogram(t, bins=hist_edges)
    tempo_hist = {int(e): int(c) for e, c in zip(hist_edges[:-1], counts) if c}

    d = np.asarray(durations)
    dur_edges = np.arange(0, d.max() + 60, 60)
    dcounts, _ = np.histogram(d, bins=dur_edges)
    dur_hist = {int(e // 60): int(c) for e, c in zip(dur_edges[:-1], dcounts) if c}

    return {
        "version": PROFILE_VERSION,
        "n_tracks": n,
        "tracks_with_demucs": demucs_tracks,
        "tracks_with_vocals": vocal_tracks,
        "tempo": {
            "min": round(float(t.min()), 1), "max": round(float(t.max()), 1),
            "median": round(float(np.median(t)), 1), "mean": round(float(t.mean()), 1), "std": round(float(t.std()), 2),
            "hist": tempo_hist,
        },
        "duration": {"median_s": round(float(np.median(d)), 1), "hist_minutes": dur_hist,
                     "median_bars": int(np.median(bars_per_track))},
        "keys": dict(keys.most_common(24)),
        "sections": {
            "sequences": dict(seq_counter.most_common(20)),
            "lengths": {k: dict(sorted(v.items())) for k, v in length_hist.items() if v},
            "transitions": {k: _share(v, 5) for k, v in transitions.items() if v},
            "drop_minus_breakdown_db": round(float(np.mean(drop_minus_break)), 2) if drop_minus_break else None,
        },
        "patterns": {k: _share(v, 12) for k, v in patterns.items()},
        "bass_movement": _share(bass_moves, 16),
    }


def describe(profile: dict) -> str:
    """A few plain-English lines for the log / the About of the profile."""
    if not profile or not profile.get("n_tracks"):
        return "No analysed tracks yet."
    t = profile["tempo"]
    lines = [f"{profile['n_tracks']} tracks analysed ({profile.get('tracks_with_demucs', 0)} with stem separation).",
             f"Tempo {t['min']}-{t['max']} BPM, median {t['median']}."]
    seqs = profile["sections"].get("sequences") or {}
    if seqs:
        top = next(iter(seqs.items()))
        lines.append(f"Most common structure: {top[0]} ({top[1]} tracks).")
    kick = profile["patterns"].get("kick") or {}
    if kick:
        pat, share = next(iter(kick.items()))
        lines.append(f"Most common kick pattern: {pat} in {share * 100:.0f} % of bars.")
    keys = profile.get("keys") or {}
    if keys:
        lines.append("Keys: " + ", ".join(f"{k} ({v})" for k, v in list(keys.items())[:4]) + ".")
    return "\n".join(lines)
