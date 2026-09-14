"""The shot list: which generator plays when, from the timeline and the user's on/off choices.

Calm generators in the intro, the breakdowns and the outro; medium ones in the builds; intense ones in the
drops. A shot changes on a section boundary and, inside long sections, every 16 bars on a downbeat -
so a 64-bar drop is not one picture for two and a half minutes. Drop hits are hard cuts; everything else
crossfades over one bar.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from tmg.visuals.signals import SECTION_MOOD

# name -> (mood, default weight, description)
GENERATORS: dict[str, tuple[str, float, str]] = {
    "plasma": ("intense", 1.0, "Interfering sine fields (the demoscene classic)"),
    "waves": ("medium", 1.0, "Moire interference between drifting plane waves"),
    "tunnel": ("intense", 1.2, "Endless zoom down a textured corridor"),
    "domainwarp": ("calm", 1.2, "Marbled liquid fields, noise warped by noise"),
    "flow": ("medium", 1.0, "Trails drifting through a flow field (feedback)"),
    "reaction": ("calm", 0.8, "Gray-Scott reaction-diffusion, slowly alive"),
    "kaleidoscope": ("intense", 1.0, "Mirror-folded noise, symmetric and spinning"),
    "julia": ("intense", 1.0, "Julia set orbiting, colour by escape time"),
    "mandelbulb": ("intense", 0.8, "Raymarched 3D fractal turning in the dark"),
    "menger": ("medium", 0.7, "Raymarched Menger sponge corridor"),
    "metaballs": ("medium", 0.9, "Merging blobs of light"),
    "particles": ("medium", 1.0, "A field of glowing particles drifting with the beat"),
    "spectrum": ("intense", 0.9, "Rings and bars driven by the real audio spectrum"),
    "stills": ("calm", 1.0, "Your images (or AI stills) with a slow camera and a liquid warp"),
}

SHOT_BARS = 16          # inside a long section, change the picture every this many bars
MIN_SHOT_BARS = 8


@dataclass
class Shot:
    generator: str
    start_s: float
    end_s: float
    section: int
    section_type: str
    palette: str
    seed: int
    cut: bool           # hard cut (drop hit) instead of a crossfade
    intensity: float    # 0..1, how hard the generator should push

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def enabled_by_mood(enabled: dict[str, float]) -> dict[str, list[tuple[str, float]]]:
    out: dict[str, list[tuple[str, float]]] = {"calm": [], "medium": [], "intense": []}
    for name, weight in enabled.items():
        if name in GENERATORS and weight > 0:
            out[GENERATORS[name][0]].append((name, float(weight)))
    return out


def _pick(rng: random.Random, pool: list[tuple[str, float]], avoid: str | None) -> str:
    choices = [c for c in pool if c[0] != avoid] or pool
    names, weights = zip(*choices)
    return rng.choices(names, weights=weights, k=1)[0]


def build_shots(timeline: dict, enabled: dict[str, float], palettes: list[str], seed: int) -> list[Shot]:
    """A list of shots that cover the whole timeline, given the enabled generators (name -> weight)."""
    rng = random.Random(seed)
    by_mood = enabled_by_mood(enabled)
    # a mood with nothing enabled borrows from its neighbours
    fallback_order = {"calm": ("medium", "intense"), "medium": ("calm", "intense"), "intense": ("medium", "calm")}
    for mood, neighbours in fallback_order.items():
        if not by_mood[mood]:
            for nb in neighbours:
                if by_mood[nb]:
                    by_mood[mood] = list(by_mood[nb])
                    break
    if not any(by_mood.values()):
        by_mood = {m: [("plasma", 1.0)] for m in by_mood}
    bar_s = float(timeline["bar_s"])
    shots: list[Shot] = []
    last: str | None = None
    palette = rng.choice(palettes) if palettes else "ocean"
    for i, s in enumerate(timeline.get("sections") or []):
        kind = s["type"]
        mood = SECTION_MOOD.get(kind, "medium")
        start, bars = float(s["start_s"]), int(s["bars"])
        if kind in ("drop", "build") or rng.random() < 0.5:
            palette = rng.choice(palettes) if palettes else palette
        # split long sections into shots on downbeats
        pieces = max(1, bars // SHOT_BARS) if bars >= SHOT_BARS + MIN_SHOT_BARS else 1
        per = bars / pieces
        for k in range(pieces):
            a = start + k * per * bar_s
            b = start + (k + 1) * per * bar_s if k < pieces - 1 else float(s["end_s"])
            gen = _pick(rng, by_mood[mood], last)
            intensity = {"calm": 0.35, "medium": 0.65, "intense": 1.0}[mood]
            shots.append(Shot(gen, round(a, 4), round(b, 4), i, kind, palette, rng.randrange(1 << 30),
                              cut=(kind == "drop" and k == 0), intensity=intensity))
            last = gen
    return shots


__all__ = ["GENERATORS", "MIN_SHOT_BARS", "SHOT_BARS", "Shot", "build_shots", "enabled_by_mood"]
