"""Colour palettes: JSON in, a 256-entry lookup table out (uploaded to the GPU as a 256x1 texture)."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tmg import paths

LUT_SIZE = 256

DEFAULT_PALETTES = [
    {"id": "ocean", "name": "Ocean", "colors": ["#02040f", "#062044", "#0b4f8a", "#18a3c9", "#5ee7f0", "#c8fbff"]},
    {"id": "ember", "name": "Ember", "colors": ["#0a0202", "#3b0a0a", "#8a1e0b", "#d9541c", "#f7a93b", "#fff3c4"]},
    {"id": "violet", "name": "Violet", "colors": ["#05010c", "#1c0b3a", "#4b1a8a", "#8e3ccf", "#d484ff", "#f7e3ff"]},
    {"id": "acid", "name": "Acid", "colors": ["#020a04", "#0a3d1c", "#1f8a2e", "#6ad13a", "#d6ff5c", "#f4ffd6"]},
    {"id": "sunset", "name": "Sunset", "colors": ["#0b0413", "#3a0f4d", "#a11d5c", "#f0553e", "#ffb547", "#fff0c2"]},
    {"id": "ice", "name": "Ice", "colors": ["#02050c", "#0d2540", "#2a5f8f", "#63a5d6", "#b3e0ff", "#ffffff"]},
    {"id": "gold", "name": "Gold", "colors": ["#0a0704", "#3a2405", "#8a5a0b", "#d9a11c", "#ffd86b", "#fff6d5"]},
    {"id": "magenta", "name": "Magenta", "colors": ["#0c0210", "#3d0a45", "#8a1e8f", "#e03ccf", "#ff8ce6", "#ffe3fb"]},
]


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    text = value.strip().lstrip("#")
    if len(text) == 3:
        text = "".join(c * 2 for c in text)
    return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)


@dataclass(frozen=True)
class Palette:
    id: str
    name: str
    colors: tuple[str, ...]
    weight: float = 1.0

    def lut(self) -> np.ndarray:
        stops = np.array([hex_to_rgb(c) for c in self.colors], dtype=np.float32)
        if len(stops) == 1:
            return np.repeat(stops.astype(np.uint8), LUT_SIZE, axis=0)
        positions = np.linspace(0.0, 1.0, len(stops))
        targets = np.linspace(0.0, 1.0, LUT_SIZE)
        channels = [np.interp(targets, positions, stops[:, i]) for i in range(3)]
        return np.clip(np.stack(channels, axis=1), 0, 255).astype(np.uint8)


def ensure_palette_file(path: Path | None = None) -> Path:
    """data/palettes.json - written with the defaults the first time so the user can edit it."""
    path = path or paths.DATA / "palettes.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"palettes": DEFAULT_PALETTES}, indent=2), encoding="utf-8")
    return path


def load_palettes(path: Path | None = None) -> list[Palette]:
    path = ensure_palette_file(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {"palettes": DEFAULT_PALETTES}
    out = []
    for e in data.get("palettes", []):
        if not e.get("enabled", True):
            continue
        colors = tuple(e.get("colors") or ())
        if len(colors) < 2:
            continue
        try:
            for c in colors:
                hex_to_rgb(c)
        except ValueError:
            continue
        out.append(Palette(str(e.get("id") or e.get("name")), str(e.get("name") or e.get("id")), colors, float(e.get("weight", 1.0))))
    if not out:
        out = [Palette(p["id"], p["name"], tuple(p["colors"])) for p in DEFAULT_PALETTES]
    return out


def choose(palettes: list[Palette], rng: random.Random) -> Palette:
    weights = [max(0.0, p.weight) for p in palettes]
    return rng.choices(palettes, weights=weights, k=1)[0] if sum(weights) > 0 else rng.choice(palettes)


__all__ = ["DEFAULT_PALETTES", "LUT_SIZE", "Palette", "choose", "ensure_palette_file", "hex_to_rgb", "load_palettes"]
