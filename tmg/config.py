"""Settings: JSON in data/settings.json with defaults and deep merge.

Settings for features that do not exist yet (music, visuals, export) are added phase by phase -
only what does something today lives here.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
import threading
from typing import Any

from tmg import paths

DEFAULTS: dict[str, Any] = {
    "version": 1,
    "capture": {
        "sink": "default",            # "default" or the node.name of a specific sink
        "rate": 48000,
        "channels": 2,
        "auto_trim": True,            # cut leading/trailing silence after Stop
        "trim_threshold_db": -45.0,
        "trim_pad_ms": 120,
        "normalize": True,            # peak normalisation
        "normalize_peak_db": -1.0,
    },
    "library": {
        "auto_analyze": True,         # start analysing right after an import
        "demucs": True,               # separate stems on the GPU (off = faster, rougher patterns)
        "keep_stems_first": 3,        # keep the stems (MP3) of the first N tracks of each run, to listen
        "extract_vocals": True,       # save vocal segments into the phrase bank
        "vocal_max_count": 3,
        "vocal_threshold_db": -35.0,
        "last_folder": "",
    },
    "compose": {"count": 5, "length_index": 0, "flavor_index": 0, "format_index": 0, "bitrate": 192},
    "phrases": {                      # phrases from the bank inside the music (phase 4)
        "enabled": True,
        "count": 2,                   # per track
        "where_index": 0,             # 0 = breakdowns + before drops, 1 = breakdowns only, 2 = before drops only
        "intro": False,
        "source_index": 0,            # 0 = captured, 1 = library, 2 = all, 3 = selected
        "ids": [],
        "level_db": -6.0,
        "telephone": False,
        "echo": 2,
        "fit": True,
    },
    "neural": {                       # MusicGen textures under the arrangement (phase 6)
        "enabled": False,
        "model": "stereo-small",      # stereo-small | medium | melody
        "level_db": -10.0,
        "energy": True,               # also an energy layer in builds/drops (else atmosphere only)
        "ab": True,                   # keep a dry copy next to the track for A/B listening
    },
    "video": {
        "resolution": "1080p",        # 720p | 1080p | 1440p | 2160p
        "fps": 30,
        "codec": "h264",              # h264 (NVENC) | hevc (NVENC) | x264 (software fallback)
        "bitrate_k": 10000,
        "ai_stills": False,           # SD-Turbo images for the 'stills' generator (downloads ~2.5 GB once)
        "ai_count": 8,
        "generators": {},             # name -> weight (0 = off); empty = every generator except stills at default weight
    },
    "ui": {
        "window_width": 1100,
        "window_height": 760,
    },
}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


class Settings:
    def __init__(self, path=None):
        self.path = path or paths.SETTINGS_FILE
        self._lock = threading.Lock()
        self.data: dict[str, Any] = copy.deepcopy(DEFAULTS)
        self.load()

    def load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as f:
                stored = json.load(f)
            if isinstance(stored, dict):
                self.data = _merge(DEFAULTS, stored)
        except FileNotFoundError:
            pass
        except (OSError, json.JSONDecodeError):
            # Corrupt file: keep defaults, keep the broken copy for diagnosis.
            try:
                os.replace(self.path, str(self.path) + ".broken")
            except OSError:
                pass

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".settings-", suffix=".json")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted: str, value: Any, save: bool = True) -> None:
        parts = dotted.split(".")
        node = self.data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
        if save:
            self.save()
