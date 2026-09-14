"""Finding the audio files of a track / a CD / a collection, and reading their tags with ffprobe.

Standard library only (runs in the worker, but must stay importable anywhere).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from tmg import paths

AUDIO_EXTENSIONS = {".mp3", ".flac", ".wav", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".wma", ".aif", ".aiff", ".alac", ".ape"}

# Folder names that are never albums (mirrors, recycle bins, system folders).
SKIP_DIRS = {"$RECYCLE.BIN", "System Volume Information", ".Trash-1000", "__MACOSX", "@eaDir", ".thumbnails"}


def is_audio(path: str | Path) -> bool:
    return Path(path).suffix.lower() in AUDIO_EXTENSIONS


def scan_track(path: str) -> list[str]:
    """One file."""
    return [str(Path(path))] if os.path.isfile(path) and is_audio(path) else []


def scan_cd(folder: str) -> list[str]:
    """One folder = one CD: only the audio files directly inside it, no sub-folders."""
    try:
        names = sorted(os.listdir(folder))
    except OSError:
        return []
    return [str(Path(folder) / n) for n in names if is_audio(n) and os.path.isfile(os.path.join(folder, n))]


def scan_collection(folder: str) -> list[str]:
    """A folder with many folders (200 CDs = 200 folders): everything, recursively."""
    out: list[str] = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
        for n in sorted(files):
            if is_audio(n):
                out.append(str(Path(root) / n))
    return out


def scan(kind: str, root: str) -> list[str]:
    if kind == "track":
        return scan_track(root)
    if kind == "cd":
        return scan_cd(root)
    if kind == "collection":
        return scan_collection(root)
    raise ValueError(f"unknown import kind {kind!r}")


def guess_kind_for_drop(path: str) -> str:
    """Drag & drop has no buttons: a file is a track, a folder with audio sub-folders is a collection, else a CD."""
    if os.path.isfile(path):
        return "track"
    try:
        for name in os.listdir(path):
            sub = os.path.join(path, name)
            if os.path.isdir(sub) and name not in SKIP_DIRS and scan_cd(sub):
                return "collection"
    except OSError:
        pass
    return "cd"


def probe(path: str) -> dict:
    """Duration, sample rate, channels and tags via ffprobe (the static build inside tools/)."""
    cmd = [str(paths.FFPROBE), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False).stdout
        info = json.loads(out or "{}")
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        info = {}
    fmt = info.get("format") or {}
    tags = {k.lower(): v for k, v in (fmt.get("tags") or {}).items()}
    audio = next((s for s in info.get("streams") or [] if s.get("codec_type") == "audio"), {})
    for k, v in (audio.get("tags") or {}).items():
        tags.setdefault(k.lower(), v)
    p = Path(path)
    try:
        duration = float(fmt.get("duration")) if fmt.get("duration") else None
    except ValueError:
        duration = None
    return {
        "duration_s": duration,
        "sample_rate": int(audio.get("sample_rate") or 0) or None,
        "channels": int(audio.get("channels") or 0) or None,
        "title": (tags.get("title") or p.stem).strip(),
        "artist": (tags.get("artist") or tags.get("album_artist") or "").strip(),
        "album": (tags.get("album") or p.parent.name).strip(),
    }
