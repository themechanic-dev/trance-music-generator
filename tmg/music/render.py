"""From a plan to files: WAV and/or tagged MP3 in data/output, plus the timeline as JSON."""

from __future__ import annotations

import json
import os
import subprocess
import wave
from pathlib import Path

import numpy as np

from tmg import paths
from tmg.music.synth import SAMPLE_RATE

_FIRST = ("Aurora", "Cascade", "Ember", "Halcyon", "Lucid", "Meridian", "Nova", "Obsidian", "Parallax", "Quartz",
          "Solstice", "Tundra", "Umbra", "Velvet", "Zenith", "Cobalt", "Drifting", "Endless", "Fathom", "Glacier",
          "Horizon", "Ionosphere", "Kelvin")
_SECOND = ("Ascent", "Bloom", "Current", "Descent", "Echo", "Field", "Gate", "Hollow", "Interval", "Journey",
           "Kinetic", "Lantern", "Mirage", "Northern", "Orbit", "Passage", "Quiet", "Return", "Signal", "Threshold",
           "Undertow", "Vantage")


def name_for(seed: int) -> str:
    rng = np.random.default_rng(seed)
    return f"{_FIRST[rng.integers(len(_FIRST))]} {_SECOND[rng.integers(len(_SECOND))]}"


def write_wav(path: str, audio: np.ndarray) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes((np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())


def encode_mp3(wav_path: str, mp3_path: str, *, bitrate_k: int, tags: dict[str, str]) -> None:
    argv = [str(paths.FFMPEG), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", wav_path,
            "-c:a", "libmp3lame", "-b:a", f"{bitrate_k}k", "-id3v2_version", "3"]
    for k, v in tags.items():
        argv += ["-metadata", f"{k}={v}"]
    argv.append(mp3_path)
    r = subprocess.run(argv, capture_output=True, text=True, check=False)
    if r.returncode != 0 or not os.path.isfile(mp3_path):
        raise RuntimeError(f"ffmpeg could not encode the track: {r.stderr.strip()[:400]}")


def write_outputs(audio: np.ndarray, plan, timeline: dict, *, formats: tuple[str, ...] = ("mp3",), bitrate_k: int = 192,
                  out_dir: Path | None = None, suffix: str = "") -> dict:
    """Write the files for one production and return their paths. `suffix` names a variant (e.g. '-dry')."""
    out_dir = out_dir or paths.OUTPUT
    out_dir.mkdir(parents=True, exist_ok=True)
    title = name_for(plan.seed)
    stem = f"{plan.seed:08d}-{title.lower().replace(' ', '-')}{suffix}"
    wav_path = out_dir / f"{stem}.wav"
    mp3_path = out_dir / f"{stem}.mp3"
    timeline_path = out_dir / f"{stem}.timeline.json"
    plan_path = out_dir / f"{stem}.plan.json"
    write_wav(str(wav_path), audio)
    result = {"title": title, "wav": None, "mp3": None, "timeline": str(timeline_path), "plan": str(plan_path)}
    if "mp3" in formats:
        encode_mp3(str(wav_path), str(mp3_path), bitrate_k=bitrate_k, tags={
            "title": title, "artist": "Trance Music Generator", "album": f"Generated · {plan.flavor.name}",
            "genre": "Trance", "TBPM": f"{plan.bpm:.0f}",
            "comment": f"{plan.flavor.name} · {plan.bpm:.1f} BPM · {plan.key_name} · seed {plan.seed} · {plan.structure}",
        })
        result["mp3"] = str(mp3_path)
    if "wav" in formats:
        result["wav"] = str(wav_path)
    else:
        os.remove(wav_path)
    with open(timeline_path, "w", encoding="utf-8") as f:
        json.dump(timeline, f)
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan.to_dict(), f, indent=1)
    return result


__all__ = ["encode_mp3", "name_for", "write_outputs", "write_wav"]
