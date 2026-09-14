"""Every path the project uses. Everything lives inside the project folder - nothing system-wide.

This module runs both in the GUI (system Python 3.14) and in the workers (.venv, Python 3.12),
so it uses the standard library only.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(os.environ.get("TMG_ROOT") or Path(__file__).resolve().parent.parent)

DATA = ROOT / "data"
PHRASES = DATA / "phrases"      # the phrase bank (WAV)
TRASH = DATA / "trash"          # deleted phrases go here - nothing is lost silently
LOGS = DATA / "logs"
TMP = DATA / "tmp"
OUTPUT = DATA / "output"        # productions (MP3/MP4) - phase 3+
SETTINGS_FILE = DATA / "settings.json"
DB_FILE = DATA / "tmg.sqlite"

MODELS = ROOT / "models"
HF_HOME = MODELS / "hf"
TORCH_HOME = MODELS / "torch"
HF_TOKEN_FILE = HF_HOME / "token"   # huggingface_hub reads the token here when HF_HOME=models/hf
TOOLS = ROOT / "tools"
CACHE = TOOLS / "cache"

VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
FFMPEG = Path(os.environ.get("TMG_FFMPEG") or TOOLS / "ffmpeg" / "bin" / "ffmpeg")
FFPROBE = Path(os.environ.get("TMG_FFPROBE") or TOOLS / "ffmpeg" / "bin" / "ffprobe")


def ensure_dirs() -> None:
    for d in (DATA, PHRASES, TRASH, LOGS, TMP, OUTPUT, MODELS, HF_HOME, TORCH_HOME, CACHE):
        d.mkdir(parents=True, exist_ok=True)


def worker_env() -> dict[str, str]:
    """Environment every worker subprocess starts with."""
    env = dict(os.environ)
    env.update(
        {
            "TMG_ROOT": str(ROOT),
            "HF_HOME": str(HF_HOME),
            "TORCH_HOME": str(TORCH_HOME),
            "XDG_CACHE_HOME": str(CACHE),
            "TMG_FFMPEG": str(FFMPEG),
            "TMG_FFPROBE": str(FFPROBE),
            "PYTHONPATH": str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PATH": str(TOOLS / "ffmpeg" / "bin") + os.pathsep + env.get("PATH", ""),
        }
    )
    return env


def which(name: str) -> str | None:
    return shutil.which(name)


def disk_free_gb(path: Path = ROOT) -> float:
    return shutil.disk_usage(path).free / 2**30
