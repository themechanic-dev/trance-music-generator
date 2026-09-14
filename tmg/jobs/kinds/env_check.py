"""Environment check: Python, CUDA, ffmpeg/NVENC (real encode), libraries, models, disk."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import tempfile
from importlib.metadata import version as _dist_version

from tmg import paths
from tmg.jobs import protocol


def _ffmpeg_check() -> dict:
    ff = str(paths.FFMPEG)
    info: dict = {"path": ff, "present": os.path.exists(ff)}
    if not info["present"]:
        return info
    out = subprocess.run([ff, "-version"], capture_output=True, text=True, check=False).stdout
    info["version"] = out.splitlines()[0].split(" Copyright")[0] if out else "?"
    # The only honest NVENC check is a real encode (lesson from AutoDJ: `-encoders` lists it without a GPU).
    with tempfile.TemporaryDirectory() as td:
        dst = os.path.join(td, "t.mp4")
        r = subprocess.run(
            [ff, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30", "-t", "0.5",
             "-c:v", "h264_nvenc", dst],
            capture_output=True, text=True, check=False,
        )
        info["nvenc_h264"] = r.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 0
        info["nvenc_error"] = r.stderr.strip()[-300:] if r.returncode != 0 else ""
    return info


def _models() -> list[dict]:
    hub = paths.HF_HOME / "hub"
    out = []
    if hub.exists():
        for d in sorted(hub.glob("models--*")):
            # blobs only: the snapshot entries are symlinks to them and would count twice
            size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file() and not f.is_symlink())
            out.append({"name": d.name.replace("models--", "").replace("--", "/"), "size_gb": round(size / 2**30, 2)})
    return out


def run(params: dict) -> dict:
    protocol.progress(0.05, "Python")
    result: dict = {
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "root": str(paths.ROOT),
        "disk_free_gb": round(paths.disk_free_gb(), 1),
    }
    protocol.progress(0.2, "PyTorch / CUDA")
    try:
        import torch

        cuda = torch.cuda.is_available()
        t: dict = {"version": torch.__version__, "cuda_available": cuda}
        if cuda:
            p = torch.cuda.get_device_properties(0)
            free, total = torch.cuda.mem_get_info()
            t.update({"device": p.name, "vram_total_gb": round(p.total_memory / 2**30, 2), "vram_free_gb": round(free / 2**30, 2)})
        result["torch"] = t
    except Exception as exc:  # noqa: BLE001
        result["torch"] = {"error": f"{type(exc).__name__}: {exc}"}
    protocol.progress(0.5, "ffmpeg + NVENC (real encode)")
    result["ffmpeg"] = _ffmpeg_check()
    protocol.progress(0.75, "libraries")
    libs = {}
    for name in ("demucs", "transformers", "diffusers", "librosa", "moderngl", "soundfile", "numpy"):
        try:
            libs[name] = _dist_version(name)
        except Exception as exc:  # noqa: BLE001
            libs[name] = f"MISSING ({type(exc).__name__})"
    result["libs"] = libs
    protocol.progress(0.9, "models")
    result["models"] = _models()
    result["hf_token_present"] = paths.HF_TOKEN_FILE.exists()
    result["pipewire_tools"] = {t: bool(paths.which(t)) for t in ("pw-record", "pw-play", "pw-dump", "pw-metadata")}
    protocol.progress(1.0, "done")
    return result
