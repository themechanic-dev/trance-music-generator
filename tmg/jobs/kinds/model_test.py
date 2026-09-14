"""Neural model self-test: is the chosen model downloaded, is it reachable, does it load on the GPU and
does it produce sound? Loads the model, generates a short clip and reports load time, clip time and
peak GPU memory. The clip is written to data/test/model-test/ so it can be listened to; nothing else changes.
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from pathlib import Path

from tmg import paths
from tmg.jobs import protocol

TEST_SECONDS = 8.0
PROMPT = "uplifting trance synth texture, 138 bpm, A minor, shimmering pad, wide stereo, no drums, no vocals"
SILENCE_DB = -45.0


def cache_info(model_id: str, hub: Path | None = None) -> tuple[bool, float]:
    """(downloaded, size in GB) for one HuggingFace repo in the project's model cache.

    Counts blobs only (snapshot entries are symlinks to them); a repo with '.incomplete' blobs or without a
    snapshot folder is a partial download, not a downloaded model.
    """
    hub = hub if hub is not None else paths.HF_HOME / "hub"
    d = hub / ("models--" + model_id.replace("/", "--"))
    if not d.exists():
        return False, 0.0
    files = [f for f in d.rglob("*") if f.is_file() and not f.is_symlink()]
    size = sum(f.stat().st_size for f in files)
    snapshots = d / "snapshots"
    partial = any(f.name.endswith(".incomplete") for f in files)
    complete = snapshots.exists() and any(snapshots.iterdir()) and not partial and size > 0
    return complete, round(size / 2**30, 2)


def _access(model_id: str) -> str:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import GatedRepoError

    try:
        HfApi().auth_check(model_id)
        return "ok"
    except GatedRepoError:
        return "gated - accept access on the model page and save a Read token (Settings > HuggingFace)"
    except Exception as exc:  # noqa: BLE001 - offline, DNS, whatever: reported, not fatal when the files are here
        return f"unreachable ({type(exc).__name__})"


def summary(r: dict) -> str:
    """Multi-line text for the Settings row and the log."""
    lines = [f"{r.get('model')}: {r.get('model_id')}"]
    state = f"downloaded ({r.get('size_gb')} GB in models/hf)" if r.get("downloaded") else "not downloaded"
    lines.append(f"{state} · access {r.get('access', '?')}")
    if r.get("working"):
        lines.append(f"WORKING: loaded in {r.get('load_s'):.0f} s, {r.get('seconds'):.0f} s clip in {r.get('gen_s'):.0f} s, "
                     f"peak VRAM {r.get('peak_vram_gb'):.1f} GB, level {r.get('rms_db'):.0f} dBFS")
    else:
        lines.append(f"NOT WORKING: {r.get('error')}")
    when = r.get("tested_utc", "")
    tail = f"tested {when[:16].replace('T', ' ')} UTC" if when else ""
    if r.get("wav"):
        tail += f" · {Path(r['wav']).relative_to(paths.ROOT) if str(r['wav']).startswith(str(paths.ROOT)) else r['wav']}"
    if tail:
        lines.append(tail)
    return "\n".join(lines)


def run(params: dict) -> dict:
    import numpy as np

    from tmg.music import neural
    from tmg.music.synth import SAMPLE_RATE

    key = params.get("model") or "stereo-small"
    if key not in neural.MODELS:
        raise ValueError(f"unknown neural model '{key}'")
    seconds = float(params.get("seconds") or TEST_SECONDS)
    spec = neural.MODELS[key]
    model_id = spec["id"]
    result: dict = {"model": key, "model_id": model_id, "seconds": seconds, "working": False,
                    "tested_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    protocol.progress(0.05, f"{model_id}: checking the downloaded files")
    downloaded, size_gb = cache_info(model_id)
    result.update(downloaded=downloaded, size_gb=size_gb)
    protocol.progress(0.10, f"{model_id}: checking access on HuggingFace")
    access = _access(model_id)
    result["access"] = access
    if not downloaded and access != "ok":
        raise RuntimeError(f"{model_id} is not downloaded and cannot be fetched: {access}")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available - the neural models need the GPU")
    torch.cuda.reset_peak_memory_stats()
    protocol.progress(0.20, f"loading {model_id} on the GPU" + ("" if downloaded else " (downloading it first - this can take a while)"))
    t0 = time.perf_counter()
    neural.ensure_loaded(key)
    load_s = time.perf_counter() - t0
    protocol.log(f"model test {key}: loaded {model_id} in {load_s:.1f} s")

    protocol.progress(0.60, f"generating a {seconds:.0f} s test clip")
    out_dir = paths.DATA / "test" / "model-test"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob(f"{key}-*.wav"):     # always a fresh generation, never the cache
        old.unlink()
    audio, info = neural.generate(key, PROMPT, seconds, seed=1, cache_dir=out_dir, log=protocol.log)

    protocol.progress(0.95, "checking the audio")
    if not np.isfinite(audio).all():
        raise RuntimeError("the clip contains NaN or infinite samples")
    rms_db = 20 * math.log10(float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) + 1e-9)
    duration = audio.shape[1] / SAMPLE_RATE
    if rms_db < SILENCE_DB:
        raise RuntimeError(f"the clip is silent ({rms_db:.0f} dBFS)")
    if duration < seconds * 0.8:
        raise RuntimeError(f"the clip is too short ({duration:.1f} s of {seconds:.0f} s)")
    vram = torch.cuda.max_memory_allocated() / 2**30
    neural.unload()
    downloaded, size_gb = cache_info(model_id)   # a first run has just downloaded it
    result.update(downloaded=downloaded, size_gb=size_gb, load_s=round(load_s, 1), gen_s=float(info.get("gen_s", 0.0)),
                  peak_vram_gb=round(vram, 2), rms_db=round(rms_db, 1), duration_s=round(duration, 2),
                  wav=info.get("path"), working=True)
    protocol.log("model test: " + summary(result).replace("\n", " | "))
    protocol.progress(1.0, "done")
    return result
