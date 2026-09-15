"""Analysis job: every queued track, one after the other, written to the database as soon as it is done.

Resumable: a crash or reboot leaves tracks 'queued'/'analyzing' and the next run picks them up. The
Demucs model is loaded once; librosa's numba JIT is paid once - which is why this is one long job
rather than one process per track (measured in phase 0: 23 s of JIT).
"""

from __future__ import annotations

import os
import random
import re
import time

import numpy as np

from tmg import db as dbmod
from tmg import paths
from tmg.jobs import protocol
from tmg.library import analysis, profile


def _phrase_id(track_id: int, n: int) -> str:
    return f"lib-{track_id}-{n}-" + "".join(random.choice("0123456789abcdefghijklmnopqrstuvwxyz") for _ in range(3))


def _safe(name: str) -> str:
    return re.sub(r"[^\w\-. ]+", "_", name)[:60]


def _save_vocal_phrases(db: dbmod.Database, track: dict, stems: dict, segments: list[dict], max_count: int) -> int:
    import soundfile as sf

    existing = db.count_phrases_for_track(track["id"])
    if existing >= max_count:
        return 0
    vocals = stems["vocals"]
    paths.PHRASES.mkdir(parents=True, exist_ok=True)
    saved = 0
    for n, seg in enumerate(segments[: max_count - existing], start=existing + 1):
        a, b = int(seg["start"] * analysis.SR), int(seg["end"] * analysis.SR)
        clip = vocals[:, a:b]
        peak = float(np.abs(clip).max()) if clip.size else 0.0
        if peak < 1e-4:
            continue
        clip = clip * (10 ** (-1.0 / 20.0) / peak)
        pid = _phrase_id(track["id"], n)
        out = paths.PHRASES / f"{pid}.wav"
        sf.write(str(out), clip.T, analysis.SR, format="WAV", subtype="PCM_16")
        name = f"{track.get('title') or os.path.basename(track['path'])} · phrase {n}"
        db.add_phrase_full(pid, name, str(out), "library", "ready", round((b - a) / analysis.SR, 3), analysis.SR, 2, -1.0, track["id"])
        saved += 1
    return saved


def _keep_stems(track: dict, stems: dict) -> str:
    out_dir = paths.DATA / "library" / "stems" / str(track["id"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, wav in stems.items():
        analysis.write_mp3(wav, analysis.SR, str(out_dir / f"{name}.mp3"))
    return str(out_dir)


def run(params: dict) -> dict:
    use_demucs = bool(params.get("demucs", True))
    keep_first = int(params.get("keep_stems_first", 3))
    vocal_opts = {
        "enabled": bool(params.get("extract_vocals", True)),
        "threshold_db": float(params.get("vocal_threshold_db", -35.0)),
        "min_len": float(params.get("vocal_min_len", 0.8)),
        "max_len": float(params.get("vocal_max_len", 8.0)),
        "max_count": int(params.get("vocal_max_count", 3)),
    }
    db = dbmod.Database()
    todo = db.tracks_to_analyze()
    if not todo:
        prof = profile.build_profile(db.analyses())
        db.save_profile(prof)
        db.close()
        protocol.log("nothing to analyse - profile rebuilt")
        return {"analyzed": 0, "failed": 0, "profile_tracks": prof.get("n_tracks", 0)}

    model = None
    if use_demucs:
        protocol.progress(0.0, "loading Demucs")
        model = analysis.load_demucs("htdemucs")
    kept_this_run = 0
    done = failed = 0
    started = time.perf_counter()
    for i, t in enumerate(todo):
        title = t.get("title") or os.path.basename(t["path"])
        elapsed = time.perf_counter() - started
        eta = (elapsed / i * (len(todo) - i)) if i else 0.0
        protocol.progress(i / len(todo), f"{i + 1}/{len(todo)} · {title}" + (f" · ~{eta / 60:.0f} min left" if i >= 2 else ""))
        db.update_track(t["id"], status="analyzing", error=None)
        try:
            if not os.path.exists(t["path"]):
                raise FileNotFoundError(t["path"])
            want_stems = kept_this_run < keep_first
            result, stems = analysis.analyze(t["path"], demucs_model=model, want_stems=want_stems, vocal_opts=vocal_opts)
            stems_dir = None
            phrases_saved = 0
            if stems is not None:
                if want_stems:
                    stems_dir = _keep_stems(t, stems)
                    kept_this_run += 1
                if result["vocal_segments"] and vocal_opts["enabled"]:
                    phrases_saved = _save_vocal_phrases(db, t, stems, result["vocal_segments"], vocal_opts["max_count"])
            db.update_track(
                t["id"], status="done", analysis=result, bpm=result["bpm"], key_name=f"{result['key']} {result['mode']}",
                n_bars=result["bars"], duration_s=result["duration_s"], stems_dir=stems_dir, analyzed_utc=dbmod.utc_now(),
            )
            done += 1
            protocol.emit("track_done", track_id=t["id"], bpm=result["bpm"], key=result["key"], bars=result["bars"],
                          seconds=result["timing_s"]["total"], phrases=phrases_saved)
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the other 1999
            failed += 1
            db.update_track(t["id"], status="failed", error=f"{type(exc).__name__}: {exc}"[:500])
            protocol.log(f"FAILED {title}: {type(exc).__name__}: {exc}")
        if model is not None and (i + 1) % 25 == 0:
            import torch

            torch.cuda.empty_cache()
    protocol.progress(0.99, "building the style profile")
    prof = profile.build_profile(db.analyses())
    db.save_profile(prof)
    protocol.log(profile.describe(prof))
    db.close()
    return {"analyzed": done, "failed": failed, "profile_tracks": prof.get("n_tracks", 0),
            "minutes": round((time.perf_counter() - started) / 60, 1)}
