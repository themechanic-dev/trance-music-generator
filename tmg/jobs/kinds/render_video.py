"""Video job: an MP4 for one production - GPU visuals following the timeline, NVENC, the track's audio."""

from __future__ import annotations

import os
import shutil
import time

from tmg import db as dbmod
from tmg import paths
from tmg.jobs import protocol
from tmg.visuals import renderer, sequencer


def run(params: dict) -> dict:
    db = dbmod.Database()
    prod = db.get_production(int(params["production_id"]))
    if not prod:
        raise RuntimeError("production not found")
    audio = prod.get("wav_path") or prod.get("mp3_path")
    if not audio or not os.path.exists(audio):
        raise RuntimeError("the audio file of this production is missing")
    tl_path = prod.get("timeline_path")
    if not tl_path or not os.path.exists(tl_path):
        raise RuntimeError("the timeline file of this production is missing")
    import json

    timeline = json.load(open(tl_path, encoding="utf-8"))
    width, height = renderer.RESOLUTIONS.get(params.get("resolution", "1080p"), (1920, 1080))
    fps = int(params.get("fps", 30))
    codec = params.get("codec", "h264")
    bitrate = int(params.get("bitrate_k", 10000))
    enabled = {k: float(v) for k, v in (params.get("generators") or {}).items() if float(v) > 0}
    if not enabled:
        enabled = {name: spec[1] for name, spec in sequencer.GENERATORS.items() if name != "stills"}
    seed = int(params.get("seed") or prod["seed"])

    images_dir = paths.DATA / "images"
    if params.get("ai_stills") and enabled.get("stills", 0) > 0:
        from tmg.visuals import ai_stills

        ai_dir = paths.DATA / "ai_images" / str(prod["id"])
        try:
            protocol.progress(0.01, "AI stills (SD-Turbo)")
            ai_stills.generate(ai_dir, int(params.get("ai_count", 8)), seed,
                               progress=lambda f, m: protocol.progress(0.01 + 0.08 * f, m))
            images_dir = ai_dir
        except Exception as exc:  # noqa: BLE001 - no stills is not a reason to lose the video
            protocol.log(f"AI stills unavailable ({type(exc).__name__}: {exc}) - using data/images or other generators")

    stem = os.path.splitext(os.path.basename(prod.get("mp3_path") or prod.get("wav_path")))[0]
    out_path = str(paths.OUTPUT / f"{stem}.mp4")
    tmp_path = out_path + ".part.mp4"
    t0 = time.perf_counter()
    summary = renderer.render_video(audio, timeline, tmp_path, enabled=enabled, width=width, height=height, fps=fps,
                                    codec=codec, bitrate_k=bitrate, seed=seed, images_dir=images_dir,
                                    progress=lambda f, m: protocol.progress(0.1 + 0.88 * f, m))
    shutil.move(tmp_path, out_path)
    video = {"path": out_path, "size": summary["size"], "fps": fps, "codec": codec, "bitrate_k": bitrate,
             "seconds": summary["seconds"], "fps_achieved": summary["fps_achieved"], "shots": len(summary["shots"]),
             "images": summary["images"], "renderer": summary["renderer"], "generators": sorted(enabled),
             "shot_list": [{k: s[k] for k in ("generator", "start_s", "end_s", "section_type", "palette")} for s in summary["shots"]]}
    db.update_production_video(prod["id"], out_path, video)
    db.close()
    protocol.log(f"video {os.path.basename(out_path)}: {summary['frames']} frames in {summary['seconds']} s ({summary['fps_achieved']} fps) on {summary['renderer']}")
    return {"production_id": prod["id"], "mp4": out_path, "seconds": summary["seconds"], "fps_achieved": summary["fps_achieved"],
            "shots": len(summary["shots"]), "total_s": round(time.perf_counter() - t0, 1)}
