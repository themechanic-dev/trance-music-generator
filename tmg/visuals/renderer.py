"""From a production (audio + timeline) to an MP4: signals, spectrum, shots, GPU frames, NVENC via ffmpeg."""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from tmg import paths
from tmg.visuals import audio_features, sequencer, signals
from tmg.visuals import palettes as palmod
from tmg.visuals.engine import Engine

RESOLUTIONS = {"720p": (1280, 720), "1080p": (1920, 1080), "1440p": (2560, 1440), "2160p": (3840, 2160)}
CROSSFADE_BARS = 1.0


def load_images(folder: Path, width: int, height: int, limit: int = 24) -> list[np.ndarray]:
    """User images (data/images) or AI stills, cover-fitted to the frame size, as uint8 RGB arrays."""
    out: list[np.ndarray] = []
    if not folder.is_dir():
        return out
    try:
        from PIL import Image
    except ImportError:
        return out
    for p in sorted(folder.iterdir())[:limit * 3]:
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
            continue
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                sr, tr = im.width / im.height, width / height
                if sr > tr:
                    nh, nw = height, int(height * sr)
                else:
                    nw, nh = width, int(width / sr)
                im = im.resize((max(1, nw), max(1, nh)), Image.LANCZOS)
                left, top = (im.width - width) // 2, (im.height - height) // 2
                im = im.crop((left, top, left + width, top + height))
                out.append(np.asarray(im, dtype=np.uint8)[::-1].copy())   # bottom-up for GL
        except Exception:  # noqa: BLE001
            continue
        if len(out) >= limit:
            break
    return out


def ffmpeg_command(out_path: str, audio_path: str, width: int, height: int, fps: int, codec: str, bitrate_k: int) -> list[str]:
    enc = {"h264": "h264_nvenc", "hevc": "hevc_nvenc", "x264": "libx264"}.get(codec, "h264_nvenc")
    cmd = [str(paths.FFMPEG), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
           "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{width}x{height}", "-framerate", str(fps), "-i", "pipe:0",
           "-i", audio_path, "-map", "0:v", "-map", "1:a",
           "-c:v", enc, "-pix_fmt", "yuv420p", "-r", str(fps), "-g", str(fps * 2)]
    if enc.endswith("_nvenc"):
        cmd += ["-preset", "p5", "-rc", "vbr", "-b:v", f"{bitrate_k}k", "-maxrate", f"{int(bitrate_k * 1.5)}k", "-bufsize", f"{bitrate_k * 2}k"]
        if enc == "hevc_nvenc":
            cmd += ["-tag:v", "hvc1"]
    else:
        cmd += ["-preset", "veryfast", "-crf", "20"]
    cmd += ["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", out_path]
    return cmd


def render_video(audio_path: str, timeline: dict, out_path: str, *, enabled: dict[str, float], width: int = 1920,
                 height: int = 1080, fps: int = 30, codec: str = "h264", bitrate_k: int = 10000, seed: int = 0,
                 images_dir: Path | None = None, progress: Callable[[float, str], None] | None = None,
                 max_seconds: float | None = None) -> dict:
    """Render the whole video. Returns a summary dict (shots, frames, seconds, fps achieved)."""
    t0 = time.perf_counter()
    duration = float(timeline["duration_s"])
    if max_seconds:
        duration = min(duration, max_seconds)
    n_frames = int(duration * fps)
    sig = signals.build_signals(timeline, fps)
    if progress:
        progress(0.02, "analysing the audio spectrum")
    spec, rms = audio_features.features_for(audio_path, fps, n_frames)
    pals = palmod.load_palettes()
    luts = {p.id: p.lut() for p in pals}
    palette_ids = [p.id for p in pals] or ["fallback"]
    shots = sequencer.build_shots(timeline, enabled, palette_ids, seed)
    images = load_images(images_dir or (paths.DATA / "images"), width, height) if enabled.get("stills", 0) > 0 else []
    if progress:
        progress(0.04, f"{len(shots)} shots, {len(images)} images, starting the GPU")
    engine = Engine(width, height, luts, images)
    bar_s = float(timeline["bar_s"])
    fade_frames = max(1, int(CROSSFADE_BARS * bar_s * fps))
    cmd = ffmpeg_command(out_path, audio_path, width, height, fps, codec, bitrate_k)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    shot_i = 0
    engine.begin_shot("a", shots[0].generator, shots[0].palette, shots[0].seed, 0)
    fading = False
    try:
        for f in range(n_frames):
            t = f / fps
            # advance shots
            while shot_i + 1 < len(shots) and t >= shots[shot_i + 1].start_s:
                nxt = shots[shot_i + 1]
                if fading:
                    engine.swap_slots()       # B (already the next shot) becomes A
                else:
                    engine.begin_shot("a", nxt.generator, nxt.palette, nxt.seed, f)
                fading = False
                shot_i += 1
            mix = 0.0
            if shot_i + 1 < len(shots):
                nxt = shots[shot_i + 1]
                frames_to_next = int(nxt.start_s * fps) - f
                if not nxt.cut and 0 < frames_to_next <= fade_frames:
                    if not fading:
                        engine.begin_shot("b", nxt.generator, nxt.palette, nxt.seed, f)
                        fading = True
                    mix = 1.0 - frames_to_next / fade_frames
            u = {
                "time": float(t), "kick": float(sig["kick"][f]), "energy": float(sig["energy"][f]),
                "impact": float(sig["impact"][f]), "drop": float(sig["drop"][f]), "phrase": float(sig["phrase"][f]),
                "fillx": float(sig["fill"][f]), "beat": float(sig["beat_phase"][f]), "barphase": float(sig["bar_phase"][f]),
                "bars": float(sig["bars"][f]), "rms": float(rms[f]), "intensity": float(shots[shot_i].intensity),
                "imgmix": float(sig["section_progress"][f]),
            }
            engine.set_spectrum(spec[f])
            frame = engine.render_frame(u, f, fps, mix)
            proc.stdin.write(frame)
            if progress and f % (fps * 5) == 0:
                done = f / max(1, n_frames)
                elapsed = time.perf_counter() - t0
                eta = elapsed / max(done, 1e-3) * (1 - done) if f > fps * 5 else 0
                progress(0.05 + 0.9 * done, f"{t:.0f}/{duration:.0f} s · {shots[shot_i].generator} · {f / max(elapsed, 1e-6):.0f} fps" + (f" · ~{eta / 60:.0f} min left" if eta else ""))
        proc.stdin.close()
        rc = proc.wait()
    finally:
        engine.release()
    err = proc.stderr.read().decode(errors="replace").strip() if proc.stderr else ""
    if rc != 0 or not os.path.exists(out_path):
        raise RuntimeError(f"ffmpeg failed ({rc}): {err[-400:]}")
    seconds = time.perf_counter() - t0
    return {"frames": n_frames, "seconds": round(seconds, 1), "fps_achieved": round(n_frames / max(seconds, 1e-6), 1),
            "shots": [s.to_dict() for s in shots], "images": len(images), "renderer": engine.renderer, "codec": codec,
            "size": f"{width}x{height}", "fps": fps}


def preview_frame(timeline: dict, at_s: float, generator: str, palette_id: str, out_png: str, seed: int = 1,
                  width: int = 640, height: int = 360, images_dir: Path | None = None) -> None:
    """One PNG at `at_s` with a given generator - for testing and the settings preview."""
    from PIL import Image

    fps = 30
    sig = signals.build_signals(timeline, fps)
    f = min(len(sig["t"]) - 1, int(at_s * fps))
    pals = palmod.load_palettes()
    luts = {p.id: p.lut() for p in pals}
    images = load_images(images_dir or (paths.DATA / "images"), width, height) if generator == "stills" else []
    engine = Engine(width, height, luts, images)
    try:
        engine.begin_shot("a", generator, palette_id, seed, 0)
        engine.set_spectrum(np.clip(np.random.default_rng(seed).random(64), 0, 1).astype(np.float32))
        u = {"time": float(at_s), "kick": float(sig["kick"][f]), "energy": float(sig["energy"][f]), "impact": float(sig["impact"][f]),
             "drop": float(sig["drop"][f]), "phrase": 0.0, "fillx": 0.0, "beat": float(sig["beat_phase"][f]), "barphase": float(sig["bar_phase"][f]),
             "bars": float(sig["bars"][f]), "rms": 0.5, "intensity": 0.8, "imgmix": 0.3}
        for k in range(45):     # let feedback generators build up a little history
            frame = engine.render_frame(u, k, fps, 0.0)
        img = np.frombuffer(frame, dtype=np.uint8).reshape(height, width, 3)
        Image.fromarray(img).save(out_png)
    finally:
        engine.release()


__all__ = ["RESOLUTIONS", "ffmpeg_command", "load_images", "preview_frame", "render_video"]
