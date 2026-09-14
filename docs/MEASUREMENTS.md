# Measurements

Everything below was measured on the development machine: Ubuntu 26.04, Ryzen 7 2700X, 30 GB RAM,
NVIDIA RTX 3060 12 GB (driver 595, CUDA 13). Numbers are what the tools did here, not estimates.

## Environment

| | |
|---|---|
| GUI interpreter | system Python 3.14 (the only one with GTK 4 / libadwaita bindings) |
| Worker interpreter | Python 3.12 managed by `uv` in `.venv` (moderngl has no wheels for 3.14 yet) |
| torch | 2.14 + CUDA 13, 6.1 TFLOPS fp32 / 10.2 TFLOPS fp16 on a 4096² matmul |
| ffmpeg | static build with NVENC, verified with a real 1080p encode (not just `-encoders`) |

## Library analysis (phase 2)

| Step | Per 7-minute track | Peak VRAM |
|---|---|---|
| Demucs `htdemucs` (default) | 14-16 s (about 30x realtime) | 1.6-1.8 GB |
| Demucs `htdemucs_ft` (optional, cleaner vocals) | 55 s | 3.2 GB |
| librosa beats / bars / key / patterns | ~3 s after a one-off 23 s numba JIT | - |

A 2,000-track collection is therefore roughly 15 hours of background analysis, resumable at any point.

## Composing (phases 3-4)

| | |
|---|---|
| numpy composer, 6-minute track | 25-75 s (7-12x realtime) |
| phrase placement (trim, stretch, echo, reverb, ducking) | seconds |

## Neural textures (phase 6)

| Model | 30 s clip | Peak VRAM |
|---|---|---|
| MusicGen stereo-small (fp32) | 35 s | 3.1 GB |
| MusicGen medium (fp16) | 68 s | 4.7 GB (9.3 GB in fp32 - not used) |
| SD-Turbo stills 768x432 | ~2 s each after a 19 s load | 3.2 GB |

## Video (phase 5)

| | |
|---|---|
| GLSL generators at 640x360 | 460-1,100 fps each |
| Full render, 5:06 track, 1080p 30 fps, h264_nvenc 10 Mbit | 70 s (131 fps) |
| Headless context | EGL via moderngl, no window needed |

## System-audio capture (phase 1)

`pw-record` on the monitor of the default PipeWire sink: a 440 Hz test tone at amplitude 0.5 came back at
peak 0.500, i.e. bit-exact capture. The default sink is read dynamically from `pw-metadata`.
