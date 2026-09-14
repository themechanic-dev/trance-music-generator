"""Φάση 0 — μέτρηση Stable Audio Open (diffusers): χρόνος, VRAM, WAV+MP3."""
import json
import os
import subprocess
import sys
import time

import soundfile as sf
import torch

FFMPEG = os.environ.get("TMG_FFMPEG", "ffmpeg")
model_id = sys.argv[1] if len(sys.argv) > 1 else "stabilityai/stable-audio-open-1.0"
outdir = sys.argv[2] if len(sys.argv) > 2 else "stableaudio"
seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
prompt = "uplifting trance, 138 bpm, four-on-the-floor kick, rolling bassline, supersaw lead, euphoric, club"
from diffusers import StableAudioPipeline

t0 = time.perf_counter()
pipe = StableAudioPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
t_load = time.perf_counter() - t0
sr = pipe.vae.sampling_rate
print(f"model={model_id} sr={sr} load={t_load:.1f}s")
g = torch.Generator("cuda").manual_seed(1234)
torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.perf_counter()
audio = pipe(prompt, negative_prompt="low quality, noise", num_inference_steps=100, audio_end_in_s=seconds, num_waveforms_per_prompt=1, generator=g).audios
torch.cuda.synchronize(); t_gen = time.perf_counter() - t0
vram = torch.cuda.max_memory_allocated() / 2**30
a = audio[0].float().cpu().numpy(); dur = a.shape[-1]/sr
raw_peak = float(abs(a).max()); a = a / (raw_peak or 1.0) * 0.95  # the VAE output exceeds +/-1.0 - normalise like MusicGen
print(f"generated {dur:.1f}s ({a.shape[0]}ch) in {t_gen:.1f}s ({dur/t_gen:.2f}x realtime) peakVRAM={vram:.2f} GB rawpeak={raw_peak:.2f}")
os.makedirs(outdir, exist_ok=True); base = model_id.split("/")[-1]
wpath = os.path.join(outdir, base + ".wav"); sf.write(wpath, a.T, sr, subtype="PCM_16")
subprocess.run([FFMPEG, "-v", "error", "-y", "-i", wpath, "-codec:a", "libmp3lame", "-q:a", "2", os.path.join(outdir, base + ".mp3")], check=True)
json.dump({"model": model_id, "load_s": t_load, "gen_s": t_gen, "audio_s": dur, "peak_vram_gb": vram, "raw_peak": raw_peak}, open(os.path.join(outdir, base + ".json"), "w"), indent=1)
print("out:", os.path.join(outdir, base + ".mp3"))
