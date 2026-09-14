"""Φάση 0 — μέτρηση MusicGen (transformers): χρόνος για ~30 s, VRAM, WAV+MP3 για ακρόαση."""
import json
import os
import subprocess
import sys
import time

import numpy as np
import soundfile as sf
import torch

FFMPEG = os.environ.get("TMG_FFMPEG", "ffmpeg")
model_id = sys.argv[1] if len(sys.argv) > 1 else "facebook/musicgen-small"
outdir = sys.argv[2] if len(sys.argv) > 2 else "musicgen"
seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
dtype_name = sys.argv[4] if len(sys.argv) > 4 else "fp32"
dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
prompt = ("uplifting trance, 138 bpm, driving four-on-the-floor kick, rolling bassline, "
          "supersaw lead melody, euphoric breakdown with sidechained pads, energetic club anthem")

from transformers import AutoProcessor, MusicgenForConditionalGeneration

t0 = time.perf_counter()
processor = AutoProcessor.from_pretrained(model_id)
model = MusicgenForConditionalGeneration.from_pretrained(model_id, dtype=dtype, use_safetensors=True).to("cuda").eval()
t_load = time.perf_counter() - t0
sr = model.config.audio_encoder.sampling_rate
frame_rate = model.config.audio_encoder.frame_rate
n_tokens = int(seconds * frame_rate)
nparams = sum(p.numel() for p in model.parameters()) / 1e9
print(f"model={model_id} params={nparams:.2f}B dtype={dtype_name} sr={sr} frame_rate={frame_rate} load={t_load:.1f}s")

inputs = processor(text=[prompt], padding=True, return_tensors="pt").to("cuda")
torch.manual_seed(1234)
torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
t0 = time.perf_counter()
with torch.no_grad():
    audio = model.generate(**inputs, do_sample=True, guidance_scale=3.0, max_new_tokens=n_tokens)
torch.cuda.synchronize()
t_gen = time.perf_counter() - t0
vram = torch.cuda.max_memory_allocated() / 2**30
a = audio[0].float().cpu().numpy()  # (C, T)
raw_peak = float(np.abs(a).max())
if raw_peak > 0.95: a = a * (0.95 / raw_peak)  # MusicGen βγάζει >1.0 — κανονικοποίηση αλλιώς clipping στο PCM_16
dur = a.shape[-1] / sr
print(f"generated {dur:.1f}s audio ({a.shape[0]}ch) in {t_gen:.1f}s  ({dur/t_gen:.2f}x realtime)  peakVRAM={vram:.2f} GB  raw_peak={raw_peak:.3f} (normalised to 0.95)")

os.makedirs(outdir, exist_ok=True)
base = model_id.split("/")[-1] + f"__{dtype_name}"
wpath = os.path.join(outdir, base + ".wav"); sf.write(wpath, a.T, sr, subtype="PCM_16")
subprocess.run([FFMPEG, "-v", "error", "-y", "-i", wpath, "-codec:a", "libmp3lame", "-q:a", "2", os.path.join(outdir, base + ".mp3")], check=True)
json.dump({"model": model_id, "dtype": dtype_name, "params_B": nparams, "load_s": t_load, "gen_s": t_gen, "audio_s": dur,
           "realtime_factor": dur/t_gen, "peak_vram_gb": vram, "prompt": prompt}, open(os.path.join(outdir, base + ".json"), "w"), indent=1)
print("out:", os.path.join(outdir, base + ".mp3"))
