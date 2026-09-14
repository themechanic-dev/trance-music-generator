"""Φάση 0 — μέτρηση Demucs: χρόνος, VRAM, stems σε WAV+MP3 για ακρόαση."""
import json
import os
import subprocess
import sys
import time

import numpy as np
import soundfile as sf
import torch

FFMPEG = os.environ.get("TMG_FFMPEG", "ffmpeg")
track = sys.argv[1]
model_name = sys.argv[2] if len(sys.argv) > 2 else "htdemucs"
outdir = sys.argv[3] if len(sys.argv) > 3 else "stems"

def load_audio(path, sr=44100):
    cmd = [FFMPEG, "-v", "error", "-i", path, "-f", "f32le", "-ac", "2", "-ar", str(sr), "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T.copy()  # (2, T)

from demucs.apply import apply_model
from demucs.pretrained import get_model

t0 = time.perf_counter()
model = get_model(model_name)
model.to("cuda").eval()
t_load = time.perf_counter() - t0
print(f"model={model_name} sources={model.sources} sr={model.samplerate} load={t_load:.1f}s")

wav = load_audio(track, model.samplerate)
dur = wav.shape[1] / model.samplerate
ref = wav.mean(0); mean, std = ref.mean(), ref.std()
x = torch.from_numpy((wav - mean) / std)[None].to("cuda")

torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
t0 = time.perf_counter()
with torch.no_grad():
    out = apply_model(model, x, device="cuda", shifts=1, split=True, overlap=0.25, progress=False)
torch.cuda.synchronize()
t_sep = time.perf_counter() - t0
vram = torch.cuda.max_memory_allocated() / 2**30
out = out[0] * std + mean  # (S, 2, T)
print(f"track={os.path.basename(track)} dur={dur:.1f}s separate={t_sep:.1f}s  ({dur/t_sep:.1f}x realtime)  peakVRAM={vram:.2f} GB")

base = os.path.splitext(os.path.basename(track))[0]
d = os.path.join(outdir, f"{base}__{model_name}"); os.makedirs(d, exist_ok=True)
for name, stem in zip(model.sources, out.cpu().numpy()):
    wpath = os.path.join(d, f"{name}.wav"); sf.write(wpath, stem.T, model.samplerate, subtype="PCM_16")
    subprocess.run([FFMPEG, "-v", "error", "-y", "-i", wpath, "-codec:a", "libmp3lame", "-q:a", "2", os.path.join(d, f"{name}.mp3")], check=True)
    rms = float(np.sqrt((stem**2).mean()))
    print(f"  {name:8s} rms={rms:.4f} -> {name}.mp3")
json.dump({"model": model_name, "track": track, "duration_s": dur, "load_s": t_load, "separate_s": t_sep,
           "realtime_factor": dur/t_sep, "peak_vram_gb": vram}, open(os.path.join(d, "metrics.json"), "w"), indent=1)
print("out:", d)
