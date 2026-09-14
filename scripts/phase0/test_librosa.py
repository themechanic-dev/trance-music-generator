"""Φάση 0 (bonus) — librosa σε CPU: BPM, beat grid, τονικότητα, ενέργεια — ο χρόνος ανά κομμάτι για τη Φάση 2."""
import os
import subprocess
import sys
import time

import librosa
import numpy as np

FFMPEG = os.environ.get("TMG_FFMPEG", "ffmpeg")
for track in sys.argv[1:]:
    sr = 22050
    raw = subprocess.run([FFMPEG, "-v", "error", "-i", track, "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"], capture_output=True, check=True).stdout
    y = np.frombuffer(raw, dtype=np.float32); dur = len(y)/sr
    t0 = time.perf_counter()
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    tempo, beats = librosa.beat.beat_track(onset_envelope=oenv, sr=sr, hop_length=512, units="time")
    t_beat = time.perf_counter() - t0
    t0 = time.perf_counter()
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=2048)
    key_idx = int(chroma.mean(1).argmax()); keys = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    rms = librosa.feature.rms(y=y, hop_length=2048)[0]
    t_feat = time.perf_counter() - t0
    tempo = float(np.atleast_1d(tempo)[0])
    ibi = np.diff(beats); print(f"{os.path.basename(track)}: dur={dur:.0f}s  BPM={tempo:.1f}  beats={len(beats)}  median-beat={np.median(ibi):.3f}s (={60/np.median(ibi):.1f} bpm)  key~{keys[key_idx]}  rms range {rms.min():.3f}-{rms.max():.3f}  | beat_track {t_beat:.1f}s, chroma+rms {t_feat:.1f}s")
