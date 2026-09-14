# Φάση 0 — scripts μέτρησης

Τρέχουν με το `.venv` (Python 3.12) αφού γίνει `source env.sh` από τη ρίζα του project.
Τα αποτελέσματα είναι καταγεγραμμένα στο `../../ΜΕΤΡΗΣΕΙΣ.md`.

| Script | Τι μετρά |
|---|---|
| `test_demucs.py <mp3> [htdemucs\|htdemucs_ft] [outdir]` | χρόνος/VRAM διαχωρισμού, stems σε WAV+MP3 |
| `test_musicgen.py [model_id] [outdir] [seconds] [fp32\|fp16]` | χρόνος/VRAM παραγωγής MusicGen, WAV+MP3 |
| `test_stableaudio.py [model_id] [outdir] [seconds]` | ίδιο για Stable Audio Open (θέλει HF token) |
| `test_moderngl.py [out.png]` | plasma shader 1080p headless EGL, fps |
| `bench_plasma_cpu.py` | το ίδιο plasma σε numpy/CPU (βάση σύγκρισης) — τρέχει και με το python3 του συστήματος |
| `test_librosa.py <mp3>...` | BPM / beat grid / τονικότητα / ενέργεια σε CPU, χρόνος ανά κομμάτι |
