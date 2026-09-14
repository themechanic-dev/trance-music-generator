"""Per-track analysis: what the composer needs to learn - beat, tempo, rhythm patterns, structure.

Runs in the worker (.venv): numpy + librosa on the CPU, Demucs on the GPU when enabled.
Everything here is measured in *bars* once the beat grid is known, because that is the unit the
composer works in. The output is a plain dict (JSON-serialisable) stored per track.
"""

from __future__ import annotations

import math
import os
import subprocess
import time
from collections import Counter

import numpy as np

ANALYSIS_VERSION = 1
SR = 44100          # decoding / Demucs rate
ASR = 22050         # analysis rate
HOP = 512
SLOTS = 16          # 16th-note slots per bar
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Kessler key profiles.
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


# ---- audio i/o -------------------------------------------------------------------------

def load_audio(path: str, sr: int = SR, ffmpeg: str | None = None) -> np.ndarray:
    """Decode anything ffmpeg can read -> float32 (2, T) at `sr`."""
    ff = ffmpeg or os.environ.get("TMG_FFMPEG", "ffmpeg")
    cmd = [ff, "-v", "error", "-i", path, "-f", "f32le", "-ac", "2", "-ar", str(sr), "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    if not raw:
        raise RuntimeError("ffmpeg produced no audio")
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T.copy()


def write_mp3(wav: np.ndarray, sr: int, path: str, ffmpeg: str | None = None, quality: str = "4") -> None:
    ff = ffmpeg or os.environ.get("TMG_FFMPEG", "ffmpeg")
    cmd = [ff, "-v", "error", "-y", "-f", "f32le", "-ac", str(wav.shape[0]), "-ar", str(sr), "-i", "-",
           "-codec:a", "libmp3lame", "-q:a", quality, path]
    subprocess.run(cmd, input=np.ascontiguousarray(wav.T, dtype=np.float32).tobytes(), check=True, capture_output=True)


def to_mono_22k(wav: np.ndarray, sr: int = SR) -> np.ndarray:
    from scipy.signal import resample_poly

    mono = wav.mean(axis=0) if wav.ndim == 2 else wav
    if sr == ASR:
        return mono.astype(np.float32)
    g = math.gcd(sr, ASR)
    return resample_poly(mono, ASR // g, sr // g).astype(np.float32)


# ---- Demucs ------------------------------------------------------------------------------

def load_demucs(name: str = "htdemucs"):
    import torch
    from demucs.pretrained import get_model

    model = get_model(name)
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return model


def separate(model, wav: np.ndarray) -> dict[str, np.ndarray]:
    """(2, T) float32 at 44.1 kHz -> {'drums','bass','other','vocals'} each (2, T)."""
    import torch
    from demucs.apply import apply_model

    device = next(model.parameters()).device
    ref = wav.mean(0)
    mean, std = float(ref.mean()), float(ref.std()) or 1.0
    x = torch.from_numpy((wav - mean) / std)[None].to(device)
    with torch.no_grad():
        out = apply_model(model, x, device=device, shifts=1, split=True, overlap=0.25, progress=False)[0]
    out = (out * std + mean).cpu().numpy()
    return {name: out[i] for i, name in enumerate(model.sources)}


# ---- building blocks (pure numpy / librosa; unit-tested) ---------------------------------

def band_onsets(y: np.ndarray, sr: int = ASR, hop: int = HOP) -> dict[str, np.ndarray]:
    """Onset strength per band: kick (<150 Hz), snare/clap (150-2000 Hz), hats (>5 kHz). Frames at sr/hop."""
    import librosa

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=hop, n_mels=96, fmax=sr / 2)
    freqs = librosa.mel_frequencies(n_mels=96, fmax=sr / 2)
    db = librosa.power_to_db(S, ref=np.max)
    bands = {"kick": freqs < 150, "snare": (freqs >= 150) & (freqs < 2000), "hat": freqs >= 5000}
    out = {}
    for name, mask in bands.items():
        env = librosa.onset.onset_strength(S=db[mask], sr=sr, hop_length=hop)
        out[name] = np.asarray(env, dtype=np.float32)
    return out


def estimate_beats(y: np.ndarray, sr: int = ASR, hop: int = HOP, prior_bpm: float = 138.0) -> tuple[float, np.ndarray, np.ndarray]:
    """BPM (folded into 100-190), beat times (s) and the full onset envelope."""
    import librosa

    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    tempo = float(np.atleast_1d(librosa.feature.tempo(onset_envelope=oenv, sr=sr, hop_length=hop, start_bpm=prior_bpm, std_bpm=1.2))[0])
    tempo = fold_bpm(tempo)
    _, beats = librosa.beat.beat_track(onset_envelope=oenv, sr=sr, hop_length=hop, bpm=tempo, units="time", trim=False)
    beats = np.asarray(beats, dtype=np.float64)
    if len(beats) > 8:
        # Refine from the overall slope, not the median interval: beat times are quantised to
        # frames (23 ms at 22 kHz / 512), which biases a single interval by up to 2 BPM.
        ibi = np.diff(beats)
        good = ibi[(ibi > 0.5 * np.median(ibi)) & (ibi < 1.5 * np.median(ibi))]
        if len(good) >= 8:
            tempo = fold_bpm(60.0 / float(good.mean()))
    return tempo, beats, oenv


def fold_bpm(bpm: float, low: float = 100.0, high: float = 190.0) -> float:
    while bpm < low:
        bpm *= 2
    while bpm > high:
        bpm /= 2
    return bpm


def downbeat_phase(kick_env: np.ndarray, beats: np.ndarray, sr: int = ASR, hop: int = HOP) -> int:
    """Which beat (0..3) carries the downbeat: the phase where kick energy on every 4th beat is strongest."""
    frames = np.clip((beats * sr / hop).round().astype(int), 0, len(kick_env) - 1)
    best, best_score = 0, -1.0
    for p in range(4):
        idx = frames[p::4]
        if len(idx) == 0:
            continue
        score = float(np.mean(np.maximum.reduce([kick_env[np.clip(idx + d, 0, len(kick_env) - 1)] for d in (-1, 0, 1)])))
        if score > best_score:
            best, best_score = p, score
    return best


def bar_grid(beats: np.ndarray, phase: int) -> np.ndarray:
    """Bar start times from the beat list and the downbeat phase; the last partial bar is dropped."""
    starts = beats[phase::4]
    if len(starts) >= 2:
        return starts[:-1] if len(beats) - (phase + 4 * (len(starts) - 1)) < 4 else starts
    return starts


def slot_pattern(env: np.ndarray, bar_start: float, bar_dur: float, sr: int = ASR, hop: int = HOP,
                 rel_threshold: float = 0.45, abs_floor: float = 0.0) -> str:
    """16-slot hit pattern of one bar from an onset envelope, e.g. '1000100010001000'."""
    fps = sr / hop
    vals = np.zeros(SLOTS, dtype=np.float32)
    for k in range(SLOTS):
        f = int(round((bar_start + k * bar_dur / SLOTS) * fps))
        lo, hi = max(0, f - 1), min(len(env), f + 2)
        if lo < hi:
            vals[k] = env[lo:hi].max()
    top = float(vals.max())
    if top <= abs_floor or top <= 0:
        return "0" * SLOTS
    return "".join("1" if v >= rel_threshold * top and v > abs_floor else "0" for v in vals)


def rms_db(y: np.ndarray) -> float:
    if len(y) == 0:
        return -100.0
    r = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    return 20 * math.log10(r) if r > 1e-7 else -100.0


def root_midi(seg: np.ndarray, sr: int = ASR, fmin: float = 30.0, fmax: float = 260.0) -> int | None:
    """Fundamental of a bass segment by autocorrelation -> pitch class 0..11 (None when silent)."""
    seg = seg.astype(np.float64)
    seg = seg - seg.mean()
    if len(seg) < int(sr / fmin) * 2 or np.sqrt(np.mean(seg**2)) < 1e-3:
        return None
    n = len(seg)
    spec = np.fft.rfft(seg, n=2 * n)
    ac = np.fft.irfft(spec * np.conj(spec))[:n]
    lag_lo, lag_hi = int(sr / fmax), int(sr / fmin)
    if lag_hi >= n:
        return None
    window = ac[lag_lo:lag_hi]
    if window.max() <= 0.3 * ac[0]:
        return None
    lag = lag_lo + int(np.argmax(window))
    freq = sr / lag
    midi = 69 + 12 * math.log2(freq / 440.0)
    return int(round(midi)) % 12


def estimate_key(chroma_mean: np.ndarray) -> tuple[str, str, float]:
    """Krumhansl correlation -> (tonic name, 'major'|'minor', confidence 0-1)."""
    c = np.asarray(chroma_mean, dtype=np.float64)
    if c.sum() <= 0:
        return "?", "?", 0.0
    scores = []
    for mode, prof in (("major", _MAJOR), ("minor", _MINOR)):
        for tonic in range(12):
            r = np.corrcoef(np.roll(prof, tonic), c)[0, 1]
            scores.append((r, tonic, mode))
    scores.sort(reverse=True)
    best, second = scores[0], scores[1]
    conf = float(max(0.0, min(1.0, (best[0] - second[0]) * 4 + 0.3)))
    return NOTE_NAMES[best[1]], best[2], conf


def smooth_bool(flags: np.ndarray, width: int = 5) -> np.ndarray:
    """Majority filter so that one odd bar does not split a section."""
    if len(flags) < width:
        return flags.copy()
    pad = width // 2
    padded = np.pad(flags.astype(int), pad, mode="edge")
    out = np.array([padded[i:i + width].sum() > width // 2 for i in range(len(flags))])
    return out


def segment_sections(kick_active: np.ndarray, energy_db: np.ndarray, min_run: int = 4) -> list[dict]:
    """Bars -> sections: intro / build / drop / breakdown / outro.

    No-kick runs are intro (first), outro (last) or breakdown (in between). A kick run starts as a
    build while its energy is still climbing towards the run's peak and becomes a drop once it is there.
    """
    flags = smooth_bool(np.asarray(kick_active, dtype=bool))
    energy = np.asarray(energy_db, dtype=float)
    n = len(flags)
    if n == 0:
        return []
    # runs of equal value
    runs: list[list] = []
    for i, v in enumerate(flags):
        if runs and runs[-1][0] == v:
            runs[-1][2] += 1
        else:
            runs.append([bool(v), i, 1])
    # absorb runs shorter than min_run into their neighbours
    merged: list[list] = []
    for r in runs:
        if merged and r[2] < min_run and len(runs) > 1:
            merged[-1][2] += r[2]
        else:
            merged.append(r)
    runs = merged
    if len(runs) >= 2 and runs[0][2] < min_run:
        runs[1][1] = runs[0][1]
        runs[1][2] += runs[0][2]
        runs = runs[1:]
    # absorbing a short run can leave two neighbours with the same value: join them (no 'drop > drop')
    joined: list[list] = []
    for r in runs:
        if joined and joined[-1][0] == r[0]:
            joined[-1][2] += r[2]
        else:
            joined.append(r)
    runs = joined
    sections: list[dict] = []
    for idx, (val, start, length) in enumerate(runs):
        e = energy[start:start + length]
        if not val:
            kind = "intro" if idx == 0 else "outro" if idx == len(runs) - 1 else "breakdown"
            sections.append({"type": kind, "start_bar": int(start), "bars": int(length), "energy_db": round(float(e.mean()), 2)})
            continue
        peak = float(e.max()) if len(e) else 0.0
        nb = 0
        while nb < length and e[nb] < peak - 2.0:
            nb += 1
        nb = (nb // 4) * 4
        if length < 8 or nb < 4 or length - nb < 4:
            nb = 0
        if nb:
            sections.append({"type": "build", "start_bar": int(start), "bars": int(nb), "energy_db": round(float(e[:nb].mean()), 2)})
        sections.append({"type": "drop", "start_bar": int(start + nb), "bars": int(length - nb), "energy_db": round(float(e[nb:].mean()), 2)})
    return sections


def vocal_segments(vocals_22k: np.ndarray, sr: int = ASR, hop: int = HOP, threshold_db: float = -35.0,
                   min_len: float = 0.8, max_len: float = 8.0, gap: float = 0.35, max_count: int = 3) -> list[dict]:
    """Where someone is speaking or singing in the vocals stem: [{start, end, level_db}], loudest first."""
    import librosa

    rms = librosa.feature.rms(y=vocals_22k, frame_length=2048, hop_length=hop)[0]
    db = 20 * np.log10(np.maximum(rms, 1e-7))
    active = db > threshold_db
    fps = sr / hop
    segs: list[list[float]] = []
    i = 0
    while i < len(active):
        if active[i]:
            j = i
            while j < len(active) and active[j]:
                j += 1
            start, end = i / fps, j / fps
            if segs and start - segs[-1][1] < gap:
                segs[-1][1] = end
            else:
                segs.append([start, end])
            i = j
        else:
            i += 1
    out = []
    for s, e in segs:
        if e - s < min_len:
            continue
        if e - s > max_len:
            e = s + max_len
        lo, hi = int(s * fps), max(int(s * fps) + 1, int(e * fps))
        out.append({"start": round(s, 3), "end": round(e, 3), "level_db": round(float(db[lo:hi].mean()), 1)})
    out.sort(key=lambda d: -d["level_db"])
    return out[:max_count]


# ---- the whole track ------------------------------------------------------------------------

def analyze(path: str, *, demucs_model=None, want_stems: bool = False, vocal_opts: dict | None = None,
            ffmpeg: str | None = None) -> tuple[dict, dict[str, np.ndarray] | None]:
    """Analyse one track. Returns (analysis dict, stems at 44.1 kHz or None)."""
    import librosa

    t0 = time.perf_counter()
    wav = load_audio(path, SR, ffmpeg)
    duration = wav.shape[1] / SR
    timing = {"decode": round(time.perf_counter() - t0, 2)}

    stems = None
    if demucs_model is not None:
        t1 = time.perf_counter()
        stems = separate(demucs_model, wav)
        timing["demucs"] = round(time.perf_counter() - t1, 2)

    t2 = time.perf_counter()
    mix22 = to_mono_22k(wav)
    if stems:
        drums22, bass22, other22, vocals22 = (to_mono_22k(stems[k]) for k in ("drums", "bass", "other", "vocals"))
    else:
        drums22 = bass22 = other22 = mix22
        vocals22 = None

    bpm, beats, _ = estimate_beats(drums22)
    bands = band_onsets(drums22)
    bass_env = np.asarray(librosa.onset.onset_strength(y=bass22, sr=ASR, hop_length=HOP), dtype=np.float32)
    phase = downbeat_phase(bands["kick"], beats) if len(beats) else 0
    bars = bar_grid(beats, phase) if len(beats) >= 8 else np.array([])
    bar_dur = 240.0 / bpm

    fps = ASR / HOP
    floors = {k: float(np.percentile(v, 95)) * 0.08 for k, v in bands.items()}
    bass_floor = float(np.percentile(bass_env, 95)) * 0.08
    kick_pat, snare_pat, hat_pat, bass_pat = Counter(), Counter(), Counter(), Counter()
    kick_energy, energy, vocal_energy, roots, patterns_per_bar = [], [], [], [], []
    for b0 in bars:
        s0, s1 = int(b0 * ASR), int((b0 + bar_dur) * ASR)
        f0, f1 = int(b0 * fps), max(int(b0 * fps) + 1, int((b0 + bar_dur) * fps))
        kp = slot_pattern(bands["kick"], b0, bar_dur, abs_floor=floors["kick"])
        kick_pat[kp] += 1
        snare_pat[slot_pattern(bands["snare"], b0, bar_dur, abs_floor=floors["snare"])] += 1
        hat_pat[slot_pattern(bands["hat"], b0, bar_dur, rel_threshold=0.5, abs_floor=floors["hat"])] += 1
        bass_pat[slot_pattern(bass_env, b0, bar_dur, abs_floor=bass_floor)] += 1
        patterns_per_bar.append(kp)
        kick_energy.append(float(bands["kick"][f0:f1].mean()) if f1 > f0 else 0.0)
        energy.append(rms_db(mix22[s0:s1]))
        vocal_energy.append(rms_db(vocals22[s0:s1]) if vocals22 is not None else -100.0)
        # bass root: the first beat of the bar, majority over 4 beats
        votes = [root_midi(bass22[int((b0 + q * bar_dur / 4) * ASR):int((b0 + q * bar_dur / 4 + 0.22) * ASR)]) for q in range(4)]
        votes = [v for v in votes if v is not None]
        roots.append(Counter(votes).most_common(1)[0][0] if votes else None)

    kick_energy_arr = np.asarray(kick_energy, dtype=float)
    if len(kick_energy_arr):
        thr = 0.3 * float(np.percentile(kick_energy_arr, 90))
        kick_active = [(ke > thr) and (p.count("1") >= 2) for ke, p in zip(kick_energy_arr, patterns_per_bar)]
    else:
        kick_active = []
    sections = segment_sections(np.asarray(kick_active, dtype=bool), np.asarray(energy, dtype=float)) if bars.size else []

    harm = other22 if stems else mix22
    chroma = librosa.feature.chroma_cqt(y=harm, sr=ASR, hop_length=2048)
    key_name, mode, key_conf = estimate_key(chroma.mean(axis=1))
    tonic = NOTE_NAMES.index(key_name) if key_name in NOTE_NAMES else 0
    rel_roots = [((r - tonic) % 12) if r is not None else None for r in roots]

    vocals = []
    if vocals22 is not None and vocal_opts is not None and vocal_opts.get("enabled", True):
        vocals = vocal_segments(vocals22, threshold_db=float(vocal_opts.get("threshold_db", -35.0)),
                                min_len=float(vocal_opts.get("min_len", 0.8)), max_len=float(vocal_opts.get("max_len", 8.0)),
                                max_count=int(vocal_opts.get("max_count", 3)))
    timing["analysis"] = round(time.perf_counter() - t2, 2)
    timing["total"] = round(time.perf_counter() - t0, 2)

    analysis = {
        "version": ANALYSIS_VERSION,
        "duration_s": round(duration, 3),
        "bpm": round(bpm, 2),
        "beats": int(len(beats)),
        "bars": int(len(bars)),
        "downbeat_phase": int(phase),
        "first_bar_s": round(float(bars[0]), 3) if len(bars) else None,
        "key": key_name, "mode": mode, "key_conf": round(key_conf, 2),
        "sections": sections,
        "kick_patterns": dict(kick_pat.most_common(12)),
        "snare_patterns": dict(snare_pat.most_common(12)),
        "hat_patterns": dict(hat_pat.most_common(12)),
        "bass_patterns": dict(bass_pat.most_common(12)),
        "bass_roots_rel": rel_roots,
        "energy_db": [round(e, 2) for e in energy],
        "vocal_db": [round(v, 1) for v in vocal_energy],
        "kick_active": [bool(k) for k in kick_active],
        "vocal_segments": vocals,
        "demucs": stems is not None,
        "timing_s": timing,
    }
    return analysis, (stems if want_stems or vocals else None)
