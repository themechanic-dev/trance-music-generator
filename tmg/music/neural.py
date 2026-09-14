"""Neural sound inside the composer: MusicGen textures fitted to the plan and laid under the arrangement.

The numpy composer stays the orchestrator (it knows every bar); MusicGen supplies what it cannot: real
timbre. Two clips per track - an atmosphere for intros / breakdowns / outros and an energy layer for
builds / drops - are generated from a prompt built out of the plan (tempo, key, flavor), stretched to the
plan's tempo when their own tempo is close enough, looped bar-aligned to each section, high-passed so the
kick and bass stay ours, ducked by the kick, faded at every section boundary, and mixed in at a chosen level.
With 'melody' the model also follows the harmony of our own rendered track (chroma conditioning).
Everything is cached in data/neural so an A/B re-run costs nothing.
"""

from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path

import numpy as np

from tmg import paths
from tmg.music import synth
from tmg.music.synth import SAMPLE_RATE

MODELS = {
    "stereo-small": {"id": "facebook/musicgen-stereo-small", "dtype": "fp32", "melody": False, "vram_gb": 3.1, "label": "MusicGen stereo-small (fast, 35 s per clip)"},
    "medium": {"id": "facebook/musicgen-medium", "dtype": "fp16", "melody": False, "vram_gb": 4.7, "label": "MusicGen medium fp16 (better, 70 s per clip, mono)"},
    "melody": {"id": "facebook/musicgen-stereo-melody", "dtype": "fp16", "melody": True, "vram_gb": 5.5, "label": "MusicGen stereo-melody fp16 (follows our harmony, ~6 GB download)"},
}
CLIP_SECONDS = 30.0
MUSICGEN_SR = 32000
ROLE_SECTIONS = {"atmosphere": ("intro", "breakdown", "outro"), "energy": ("build", "drop")}
ROLE_LEVEL_DB = {"intro": -2.0, "breakdown": 0.0, "outro": -2.0, "build": -6.0, "drop": -8.0}   # relative to the layer level

_loaded: dict[str, tuple] = {}


# ---- prompts ---------------------------------------------------------------------------------

def prompt_for(plan, role: str) -> str:
    key = plan.key_name
    bpm = int(round(plan.bpm))
    flavor = plan.flavor.name
    if role == "atmosphere":
        mood = {"psy": "psychedelic, hypnotic", "progressive": "deep, warm, spacious", "tech": "dark, minimal, cinematic", "uplifting": "euphoric, emotional"}.get(flavor, "atmospheric")
        return (f"atmospheric trance breakdown texture, {mood} pad and shimmering synth layers, {key}, {bpm} bpm, "
                "wide stereo reverb, no drums, no vocals, ambient, evolving")
    mood = {"psy": "psytrance rolling energy, acid lead", "progressive": "progressive trance groove, plucky synth", "tech": "tech trance drive, filtered synth stabs", "uplifting": "uplifting trance supersaw lead, euphoric melody"}.get(flavor, "trance lead")
    return f"{mood}, {key}, {bpm} bpm, energetic synth layer, driving, no vocals, club mix"


# ---- generation (cached) -----------------------------------------------------------------------

def _cache_key(model_key: str, prompt: str, seconds: float, seed: int, melody_tag: str) -> str:
    h = hashlib.sha1(f"{model_key}|{prompt}|{seconds}|{seed}|{melody_tag}".encode()).hexdigest()[:16]
    return f"{model_key}-{h}"


def _load(model_key: str):
    if model_key in _loaded:
        return _loaded[model_key]
    import torch
    from transformers import AutoProcessor

    spec = MODELS[model_key]
    dtype = torch.float16 if spec["dtype"] == "fp16" else torch.float32
    processor = AutoProcessor.from_pretrained(spec["id"])
    if spec["melody"]:
        from transformers import MusicgenMelodyForConditionalGeneration as Cls
    else:
        from transformers import MusicgenForConditionalGeneration as Cls
    model = Cls.from_pretrained(spec["id"], dtype=dtype, use_safetensors=True).to("cuda").eval()
    _loaded[model_key] = (processor, model)
    return processor, model


def unload() -> None:
    _loaded.clear()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def generate(model_key: str, prompt: str, seconds: float, seed: int, *, melody: np.ndarray | None = None,
             melody_sr: int = SAMPLE_RATE, cache_dir: Path | None = None, log=None) -> tuple[np.ndarray, dict]:
    """(2, T) float32 at 44.1 kHz, peak-normalised, and a dict with timings. Cached on disk."""
    import soundfile as sf

    cache_dir = cache_dir or (paths.DATA / "neural")
    cache_dir.mkdir(parents=True, exist_ok=True)
    melody_tag = hashlib.sha1(melody.tobytes()).hexdigest()[:10] if melody is not None else "text"
    key = _cache_key(model_key, prompt, seconds, seed, melody_tag)
    path = cache_dir / f"{key}.wav"
    if path.exists():
        y, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return y.T.copy(), {"cached": True, "path": str(path), "model": model_key}

    import torch
    from scipy.signal import resample_poly

    t0 = time.perf_counter()
    processor, model = _load(model_key)
    load_s = time.perf_counter() - t0
    spec = MODELS[model_key]
    frame_rate = model.config.audio_encoder.frame_rate
    sr_out = model.config.audio_encoder.sampling_rate
    if spec["melody"] and melody is not None:
        mono = melody.mean(axis=0) if melody.ndim == 2 else melody
        if melody_sr != sr_out:
            g = math.gcd(int(melody_sr), sr_out)
            mono = resample_poly(mono, sr_out // g, int(melody_sr) // g).astype(np.float32)
        inputs = processor(audio=mono, sampling_rate=sr_out, text=[prompt], padding=True, return_tensors="pt").to("cuda")
    else:
        inputs = processor(text=[prompt], padding=True, return_tensors="pt").to("cuda")
    if spec["dtype"] == "fp16" and "input_features" in inputs:
        inputs["input_features"] = inputs["input_features"].half()
    torch.manual_seed(seed)
    t1 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, do_sample=True, guidance_scale=3.0, max_new_tokens=int(seconds * frame_rate))
    gen_s = time.perf_counter() - t1
    audio = out[0].float().cpu().numpy()          # (C, T) at sr_out
    if audio.shape[0] == 1:
        audio = np.vstack([audio, audio])
    g = math.gcd(sr_out, SAMPLE_RATE)
    audio = resample_poly(audio, SAMPLE_RATE // g, sr_out // g, axis=1).astype(np.float32)
    peak = float(np.abs(audio).max()) or 1.0
    audio = (audio / peak * 0.9).astype(np.float32)
    sf.write(str(path), audio.T, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    if log:
        log(f"neural clip {model_key}: {seconds:.0f} s in {gen_s:.0f} s (load {load_s:.0f} s) -> {path.name}")
    return audio, {"cached": False, "path": str(path), "model": model_key, "load_s": round(load_s, 1), "gen_s": round(gen_s, 1)}


# ---- fitting to the plan (pure numpy / librosa, unit-tested) ------------------------------------

def estimate_bpm(audio: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    from tmg.library import analysis

    mono = analysis.to_mono_22k(audio, sr)
    bpm, _beats, _ = analysis.estimate_beats(mono)
    return float(bpm)


def stretch_to_bpm(audio: np.ndarray, clip_bpm: float, target_bpm: float, limit: float = 0.12) -> tuple[np.ndarray, float]:
    """Time-stretch so the clip runs at the plan's tempo when its own is close enough (halves/doubles too)."""
    candidates = [clip_bpm, clip_bpm * 2, clip_bpm / 2]
    best = min(candidates, key=lambda b: abs(b / target_bpm - 1.0))
    ratio = target_bpm / best
    if abs(ratio - 1.0) > limit or abs(ratio - 1.0) < 1e-3:
        return audio, 1.0
    import librosa

    out = [librosa.effects.time_stretch(ch.astype(np.float32), rate=ratio) for ch in audio]
    n = min(len(o) for o in out)
    return np.vstack([o[:n] for o in out]).astype(np.float32), ratio


def loop_to_length(audio: np.ndarray, target_samples: int, bar_samples: int, fade_samples: int | None = None) -> np.ndarray:
    """Bar-aligned loop with an equal-power crossfade of one bar at each seam, to exactly target_samples."""
    if audio.shape[1] == 0:
        return np.zeros((2, target_samples), dtype=np.float32)
    bars = max(1, audio.shape[1] // bar_samples)
    body = audio[:, : bars * bar_samples].astype(np.float32)
    fade = fade_samples if fade_samples is not None else min(bar_samples, body.shape[1] // 4)
    out = np.zeros((2, target_samples + body.shape[1] + fade), dtype=np.float32)
    pos = 0
    ramp_in = np.sqrt(np.linspace(0.0, 1.0, fade, dtype=np.float32))
    ramp_out = np.sqrt(np.linspace(1.0, 0.0, fade, dtype=np.float32))
    while pos < target_samples:
        piece = body.copy()
        if pos > 0 and fade > 0:
            piece[:, :fade] *= ramp_in
        if fade > 0:
            piece[:, -fade:] *= ramp_out
        out[:, pos:pos + piece.shape[1]] += piece
        pos += piece.shape[1] - fade
    return out[:, :target_samples]


def build_layer(timeline: dict, clips: dict[str, np.ndarray], *, total_samples: int, hpf_hz: float = 150.0,
                fade_bars: float = 1.0, duck_depth: float = 0.55) -> np.ndarray:
    """(2, total_samples): the neural layer aligned to the sections, faded, high-passed, ducked by the kick."""
    layer = np.zeros((2, total_samples), dtype=np.float32)
    bar_s = float(timeline["bar_s"])
    bar_samples = max(1, int(bar_s * SAMPLE_RATE))
    fade = max(1, int(fade_bars * bar_s * SAMPLE_RATE))
    for s in timeline.get("sections") or []:
        role = next((r for r, kinds in ROLE_SECTIONS.items() if s["type"] in kinds), None)
        clip = clips.get(role) if role else None
        if clip is None or clip.shape[1] == 0:
            continue
        a, b = int(float(s["start_s"]) * SAMPLE_RATE), int(float(s["end_s"]) * SAMPLE_RATE)
        a, b = max(0, a), min(total_samples, b)
        if b - a <= fade * 2:
            continue
        piece = loop_to_length(clip, b - a, bar_samples, fade_samples=min(fade, clip.shape[1] // 3))
        gain = 10 ** (ROLE_LEVEL_DB.get(s["type"], 0.0) / 20.0)
        env = np.ones(b - a, dtype=np.float32)
        env[:fade] = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        env[-fade:] = np.linspace(1.0, 0.0, fade, dtype=np.float32)
        layer[:, a:b] += piece * env * gain
    if hpf_hz > 0:
        for ch in range(2):
            layer[ch] = synth.highpass(layer[ch].astype(np.float64), hpf_hz).astype(np.float32)
    kicks = np.asarray([int(float(k) * SAMPLE_RATE) for k in timeline.get("kicks") or []], dtype=np.int64)
    if kicks.size:
        duck = synth.sidechain(total_samples, kicks, depth=duck_depth, release_s=0.25).astype(np.float32)
        layer *= duck
    return layer


def mix_layer(audio: np.ndarray, layer: np.ndarray, level_db: float) -> np.ndarray:
    """audio (T, 2) from the composer + layer (2, T) at level_db relative to the mix peak -> (T, 2), peak 0.89."""
    mix = audio.T.astype(np.float32).copy()
    n = min(mix.shape[1], layer.shape[1])
    lpeak = float(np.abs(layer).max()) or 1.0
    mix[:, :n] += layer[:, :n] * (0.89 * 10 ** (level_db / 20.0) / lpeak)
    np.tanh(mix * 1.1, out=mix)
    mix /= math.tanh(1.1)
    loudest = max(float(np.abs(mix).max()), synth.SILENCE)
    mix *= 0.89 / loudest
    return mix.T.copy()


def excerpt_for_melody(audio: np.ndarray, timeline: dict, role: str, seconds: float = CLIP_SECONDS) -> np.ndarray | None:
    """A slice of our own rendered track (T, 2) that the melody model should follow: the first section of the role."""
    for s in timeline.get("sections") or []:
        if s["type"] in ROLE_SECTIONS[role]:
            a = int(float(s["start_s"]) * SAMPLE_RATE)
            b = min(audio.shape[0], a + int(seconds * SAMPLE_RATE))
            if b - a > SAMPLE_RATE * 4:
                return audio[a:b].T.astype(np.float32)
    return None


def apply(audio: np.ndarray, timeline: dict, plan, *, model_key: str = "stereo-small", level_db: float = -10.0,
          energy_too: bool = True, seed: int = 0, log=None, progress=None) -> tuple[np.ndarray, dict]:
    """The whole thing: clips -> fit -> layer -> mix. Returns (audio (T, 2), info)."""
    if model_key not in MODELS:
        raise ValueError(f"unknown neural model {model_key!r}")
    total = audio.shape[0]
    bar_s = float(timeline["bar_s"])
    roles = ["atmosphere"] + (["energy"] if energy_too else [])
    clips: dict[str, np.ndarray] = {}
    info: dict = {"model": model_key, "clips": {}}
    for i, role in enumerate(roles):
        if progress:
            progress(i / len(roles), f"neural {role} ({MODELS[model_key]['label'].split(' (')[0]})")
        prompt = prompt_for(plan, role)
        melody = excerpt_for_melody(audio, timeline, role) if MODELS[model_key]["melody"] else None
        clip, meta = generate(model_key, prompt, CLIP_SECONDS, seed + i * 11, melody=melody, log=log)
        bpm = estimate_bpm(clip)
        fitted, ratio = stretch_to_bpm(clip, bpm, plan.bpm)
        clips[role] = fitted
        info["clips"][role] = {"prompt": prompt, "clip_bpm": round(bpm, 1), "stretch": round(ratio, 3), **meta}
        if log:
            log(f"neural {role}: clip {bpm:.1f} bpm -> {'stretched x%.3f' % ratio if ratio != 1.0 else 'used as texture'}")
    layer = build_layer(timeline, clips, total_samples=total)
    unload()
    return mix_layer(audio, layer, level_db), {**info, "level_db": level_db, "bar_s": bar_s}


__all__ = ["CLIP_SECONDS", "MODELS", "apply", "build_layer", "estimate_bpm", "generate", "loop_to_length", "mix_layer",
           "prompt_for", "stretch_to_bpm", "unload"]
