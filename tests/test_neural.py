import math

import numpy as np

from tmg.music import neural
from tmg.music import plan as planmod
from tmg.music.synth import SAMPLE_RATE

BAR = 240 / 138
TIMELINE = {
    "bpm": 138.0, "bar_s": BAR, "duration_s": 40 * BAR,
    "sections": [
        {"type": "intro", "start_bar": 0, "bars": 8, "start_s": 0.0, "end_s": 8 * BAR},
        {"type": "drop", "start_bar": 8, "bars": 16, "start_s": 8 * BAR, "end_s": 24 * BAR},
        {"type": "breakdown", "start_bar": 24, "bars": 8, "start_s": 24 * BAR, "end_s": 32 * BAR},
        {"type": "outro", "start_bar": 32, "bars": 8, "start_s": 32 * BAR, "end_s": 40 * BAR},
    ],
    "kicks": [8 * BAR + k * BAR / 4 for k in range(64)],
}


def _tone(freq, seconds, amp=0.5):
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    y = (amp * np.sin(2 * math.pi * freq * t)).astype(np.float32)
    return np.vstack([y, y])


def test_prompts_carry_the_plan():
    p = planmod.build_plan(None, 3, minutes=4.0, flavor="psy")
    a, e = neural.prompt_for(p, "atmosphere"), neural.prompt_for(p, "energy")
    assert p.key_name in a and str(int(round(p.bpm))) in a and "no drums" in a
    assert "psy" in e and "no vocals" in e and a != e


def test_stretch_to_bpm():
    clip = _tone(220, 2.0)
    out, ratio = neural.stretch_to_bpm(clip, 130.0, 138.0)
    assert abs(ratio - 138 / 130) < 1e-6 and out.shape[0] == 2
    assert abs(out.shape[1] / clip.shape[1] - 130 / 138) < 0.03            # faster -> shorter
    same, r1 = neural.stretch_to_bpm(clip, 138.0, 138.0)
    assert r1 == 1.0 and same is clip
    far, r2 = neural.stretch_to_bpm(clip, 100.0, 138.0)
    assert r2 == 1.0 and far is clip                                         # too far: a free texture
    half, r3 = neural.stretch_to_bpm(clip, 69.0, 138.0)
    assert r3 == 1.0                                                         # half-time reading of the same tempo: nothing to do


def test_loop_to_length_is_exact_and_seamless():
    bar = int(BAR * SAMPLE_RATE)
    clip = _tone(110, 4 * BAR, amp=0.4)
    target = 10 * bar + 123
    out = neural.loop_to_length(clip, target, bar)
    assert out.shape == (2, target)
    assert np.abs(out).max() <= 0.4 * math.sqrt(2) + 0.01                  # equal-power seams: at most +3 dB on a coherent tone
    assert np.abs(out[:, 6 * bar: 6 * bar + 1000]).max() > 0.2               # still playing deep into the loop
    assert neural.loop_to_length(np.zeros((2, 0), dtype=np.float32), 500, bar).shape == (2, 500)


def test_build_layer_and_mix():
    total = int(TIMELINE["duration_s"] * SAMPLE_RATE)
    clips = {"atmosphere": _tone(440, 6.0, 0.5), "energy": _tone(880, 6.0, 0.5)}
    layer = neural.build_layer(TIMELINE, clips, total_samples=total, hpf_hz=0.0, duck_depth=0.0)
    assert layer.shape == (2, total)
    bar = int(BAR * SAMPLE_RATE)
    intro_mid = np.abs(layer[0, 4 * bar: 4 * bar + 2000]).max()
    drop_mid = np.abs(layer[0, 16 * bar: 16 * bar + 2000]).max()
    assert intro_mid > 0.2 and drop_mid > 0.05
    assert drop_mid < intro_mid                                              # drops sit lower (-8 dB) than intros (-2 dB)
    assert np.abs(layer[0, :50]).max() < 0.01                                # fades in from silence
    ducked = neural.build_layer(TIMELINE, clips, total_samples=total, hpf_hz=0.0, duck_depth=0.6)
    k = int(8 * BAR * SAMPLE_RATE) + 200                                     # right after the first kick
    assert np.abs(ducked[0, k: k + 400]).max() < np.abs(layer[0, k: k + 400]).max()
    music = np.zeros((total, 2), dtype=np.float32)
    music[:, 0] = music[:, 1] = 0.6 * np.sin(2 * math.pi * 55 * np.arange(total) / SAMPLE_RATE)
    mixed = neural.mix_layer(music, layer, level_db=-6.0)
    assert mixed.shape == music.shape and abs(np.abs(mixed).max() - 0.89) < 0.01
    x = neural.excerpt_for_melody(music, TIMELINE, "energy")
    assert x is not None and x.shape[0] == 2 and abs(x.shape[1] / SAMPLE_RATE - 16 * BAR) < 0.01 or x.shape[1] == 30 * SAMPLE_RATE


def test_models_table():
    assert set(neural.MODELS) == {"stereo-small", "medium", "melody"}
    assert neural.MODELS["melody"]["melody"] and not neural.MODELS["medium"]["melody"]
