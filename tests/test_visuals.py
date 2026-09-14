import json
import math

import numpy as np
import pytest

from tmg.visuals import audio_features, sequencer, signals
from tmg.visuals import palettes as palmod

BAR = 240 / 140
TIMELINE = {
    "bpm": 140.0, "bar_s": BAR, "duration_s": 72 * BAR,
    "sections": [
        {"type": "intro", "start_bar": 0, "bars": 8, "start_s": 0.0, "end_s": 8 * BAR},
        {"type": "build", "start_bar": 8, "bars": 16, "start_s": 8 * BAR, "end_s": 24 * BAR},
        {"type": "drop", "start_bar": 24, "bars": 32, "start_s": 24 * BAR, "end_s": 56 * BAR},
        {"type": "breakdown", "start_bar": 56, "bars": 8, "start_s": 56 * BAR, "end_s": 64 * BAR},
        {"type": "outro", "start_bar": 64, "bars": 8, "start_s": 64 * BAR, "end_s": 72 * BAR},
    ],
    "kicks": [24 * BAR + k * BAR / 4 for k in range(128)],
    "fills": [31 * BAR + 3 * BAR / 4],
    "crashes": [24 * BAR],
    "risers": [[16 * BAR, 24 * BAR]],
    "phrases": [{"start_s": 57 * BAR, "end_s": 59 * BAR}],
}


def test_signals_follow_the_timeline():
    fps = 30
    sig = signals.build_signals(TIMELINE, fps)
    n = int(math.ceil(TIMELINE["duration_s"] * fps))
    assert len(sig["kick"]) == n
    f_drop = int(24 * BAR * fps) + 1
    assert sig["kick"][f_drop] > 0.8 and sig["impact"][f_drop] > 0.8        # first kick of the drop + the drop hit
    assert sig["kick"][f_drop + 10] < 0.2                                    # decayed a third of a second later (kicks are 0.43 s apart)
    assert sig["kick"].max() <= 1.0
    assert sig["drop"][f_drop] == 1.0 and sig["drop"][int(4 * BAR * fps)] == 0.0
    assert sig["energy_raw"][int(2 * BAR * fps)] == 0.25                     # intro
    assert sig["energy_raw"][int(23 * BAR * fps)] > 0.9                      # riser climbing into the drop
    assert sig["phrase"][int(58 * BAR * fps)] == 1.0 and sig["phrase"][int(50 * BAR * fps)] == 0.0
    assert 0.0 <= sig["beat_phase"].min() and sig["beat_phase"].max() < 1.0
    assert sig["section_index"][int(60 * BAR * fps)] == 3
    assert sig["fill"][int((31 * BAR + 3 * BAR / 4) * fps) + 1] > 0.8


def test_spectrum_frames():
    sr, fps = 22050, 30
    t = np.arange(sr * 2) / sr
    y = (0.5 * np.sin(2 * math.pi * 100 * t) + 0.2 * np.sin(2 * math.pi * 5000 * t)).astype(np.float32)
    y[sr:] *= 0.05
    spec, rms = audio_features.spectrum_frames(y, sr, fps, 60)
    assert spec.shape == (60, 64) and rms.shape == (60,)
    assert spec.min() >= 0 and spec.max() <= 1
    loud, quiet = spec[10], spec[50]
    assert loud.max() > quiet.max()
    assert rms[10] > rms[50] * 5


def test_palettes(tmp_path):
    path = tmp_path / "palettes.json"
    pals = palmod.load_palettes(path)                 # written with defaults on first use
    assert path.exists() and len(pals) >= 4
    lut = pals[0].lut()
    assert lut.shape == (256, 3) and lut.dtype == np.uint8
    assert tuple(lut[0]) == palmod.hex_to_rgb(pals[0].colors[0]) and tuple(lut[-1]) == palmod.hex_to_rgb(pals[0].colors[-1])
    path.write_text(json.dumps({"palettes": [{"id": "x", "name": "X", "colors": ["#000000", "#ffffff"]}, {"id": "bad", "colors": ["#123"]}]}))
    pals2 = palmod.load_palettes(path)
    assert [p.id for p in pals2] == ["x"]


def test_sequencer_covers_timeline_by_mood():
    enabled = {n: w for n, (m, w, _d) in sequencer.GENERATORS.items() if n != "stills"}
    shots = sequencer.build_shots(TIMELINE, enabled, ["deep_ocean", "ember"], seed=5)
    assert shots[0].start_s == 0.0 and abs(shots[-1].end_s - TIMELINE["duration_s"]) < 1e-3
    for a, b in zip(shots, shots[1:]):
        assert abs(a.end_s - b.start_s) < 1e-3 and a.generator != b.generator
    for s in shots:
        mood = sequencer.GENERATORS[s.generator][0]
        expected = signals.SECTION_MOOD[s.section_type]
        assert mood == expected, (s.section_type, s.generator)
    drop_shots = [s for s in shots if s.section_type == "drop"]
    assert len(drop_shots) == 2 and drop_shots[0].cut and not drop_shots[1].cut        # 32 bars -> two 16-bar shots
    only_calm = sequencer.build_shots(TIMELINE, {"domainwarp": 1.0}, ["deep_ocean"], seed=1)
    assert {s.generator for s in only_calm} == {"domainwarp"}                            # borrowed for every mood
    assert sequencer.build_shots(TIMELINE, {}, ["deep_ocean"], seed=1)[0].generator == "plasma"


@pytest.fixture(scope="module")
def gpu():
    try:
        import moderngl

        ctx = moderngl.create_context(standalone=True, backend="egl")
        ctx.release()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no EGL/GPU context: {exc}")
    return True


def test_every_generator_renders_on_the_gpu(gpu):
    from tmg.visuals import shaders
    from tmg.visuals.engine import Engine

    luts = {p.id: p.lut() for p in palmod.load_palettes()}
    img = (np.random.default_rng(0).random((90, 160, 3)) * 255).astype(np.uint8)
    eng = Engine(160, 90, luts, [img])
    u = {"time": 2.0, "kick": 0.5, "energy": 0.8, "impact": 0.2, "drop": 1.0, "phrase": 0.0, "fillx": 0.0, "beat": 0.2,
         "barphase": 0.3, "bars": 9.0, "rms": 0.5, "intensity": 0.9, "imgmix": 0.5}
    eng.set_spectrum(np.linspace(0, 1, 64).astype(np.float32))
    try:
        for name in shaders.GENERATORS:
            eng.begin_shot("a", name, "deep_ocean", 3, 0)
            for k in range(20):
                frame = eng.render_frame(u, k, 30.0, 0.0)
            arr = np.frombuffer(frame, dtype=np.uint8).reshape(90, 160, 3)
            assert arr.std() > 4, f"{name} produced a flat picture"
        eng.begin_shot("b", "tunnel", "ember", 4, 0)
        assert len(eng.render_frame(u, 21, 30.0, 0.5)) == 160 * 90 * 3
    finally:
        eng.release()


def test_render_short_video(gpu, tmp_path):
    import soundfile as sf

    from tmg import paths
    from tmg.visuals import renderer

    sr = 44100
    t = np.arange(int(6 * sr)) / sr
    audio = (0.3 * np.sin(2 * math.pi * 55 * t) * (np.sin(2 * math.pi * 2.33 * t) > 0)).astype(np.float32)
    wav = tmp_path / "a.wav"
    sf.write(str(wav), np.stack([audio, audio], 1), sr)
    tl = dict(TIMELINE)
    out = tmp_path / "v.mp4"
    summary = renderer.render_video(str(wav), tl, str(out), enabled={"plasma": 1.0, "tunnel": 1.0, "domainwarp": 1.0, "flow": 1.0},
                                    width=320, height=180, fps=30, codec="h264", bitrate_k=1500, seed=2, max_seconds=6.0)
    assert out.exists() and out.stat().st_size > 20_000
    assert summary["frames"] == 180 and summary["fps_achieved"] > 20
    import subprocess

    info = subprocess.run([str(paths.FFPROBE), "-v", "error", "-show_entries", "stream=codec_name,width,height,r_frame_rate",
                           "-of", "json", str(out)], capture_output=True, text=True, check=True).stdout
    streams = json.loads(info)["streams"]
    codecs = {s["codec_name"] for s in streams}
    assert "h264" in codecs and "aac" in codecs
    v = next(s for s in streams if s["codec_name"] == "h264")
    assert (v["width"], v["height"]) == (320, 180) and v["r_frame_rate"] == "30/1"
