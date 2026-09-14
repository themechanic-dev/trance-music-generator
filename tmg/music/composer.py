"""Arrangement: from a Plan (drawn from the profile) to stereo audio plus a timeline.

Carried over from the AutoDJ composer and changed in two ways: every rhythm comes from the plan's
learned 16-step patterns instead of fixed loops, and the render returns a timeline (sections, bars,
beats, kicks, fills, transitions) so the visuals can follow the music without analysing it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from tmg.music import synth, voices
from tmg.music.plan import Plan, Section, hits, scale_note, triad
from tmg.music.synth import SAMPLE_RATE

BEATS_PER_BAR = 4
STEPS_PER_BAR = 16
PHRASE_BARS = 4
LONG_PHRASE = 8
BREAKDOWN_LIFT = 2.3


@dataclass
class _Canvas:
    drums: np.ndarray
    low: np.ndarray
    mid_l: np.ndarray
    mid_r: np.ndarray
    high_l: np.ndarray
    high_r: np.ndarray
    effects: np.ndarray
    kick_hits: list[int] = field(default_factory=list)

    @classmethod
    def blank(cls, samples: int) -> _Canvas:
        return cls(*(np.zeros(samples, dtype=np.float32) for _ in range(7)))


@dataclass(frozen=True)
class _Kit:
    kick: np.ndarray
    clap: np.ndarray
    closed_hat: np.ndarray
    open_hat: np.ndarray
    ride: np.ndarray

    @classmethod
    def build(cls, rng: np.random.Generator, *, punch: float = 1.0) -> _Kit:
        return cls(voices.kick(punch=punch), voices.clap(rng), voices.hat(rng), voices.hat(rng, open_=True), voices.ride(rng))


@dataclass
class Timeline:
    """What happens when - the contract between the music and the visuals (and the phrases, phase 4)."""

    bpm: float
    bar_s: float
    duration_s: float
    sections: list[dict] = field(default_factory=list)
    bars: list[float] = field(default_factory=list)
    beats: list[float] = field(default_factory=list)
    kicks: list[float] = field(default_factory=list)
    fills: list[float] = field(default_factory=list)
    crashes: list[float] = field(default_factory=list)
    risers: list[list[float]] = field(default_factory=list)
    sweeps: list[list[float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "bpm": self.bpm, "bar_s": round(self.bar_s, 6), "duration_s": round(self.duration_s, 3),
            "sections": self.sections, "bars": [round(b, 4) for b in self.bars], "beats": [round(b, 4) for b in self.beats],
            "kicks": [round(k, 4) for k in self.kicks], "fills": [round(f, 4) for f in self.fills],
            "crashes": [round(c, 4) for c in self.crashes], "risers": self.risers, "sweeps": self.sweeps,
        }


def _steps(pattern: str) -> list[int]:
    return [i for i, ch in enumerate(pattern[:STEPS_PER_BAR]) if ch == "1"]


def _lay_drums(canvas: _Canvas, kit: _Kit, section: Section, rng, tl: Timeline, *, at_bar: float, beat: float,
               bar_no: int, opening: float, level: float, bpm: float, sr: int) -> None:
    is_fill = bar_no % LONG_PHRASE == LONG_PHRASE - 1
    step_s = beat / 4.0
    if section.kick:
        for step in _steps(section.kick_pattern):
            if is_fill and step >= STEPS_PER_BAR - 4:      # the fill takes the last beat
                continue
            t = at_bar + step * step_s
            at = int(t * sr)
            canvas.kick_hits.append(at)
            tl.kicks.append(t)
            synth.place(canvas.drums, kit.kick * level, at)
    if section.clap and hits(section.clap_pattern):
        for step in _steps(section.clap_pattern):
            if is_fill and step >= STEPS_PER_BAR - 4:
                continue
            synth.place(canvas.drums, kit.clap * 0.55 * level, int((at_bar + step * step_s) * sr))
    if section.hats:
        pattern = section.hat_pattern if hits(section.hat_pattern) else "0010001000100010"
        in_phrase = bar_no % LONG_PHRASE
        last_step = max(_steps(pattern))
        for step in _steps(pattern):
            marks_phrase = step == last_step and bar_no % PHRASE_BARS == PHRASE_BARS - 1
            sound = kit.open_hat if marks_phrase else kit.closed_hat
            synth.place(canvas.drums, sound * opening * level, int((at_bar + step * step_s) * sr))
        if in_phrase >= LONG_PHRASE // 2:                 # sixteenths arrive halfway through the phrase
            for step in range(1, STEPS_PER_BAR, 2):
                synth.place(canvas.drums, kit.closed_hat * 0.4 * opening * level, int((at_bar + step * step_s) * sr))
        if bar_no // LONG_PHRASE >= 1 and section.kick:
            for index in range(BEATS_PER_BAR):
                synth.place(canvas.drums, kit.ride * opening * level, int((at_bar + index * beat) * sr))
    if is_fill and (section.kick or section.hats):
        t = at_bar + (BEATS_PER_BAR - 1) * beat
        tl.fills.append(t)
        synth.place(canvas.drums, voices.roll(rng, beats=1.0, bpm=bpm) * 0.5 * level, int(t * sr))


def _lay_bass(canvas: _Canvas, section: Section, chord: list[int], rng, *, at_bar: float, beat: float,
              bar_no: int, level: float, sr: int) -> None:
    """The learned bass pattern, played on the chord root: each hit lasts until the next one."""
    root = chord[0] - 12
    steps = _steps(section.bass_pattern) or [2, 6, 10, 14]
    is_fill = bar_no % LONG_PHRASE == LONG_PHRASE - 1
    step_s = beat / 4.0
    dense = len(steps) >= 8
    for i, step in enumerate(steps):
        until = steps[i + 1] if i + 1 < len(steps) else STEPS_PER_BAR
        length = max(0.19, min(0.95, (until - step) * 0.9)) * beat
        if dense:
            length = min(length, beat * 0.22)              # rolling sixteenths stay short and clean
        pitch = root + 12 if (step == steps[-1] and bar_no % 2 == 1) else root
        if is_fill and step >= STEPS_PER_BAR - 4:
            pitch += 3 * ((step - (STEPS_PER_BAR - 4)) % 2)
        note = voices.bass(voices.hz(pitch), length, rng)
        synth.place(canvas.low, note * (0.95 if dense else 0.9) * level, int((at_bar + step * step_s) * sr))


def _lay_arp(canvas: _Canvas, spec: Plan, chord: list[int], rng, *, at_bar: float, beat: float, opening: float, level: float, sr: int) -> None:
    notes = [chord[0], chord[1], chord[2], chord[0] + 12]
    half = STEPS_PER_BAR // 2
    for step in range(STEPS_PER_BAR):
        which = spec.arp_shape[step % len(spec.arp_shape)]
        octave = 12 if step % half >= half // 2 else 0
        voice = voices.pluck(voices.hz(notes[which] + 12 + octave), beat * spec.flavor.arp_length, rng)
        at = int((at_bar + step * beat / 4.0) * sr)
        near, far = (0.92, 0.38) if step % 2 == 0 else (0.38, 0.92)
        gain = 0.42 * opening * level
        synth.place(canvas.mid_l, voice * gain * near, at)
        synth.place(canvas.mid_r, voice * gain * far, at)


def _lay_chords(canvas: _Canvas, spec: Plan, section: Section, chord: list[int], chord_degree: int, rng, *,
                at_bar: float, bar: float, bar_no: int, opening: float, level: float, sr: int) -> None:
    carrying = BREAKDOWN_LIFT if not section.kick else 1.0
    if section.pad and bar_no % 2 == 0:
        gain = 0.45 * opening * level * carrying
        left, right = voices.pad([voices.hz(n) for n in chord], bar * 2.0, rng, cutoff=spec.flavor.pad_cutoff)
        synth.place_stereo(canvas.mid_l, canvas.mid_r, (left * gain, right * gain), int(at_bar * sr))
        if carrying > 1.0:
            below_l, below_r = voices.pad([voices.hz(chord[0] - 12)], bar * 2.0, rng)
            synth.place(canvas.low, (below_l + below_r) * 0.15 * level, int(at_bar * sr))
    if section.lead and bar_no % 2 == 0:
        _lay_lead(canvas, spec, chord_degree, rng, at_bar=at_bar, beat=bar / BEATS_PER_BAR, level=level * carrying,
                  answering=(bar_no % LONG_PHRASE) >= LONG_PHRASE - 2, sr=sr)


def _lay_lead(canvas: _Canvas, spec: Plan, chord_degree: int, rng, *, at_bar: float, beat: float, level: float, answering: bool, sr: int) -> None:
    onsets = spec.lead_rhythm
    span = STEPS_PER_BAR * 2
    for index, step in enumerate(onsets):
        offset = spec.lead_contour[index % len(spec.lead_contour)]
        if answering and index == len(onsets) - 1:
            offset = 0
        note = scale_note(spec.root_midi, spec.scale, chord_degree + offset) + 24
        until = onsets[index + 1] if index + 1 < len(onsets) else span
        length = max(0.12, (until - step) * beat / 4.0)
        at = int((at_bar + step * beat / 4.0) * sr)
        gain = 0.34 * level * spec.flavor.lead_gain
        if spec.flavor.acid_lead:
            voice = voices.acid(voices.hz(note - 12), min(length, beat), rng)
            synth.place(canvas.high_l, voice * gain, at)
            synth.place(canvas.high_r, voice * gain, at)
            continue
        left, right = voices.lead(voices.hz(note), min(length, beat * 2.0), rng)
        synth.place_stereo(canvas.high_l, canvas.high_r, (left * gain, right * gain), at)


def _ping_pong(left, right, *, time_s, feedback, repeats, mix) -> None:
    echo_l = synth.delay_line(right, time_s=time_s, feedback=feedback, repeats=repeats)
    echo_r = synth.delay_line(left, time_s=time_s, feedback=feedback, repeats=repeats)
    echo_l *= mix
    echo_r *= mix
    left += echo_l
    right += echo_r


def _mixdown(canvas: _Canvas, rng, *, beat: float) -> np.ndarray:
    duck = synth.sidechain(canvas.drums.size, np.array(canvas.kick_hits, dtype=np.int64), depth=0.62).astype(np.float32)
    canvas.low *= duck
    canvas.mid_l *= duck
    canvas.mid_r *= duck
    duck *= 0.85
    duck += 0.15
    canvas.high_l *= duck
    canvas.high_r *= duck
    del duck
    _ping_pong(canvas.mid_l, canvas.mid_r, time_s=beat * 0.75, feedback=0.3, repeats=5, mix=0.35)
    _ping_pong(canvas.high_l, canvas.high_r, time_s=beat * 1.5, feedback=0.34, repeats=4, mix=0.4)
    wet_in = canvas.mid_l * 0.35
    wet_in += canvas.high_l * 0.45
    left = synth.reverb(wet_in, rng=rng, seconds=2.2).astype(np.float32)
    left *= 0.5
    wet_in = canvas.mid_r * 0.35
    wet_in += canvas.high_r * 0.45
    right = synth.reverb(wet_in, rng=rng, seconds=2.2).astype(np.float32)
    right *= 0.5
    del wet_in
    centre = canvas.drums * 0.9
    centre += canvas.low
    centre += canvas.effects
    left += centre
    left += canvas.mid_l
    left += canvas.high_l
    right += centre
    right += canvas.mid_r
    right += canvas.high_r
    del centre
    left *= 0.62
    right *= 0.62
    np.tanh(left * 1.3, out=left)
    np.tanh(right * 1.3, out=right)
    left /= np.tanh(1.3)
    right /= np.tanh(1.3)
    loudest = max(float(np.abs(left).max()), float(np.abs(right).max()), synth.SILENCE)
    left *= 0.89 / loudest
    right *= 0.89 / loudest
    return synth.stereo(left, right)


def render(spec: Plan, progress: Callable[[float, str], None] | None = None) -> tuple[np.ndarray, Timeline]:
    """Build the whole track: stereo float32 (samples, 2) at 44.1 kHz, and its timeline."""
    rng = np.random.default_rng(spec.seed)
    sr = SAMPLE_RATE
    beat, bar = spec.beat_s, spec.bar_s
    canvas = _Canvas.blank(int(spec.duration_s * sr) + sr)
    kit = _Kit.build(rng, punch=spec.flavor.kick_punch)
    tl = Timeline(bpm=spec.bpm, bar_s=bar, duration_s=spec.duration_s + 1.0)

    bar_index = 0
    total_bars = spec.total_bars
    for s_index, section in enumerate(spec.sections):
        section_start = bar_index * bar
        tl.sections.append({"type": section.type, "start_bar": bar_index, "bars": section.bars,
                            "start_s": round(section_start, 4), "end_s": round(section_start + section.bars * bar, 4),
                            "level": section.level, "kick": section.kick,
                            "kick_pattern": section.kick_pattern, "bass_pattern": section.bass_pattern})
        for local_bar in range(section.bars):
            at_bar = section_start + local_bar * bar
            tl.bars.append(at_bar)
            tl.beats.extend(at_bar + q * beat for q in range(BEATS_PER_BAR))
            chord_degree = spec.progression[bar_index % len(spec.progression)]
            chord = triad(spec.root_midi, spec.scale, chord_degree)
            through = local_bar / max(1, section.bars - 1) if section.bars > 1 else 1.0
            opens_from, opens_to = section.filter_open
            opening = opens_from + through * (opens_to - opens_from)
            _lay_drums(canvas, kit, section, rng, tl, at_bar=at_bar, beat=beat, bar_no=local_bar, opening=opening,
                       level=section.level, bpm=spec.bpm, sr=sr)
            if section.bass:
                _lay_bass(canvas, section, chord, rng, at_bar=at_bar, beat=beat, bar_no=local_bar, level=section.level, sr=sr)
            if section.arp:
                _lay_arp(canvas, spec, chord, rng, at_bar=at_bar, beat=beat, opening=opening, level=section.level, sr=sr)
            _lay_chords(canvas, spec, section, chord, chord_degree, rng, at_bar=at_bar, bar=bar, bar_no=local_bar,
                        opening=opening, level=section.level, sr=sr)
            bar_index += 1
            if progress and bar_index % 8 == 0:
                progress(0.05 + 0.75 * bar_index / total_bars, f"bar {bar_index}/{total_bars} · {section.type}")
        ends_at = section_start + section.bars * bar
        if section.riser:
            length = min(section.bars, 8) * bar
            tl.risers.append([round(ends_at - length, 4), round(ends_at, 4)])
            synth.place(canvas.effects, voices.riser(length, rng) * 0.3, int((ends_at - length) * sr))
            tl.crashes.append(round(ends_at, 4))
            synth.place(canvas.effects, voices.crash(rng) * 0.4, int(ends_at * sr))
        elif not section.kick and s_index + 1 < len(spec.sections):
            tl.sweeps.append([round(ends_at - bar, 4), round(ends_at, 4)])
            synth.place(canvas.effects, voices.reverse_cymbal(rng, length_s=bar), int((ends_at - bar) * sr))
    if progress:
        progress(0.82, "mixdown: sidechain, echoes, reverb")
    audio = _mixdown(canvas, rng, beat=beat)
    return audio, tl


__all__ = ["Timeline", "render"]
