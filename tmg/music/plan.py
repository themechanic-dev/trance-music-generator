"""The plan: everything decided before a single sample is written - drawn from the profile, not from constants.

Where the AutoDJ composer had four progressions, three bass patterns, four forms and a tempo per style,
every one of those is now a sample from what the library taught us: the tempo histogram, the section
transitions and lengths, the per-bar kick / snare / hat / bass patterns, the keys, the bass movement.
Same seed, same profile -> same plan, always.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

MINOR = (0, 2, 3, 5, 7, 8, 10)
MAJOR = (0, 2, 4, 5, 7, 9, 11)
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
SECTION_TYPES = ("intro", "build", "drop", "breakdown", "outro")
STEPS = 16

FOUR_ON_THE_FLOOR = "1000100010001000"
OFFBEAT_BASS = "0010001000100010"
BACKBEAT = "0000100000001000"
OFFBEAT_HATS = "0010001000100010"

#: Melody is not learned (pitch tracking on polyphonic leads is unreliable) - these stay as in AutoDJ.
LEAD_RHYTHMS = (
    (0, 6, 8, 12, 16, 22, 24, 28),
    (0, 4, 6, 8, 16, 20, 22, 24),
    (0, 8, 12, 14, 16, 24, 28, 30),
    (0, 3, 6, 8, 12, 16, 19, 26),
)
CONTOURS = (
    (0, 2, 4, 2, 7, 4, 2, 0),
    (0, 4, 2, 7, 4, 2, 0, -3),
    (7, 4, 2, 0, 2, 4, 7, 9),
    (0, 0, 2, 4, 4, 2, 0, -1),
)
ARPS = (
    (0, 1, 2, 3, 2, 1, 2, 1),
    (0, 2, 1, 3, 0, 2, 1, 3),
    (3, 2, 1, 0, 1, 2, 3, 2),
    (0, 1, 0, 2, 0, 1, 3, 2),
)

#: Used when the library is still empty, so composing works from the first minute. Shaped like a
#: small uplifting/psy collection; replaced by the real profile as soon as tracks are analysed.
DEFAULT_PROFILE = {
    "n_tracks": 0,
    "tempo": {"hist": {"134": 1, "136": 2, "138": 3, "140": 3, "142": 2, "144": 1}, "median": 139.0},
    "duration": {"hist_minutes": {"5": 2, "6": 3, "7": 2}},
    "keys": {"A minor": 3, "F minor": 2, "D minor": 2, "G minor": 1, "C major": 1},
    "sections": {
        "sequences": {"intro>build>drop>breakdown>build>drop>outro": 3, "intro>drop>breakdown>drop>outro": 2},
        "lengths": {"intro": {"16": 3, "8": 2}, "build": {"16": 3, "8": 2}, "drop": {"32": 4, "48": 1},
                    "breakdown": {"16": 3, "32": 2, "8": 1}, "outro": {"16": 3, "8": 2}},
        "transitions": {"intro": {"build": 0.6, "drop": 0.4}, "build": {"drop": 1.0},
                        "drop": {"breakdown": 0.7, "outro": 0.3}, "breakdown": {"build": 0.6, "drop": 0.4}},
    },
    "patterns": {
        "kick": {FOUR_ON_THE_FLOOR: 0.8, "1000100010001010": 0.1, "1000100010001000": 0.1},
        "snare": {BACKBEAT: 0.7, "0000000000000000": 0.3},
        "hat": {OFFBEAT_HATS: 0.6, "0101010101010101": 0.3, "0000000000000000": 0.1},
        "bass": {OFFBEAT_BASS: 0.5, "0111011101110111": 0.3, "1000100010001000": 0.2},
    },
    "bass_movement": {"0,0,0,0": 0.5, "0,8,3,10": 0.2, "0,5,8,10": 0.15, "0,0,5,5": 0.15},
}

_SECTION_CAPS = {"intro": (8, 32), "build": (8, 32), "drop": (16, 48), "breakdown": (8, 32), "outro": (8, 24)}
_DEFAULT_LEN = {"intro": 16, "build": 16, "drop": 32, "breakdown": 16, "outro": 16}


@dataclass
class Section:
    """A stretch of bars, what plays in it, and the learned patterns it plays with."""

    type: str
    bars: int
    kick: bool
    bass: bool
    hats: bool
    clap: bool
    arp: bool
    pad: bool
    lead: bool
    riser: bool
    filter_open: tuple[float, float]
    level: float
    kick_pattern: str = FOUR_ON_THE_FLOOR
    bass_pattern: str = OFFBEAT_BASS
    hat_pattern: str = OFFBEAT_HATS
    clap_pattern: str = BACKBEAT


@dataclass(frozen=True)
class Flavor:
    """Timbre choices (the AutoDJ 'styles'), picked from the tempo and the seed - not from the library."""

    name: str
    kick_punch: float = 1.0
    lead_gain: float = 1.0
    arp_length: float = 0.3
    pad_cutoff: float = 2600.0
    acid_lead: bool = False


FLAVORS = {
    "uplifting": Flavor("uplifting"),
    "progressive": Flavor("progressive", kick_punch=0.55, lead_gain=0.6, arp_length=0.45, pad_cutoff=1800.0),
    "psy": Flavor("psy", kick_punch=1.3, arp_length=0.2, pad_cutoff=3200.0, acid_lead=True),
    "tech": Flavor("tech", kick_punch=1.15, lead_gain=0.4, arp_length=0.25, pad_cutoff=2200.0),
}


@dataclass
class Plan:
    seed: int
    bpm: float
    root_midi: int
    mode: str
    progression: tuple[int, ...]
    arp_shape: tuple[int, ...]
    lead_rhythm: tuple[int, ...]
    lead_contour: tuple[int, ...]
    sections: list[Section]
    flavor: Flavor
    target_minutes: float
    profile_tracks: int = 0
    bass_movement: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def scale(self) -> tuple[int, ...]:
        return MAJOR if self.mode == "major" else MINOR

    @property
    def beat_s(self) -> float:
        return 60.0 / self.bpm

    @property
    def bar_s(self) -> float:
        return self.beat_s * 4.0

    @property
    def total_bars(self) -> int:
        return sum(s.bars for s in self.sections)

    @property
    def duration_s(self) -> float:
        return self.total_bars * self.bar_s

    @property
    def key_name(self) -> str:
        return f"{NOTE_NAMES[self.root_midi % 12]} {self.mode}"

    @property
    def structure(self) -> str:
        return " > ".join(f"{s.type}({s.bars})" for s in self.sections)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["key"] = self.key_name
        d["structure"] = self.structure
        d["duration_s"] = round(self.duration_s, 2)
        d["total_bars"] = self.total_bars
        return d


# ---- sampling helpers ----------------------------------------------------------------------

def _weighted(rng: np.random.Generator, mapping: dict, exclude: set | None = None):
    items = [(k, float(v)) for k, v in (mapping or {}).items() if v and (not exclude or k not in exclude)]
    if not items:
        return None
    keys, weights = zip(*items)
    w = np.asarray(weights, dtype=float)
    return keys[int(rng.choice(len(keys), p=w / w.sum()))]


def hits(pattern: str) -> int:
    return pattern.count("1")


def _pattern(rng: np.random.Generator, mapping: dict, fallback: str, min_hits: int, allow_empty: bool = False) -> str:
    for _ in range(6):
        p = _weighted(rng, mapping)
        if p is None:
            break
        p = str(p)[:STEPS].ljust(STEPS, "0")
        if hits(p) >= min_hits or (allow_empty and hits(p) == 0):
            return p
    return fallback


def _length(rng: np.random.Generator, lengths: dict, kind: str) -> int:
    lo, hi = _SECTION_CAPS[kind]
    picked = _weighted(rng, {int(k): v for k, v in (lengths.get(kind) or {}).items()})
    n = int(picked) if picked is not None else _DEFAULT_LEN[kind]
    n = max(4, int(round(n / 4.0)) * 4)
    return max(lo, min(hi, n))


def semitones_to_degrees(offsets: list[int], scale: tuple[int, ...]) -> tuple[int, ...]:
    """Bass roots as semitones above the key -> nearest scale degrees (what the chords are built on)."""
    out = []
    for semi in offsets:
        semi %= 12
        best = min(range(len(scale)), key=lambda i: min((scale[i] - semi) % 12, (semi - scale[i]) % 12))
        out.append(best)
    return tuple(out)


def _walk_structure(rng: np.random.Generator, sections_prof: dict, target_bars: int) -> list[tuple[str, int]]:
    """A Markov walk over the learned transitions, with learned lengths, ending in an outro."""
    sequences = sections_prof.get("sequences") or {}
    firsts: dict[str, float] = {}
    for seq, n in sequences.items():
        first = seq.split(">")[0]
        if first in SECTION_TYPES:
            firsts[first] = firsts.get(first, 0.0) + float(n)
    transitions = sections_prof.get("transitions") or {}
    lengths = sections_prof.get("lengths") or {}
    current = _weighted(rng, firsts) or "intro"
    if current == "outro":
        current = "intro"
    out: list[tuple[str, int]] = []
    total = 0
    body_target = target_bars - 12            # the outro (added at the end) takes the rest
    while len(out) < 14:
        bars = _length(rng, lengths, current)
        remaining = body_target - total
        if current in ("drop", "breakdown") and bars > remaining + 8:
            bars = max(_SECTION_CAPS[current][0], (remaining // 4) * 4)   # do not blow past the target
        out.append((current, bars))
        total += bars
        remaining = body_target - total
        if remaining <= 8:
            break
        if remaining < 48:
            _land(out, current, remaining)
            break
        exclude = {current, "intro"}
        if remaining > 32:
            exclude.add("outro")            # the outro is only allowed once the track is nearly long enough
        nxt = _weighted(rng, transitions.get(current) or {}, exclude=exclude)
        if nxt is None or nxt not in SECTION_TYPES:
            nxt = {"intro": "build", "build": "drop", "drop": "breakdown", "breakdown": "drop"}.get(current, "drop")
        if nxt == "outro":
            _land(out, current, remaining)
            break
        current = nxt
    return _finish(out, rng, lengths)


def _land(out: list[tuple[str, int]], current: str, remaining: int) -> None:
    """Close to the end: stretch the playing drop, or land one last drop, so the length is met."""
    extra = (remaining // 4) * 4
    if extra <= 0:
        return
    if current == "drop":
        kind, n = out[-1]
        grown = min(64, n + extra)
        out[-1] = (kind, grown)
        left = extra - (grown - n)
        if left >= 16:                    # a drop cannot be stretched forever: breathe, then one more drop
            out.append(("breakdown", 8))
            out.append(("drop", max(16, left - 8)))
    elif current in ("build", "breakdown"):
        out.append(("drop", max(16, extra)))
    elif current == "intro":
        out.append(("build", 8))
        out.append(("drop", max(16, extra - 8)))


def _finish(out: list[tuple[str, int]], rng: np.random.Generator, lengths: dict) -> list[tuple[str, int]]:
    if out and out[-1][0] == "outro":
        body, tail = out[:-1], [out[-1]]
    else:
        body, tail = out, []
    if not any(t == "drop" for t, _ in body):
        body.append(("build", _length(rng, lengths, "build")))
        body.append(("drop", _length(rng, lengths, "drop")))
    if body[-1][0] == "build":            # a build that never arrives is a broken promise
        body.append(("drop", max(16, _length(rng, lengths, "drop"))))
    if not tail:
        tail = [("outro", _length(rng, lengths, "outro"))]
    return body + tail


def _section(kind: str, bars: int, next_kind: str | None, patterns: dict[str, str], rng: np.random.Generator, prof_patterns: dict) -> Section:
    bass_pat = _pattern(rng, prof_patterns.get("bass"), OFFBEAT_BASS, 1)
    hat_pat = _pattern(rng, prof_patterns.get("hat"), OFFBEAT_HATS, 1, allow_empty=True)
    common = {"kick_pattern": patterns["kick"], "clap_pattern": patterns["clap"], "bass_pattern": bass_pat, "hat_pattern": hat_pat}
    if kind == "intro":
        return Section(kind, bars, kick=False, bass=False, hats=True, clap=False, arp=True, pad=True, lead=False, riser=False,
                       filter_open=(0.25, 0.6), level=0.62, **common)
    if kind == "build":
        return Section(kind, bars, kick=True, bass=True, hats=True, clap=True, arp=True, pad=True, lead=False, riser=True,
                       filter_open=(0.5, 1.0), level=0.85, **common)
    if kind == "drop":
        return Section(kind, bars, kick=True, bass=True, hats=True, clap=True, arp=True, pad=True, lead=True, riser=False,
                       filter_open=(1.0, 1.0), level=1.0, **common)
    if kind == "breakdown":
        return Section(kind, bars, kick=False, bass=False, hats=False, clap=False, arp=True, pad=True, lead=True,
                       riser=next_kind in ("drop", "build"), filter_open=(0.6, 1.0), level=1.0, **common)
    return Section("outro", bars, kick=False, bass=False, hats=True, clap=False, arp=True, pad=True, lead=False, riser=False,
                   filter_open=(1.0, 0.35), level=0.7, **common)


def build_plan(profile: dict | None, seed: int, *, minutes: float | None = None, flavor: str | None = None) -> Plan:
    """Draw one track from the profile. Deterministic in (profile, seed, minutes, flavor)."""
    prof = profile if profile and profile.get("n_tracks") else DEFAULT_PROFILE
    rng = np.random.default_rng(seed)
    notes: list[str] = []

    # tempo: a histogram bin, then somewhere inside it
    tempo_hist = {int(k): v for k, v in (prof.get("tempo", {}).get("hist") or {}).items()}
    bin_lo = _weighted(rng, tempo_hist)
    if bin_lo is None:
        bin_lo = 138
        notes.append("no tempo data - 138")
    bpm = round(float(bin_lo) + float(rng.uniform(0.0, 2.0)), 1)

    # key
    key = _weighted(rng, prof.get("keys") or {}) or "A minor"
    tonic_name, _, mode = str(key).partition(" ")
    mode = "major" if mode == "major" else "minor"
    pc = NOTE_NAMES.index(tonic_name) if tonic_name in NOTE_NAMES else 9
    root_midi = 45 + ((pc - 9) % 12)          # A2 .. G#3, low enough to sit under a mix
    scale = MAJOR if mode == "major" else MINOR

    # harmony: bass movement over 4 bars (semitones above the key) -> chord degrees
    moves = prof.get("bass_movement") or {}
    move = _weighted(rng, moves) or "0,0,0,0"
    offsets = [int(x) for x in str(move).split(",")][:4] or [0, 0, 0, 0]
    if all(o == 0 for o in offsets) and rng.random() < 0.5:
        other = _weighted(rng, {k: v for k, v in moves.items() if k != move and not all(int(x) == 0 for x in k.split(","))})
        if other:
            move = other
            offsets = [int(x) for x in str(other).split(",")][:4]
    progression = semitones_to_degrees(offsets, scale)
    if len(set(progression)) == 1:
        progression = (0, 5, 2, 6) if mode == "minor" else (0, 3, 4, 5)   # the pads move even when the bass stays home
        notes.append("static bass - pads use a stock progression")

    # patterns: kick and clap per track, bass and hats per section
    pats = prof.get("patterns") or {}
    kick_pat = _pattern(rng, pats.get("kick"), FOUR_ON_THE_FLOOR, 2)
    clap_pat = _pattern(rng, pats.get("snare"), BACKBEAT, 1, allow_empty=True)
    patterns = {"kick": kick_pat, "clap": clap_pat}

    # length
    if minutes is None:
        dur_hist = {int(k): v for k, v in (prof.get("duration", {}).get("hist_minutes") or {}).items()}
        m = _weighted(rng, dur_hist)
        minutes = float(m) + 0.5 if m is not None else 6.0
    minutes = max(3.0, min(10.0, float(minutes)))
    target_bars = int(round(minutes * 60.0 / (240.0 / bpm) / 4.0)) * 4

    # structure
    shape = _walk_structure(rng, prof.get("sections") or {}, target_bars)
    sections: list[Section] = []
    for i, (kind, bars) in enumerate(shape):
        nxt = shape[i + 1][0] if i + 1 < len(shape) else None
        sections.append(_section(kind, bars, nxt, patterns, rng, pats))

    # flavor from tempo + seed
    if flavor and flavor in FLAVORS:
        fl = FLAVORS[flavor]
    elif bpm >= 144:
        fl = FLAVORS["psy"] if rng.random() < 0.7 else FLAVORS["tech"]
    elif bpm <= 131:
        fl = FLAVORS["progressive"]
    else:
        fl = FLAVORS["uplifting"] if rng.random() < 0.65 else FLAVORS["tech"]

    return Plan(
        seed=seed, bpm=bpm, root_midi=root_midi, mode=mode, progression=progression,
        arp_shape=ARPS[int(rng.integers(0, len(ARPS)))],
        lead_rhythm=LEAD_RHYTHMS[int(rng.integers(0, len(LEAD_RHYTHMS)))],
        lead_contour=CONTOURS[int(rng.integers(0, len(CONTOURS)))],
        sections=sections, flavor=fl, target_minutes=minutes,
        profile_tracks=int(prof.get("n_tracks", 0)), bass_movement=str(move), notes=notes,
    )


def scale_note(root_midi: int, scale: tuple[int, ...], degree: int) -> int:
    octave, index = divmod(degree, len(scale))
    return root_midi + scale[index] + 12 * octave


def triad(root_midi: int, scale: tuple[int, ...], degree: int) -> list[int]:
    return [root_midi + scale[(degree + step) % 7] + 12 * ((degree + step) // 7) for step in (0, 2, 4)]


__all__ = ["DEFAULT_PROFILE", "FLAVORS", "Flavor", "Plan", "Section", "build_plan", "hits", "scale_note",
           "semitones_to_degrees", "triad"]
