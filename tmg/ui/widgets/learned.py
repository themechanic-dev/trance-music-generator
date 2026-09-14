"""'What it learned': the style profile drawn so the user can see it - tempo histogram, structures, patterns."""

from __future__ import annotations

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk  # noqa: E402

SECTION_ORDER = ("intro", "build", "drop", "breakdown", "outro")


class TempoHistogram(Gtk.DrawingArea):
    def __init__(self) -> None:
        super().__init__()
        self.set_content_height(150)
        self.set_hexpand(True)
        self.hist: dict[int, int] = {}
        self.median = 0.0
        self.set_draw_func(self._draw)

    def set_data(self, hist: dict, median: float) -> None:
        self.hist = {int(k): int(v) for k, v in (hist or {}).items()}
        self.median = median
        self.queue_draw()

    def _draw(self, area, cr, w, h) -> None:
        cr.set_source_rgba(0.09, 0.09, 0.11, 1)
        cr.rectangle(0, 0, w, h)
        cr.fill()
        cr.set_font_size(11)
        if not self.hist:
            cr.set_source_rgba(0.5, 0.5, 0.55, 1)
            cr.move_to(12, h / 2)
            cr.show_text("Tempo histogram appears after the first analysed tracks")
            return
        lo, hi = min(self.hist), max(self.hist) + 2
        span = max(hi - lo, 2)
        top = max(self.hist.values())
        pad_l, pad_b, pad_t = 8, 22, 12
        for bpm, n in self.hist.items():
            x0 = pad_l + (bpm - lo) / span * (w - 2 * pad_l)
            bw = 2 / span * (w - 2 * pad_l)
            bh = (h - pad_b - pad_t) * n / top
            cr.set_source_rgba(0.98, 0.45, 0.20, 1)
            cr.rectangle(x0 + 1, h - pad_b - bh, max(1, bw - 2), bh)
            cr.fill()
            cr.set_source_rgba(0.95, 0.95, 0.95, 1)
            cr.move_to(x0 + 2, h - pad_b - bh - 3)
            cr.show_text(str(n))
        cr.set_source_rgba(0.6, 0.6, 0.66, 1)
        step = 2 if span <= 24 else 4 if span <= 60 else 10
        b = lo
        while b <= hi:
            x = pad_l + (b - lo) / span * (w - 2 * pad_l)
            cr.rectangle(x, h - pad_b, 1, 4)
            cr.fill()
            cr.move_to(x - 8, h - 6)
            cr.show_text(str(b))
            b += step
        if self.median:
            x = pad_l + (self.median - lo) / span * (w - 2 * pad_l)
            cr.set_source_rgba(1.0, 0.85, 0.3, 1)
            cr.rectangle(x - 1, pad_t - 6, 2, h - pad_b - pad_t + 6)
            cr.fill()
            cr.move_to(x + 4, pad_t + 4)
            cr.show_text(f"median {self.median:.1f} BPM")


class PatternGrid(Gtk.DrawingArea):
    """Rows of 16 cells (one bar in 16ths) for the most common patterns of one instrument."""

    def __init__(self, rows: int = 4) -> None:
        super().__init__()
        self.rows = rows
        self.set_content_height(18 * rows + 8)
        self.set_hexpand(True)
        self.patterns: list[tuple[str, float]] = []
        self.set_draw_func(self._draw)

    def set_data(self, patterns: dict) -> None:
        self.patterns = [(p, float(s)) for p, s in list((patterns or {}).items())[: self.rows]]
        self.queue_draw()

    def _draw(self, area, cr, w, h) -> None:
        cr.set_font_size(11)
        if not self.patterns:
            cr.set_source_rgba(0.5, 0.5, 0.55, 1)
            cr.move_to(4, 14)
            cr.show_text("-")
            return
        label_w = 62
        cell = min(22.0, (w - label_w - 8) / 16.0)
        for r, (pat, share) in enumerate(self.patterns):
            y = 4 + r * 18
            cr.set_source_rgba(0.85, 0.85, 0.9, 1)
            cr.move_to(2, y + 12)
            cr.show_text(f"{share * 100:4.0f} %")
            for k, ch in enumerate(pat[:16]):
                x = label_w + k * cell
                if ch == "1":
                    cr.set_source_rgba(0.98, 0.45, 0.20, 1)
                else:
                    cr.set_source_rgba(0.2, 0.2, 0.24, 1)
                cr.rectangle(x + 1, y + 1, cell - 2, 14)
                cr.fill()
                if k % 4 == 0:
                    cr.set_source_rgba(0.6, 0.6, 0.66, 0.9)
                    cr.rectangle(x, y, 1, 16)
                    cr.fill()


class LearnedView(Gtk.Box):
    def __init__(self) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.add_css_class("card")
        self.add_css_class("card-pad")
        self.summary = Gtk.Label(xalign=0, wrap=True)
        self.append(self.summary)
        self.tempo = TempoHistogram()
        self.append(self._titled("Tempo (BPM) - how many tracks at each tempo", self.tempo))
        self.structures = Gtk.Label(xalign=0, wrap=True, selectable=True)
        self.structures.add_css_class("mono")
        self.append(self._titled("Structures - the most common section sequences and typical lengths (bars)", self.structures))
        grids = Gtk.Grid(column_spacing=18, row_spacing=10, column_homogeneous=True)
        self.grid_kick, self.grid_snare, self.grid_hat, self.grid_bass = (PatternGrid() for _ in range(4))
        cells = (("Kick", self.grid_kick), ("Snare / clap", self.grid_snare), ("Hats", self.grid_hat), ("Bass", self.grid_bass))
        for i, (name, g) in enumerate(cells):
            grids.attach(self._titled(f"{name} - most common bar patterns (one bar in 16ths)", g), i % 2, i // 2, 1, 1)
        self.append(grids)
        self.extra = Gtk.Label(xalign=0, wrap=True, selectable=True)
        self.append(self.extra)

    @staticmethod
    def _titled(title: str, widget: Gtk.Widget) -> Gtk.Box:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        lbl = Gtk.Label(label=title, xalign=0, wrap=True)
        lbl.add_css_class("dim-label")
        box.append(lbl)
        box.append(widget)
        return box

    def set_profile(self, profile: dict | None) -> None:
        if not profile or not profile.get("n_tracks"):
            self.summary.set_label("Nothing learned yet. Import a track, a CD or a collection - the profile appears after the first analysed tracks.")
            self.tempo.set_data({}, 0.0)
            self.structures.set_label("-")
            for g in (self.grid_kick, self.grid_snare, self.grid_hat, self.grid_bass):
                g.set_data({})
            self.extra.set_label("")
            return
        t = profile.get("tempo", {})
        d = profile.get("duration", {})
        self.summary.set_label(
            f"{profile['n_tracks']} tracks analysed ({profile.get('tracks_with_demucs', 0)} with stem separation, "
            f"{profile.get('tracks_with_vocals', 0)} with vocals found). Tempo {t.get('min')}-{t.get('max')} BPM, "
            f"median {t.get('median')}. Typical length {d.get('median_s', 0) / 60:.1f} min ({d.get('median_bars', 0)} bars). "
            f"Profile built {profile.get('built_utc', '')[:16].replace('T', ' ')} UTC."
        )
        self.tempo.set_data(t.get("hist", {}), float(t.get("median") or 0))
        secs = profile.get("sections", {})
        lines = []
        for seq, n in list((secs.get("sequences") or {}).items())[:6]:
            lines.append(f"{n:4d} ×  {seq.replace('>', ' > ')}")
        lengths = secs.get("lengths") or {}
        for kind in SECTION_ORDER:
            if kind in lengths:
                items = sorted(lengths[kind].items(), key=lambda kv: -kv[1])[:4]
                lines.append(f"{kind:9s} bars: " + ", ".join(f"{int(b)} ({n}×)" for b, n in items))
        trans = secs.get("transitions") or {}
        for kind in SECTION_ORDER:
            if kind in trans:
                lines.append(f"after {kind:9s} -> " + ", ".join(f"{k} {v * 100:.0f} %" for k, v in trans[kind].items()))
        if secs.get("drop_minus_breakdown_db") is not None:
            lines.append(f"drops are {secs['drop_minus_breakdown_db']:.1f} dB louder than breakdowns")
        self.structures.set_label("\n".join(lines) or "-")
        pats = profile.get("patterns", {})
        self.grid_kick.set_data(pats.get("kick"))
        self.grid_snare.set_data(pats.get("snare"))
        self.grid_hat.set_data(pats.get("hat"))
        self.grid_bass.set_data(pats.get("bass"))
        keys = ", ".join(f"{k} ({v})" for k, v in list((profile.get("keys") or {}).items())[:8]) or "-"
        moves = ", ".join(f"[{k}] {v * 100:.0f} %" for k, v in list((profile.get("bass_movement") or {}).items())[:6]) or "-"
        self.extra.set_label(f"Keys: {keys}\nBass movement over 4 bars (semitones above the key): {moves}")
