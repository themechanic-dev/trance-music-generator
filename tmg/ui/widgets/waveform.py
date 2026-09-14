"""Waveform with two handles (selection start/end) for trimming a phrase. Gtk.DrawingArea + cairo."""

from __future__ import annotations

import threading

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, GObject, Gtk  # noqa: E402

from tmg.capture import phrases  # noqa: E402

HANDLE_GRAB_PX = 14
RULER_H = 18
_TICK_STEPS = (0.1, 0.2, 0.5, 1, 2, 5, 10, 30, 60)


class WaveformView(Gtk.DrawingArea):
    __gsignals__ = {
        "selection-changed": (GObject.SignalFlags.RUN_FIRST, None, (float, float)),
    }

    def __init__(self) -> None:
        super().__init__()
        self.set_content_height(230)
        self.set_hexpand(True)
        self.set_draw_func(self._draw)
        self.bins: list[tuple[float, float]] = []
        self.duration = 0.0
        self.sel_start = 0.0
        self.sel_end = 0.0
        self.loading = False
        self._drag_handle: str | None = None
        self._drag_x0 = 0.0
        self._load_token = 0

        drag = Gtk.GestureDrag()
        drag.connect("drag-begin", self._on_drag_begin)
        drag.connect("drag-update", self._on_drag_update)
        drag.connect("drag-end", self._on_drag_end)
        self.add_controller(drag)

    # ---- data -------------------------------------------------------------------
    def clear(self) -> None:
        self.bins, self.duration, self.sel_start, self.sel_end = [], 0.0, 0.0, 0.0
        self.queue_draw()

    def load(self, path: str) -> None:
        self._load_token += 1
        token = self._load_token
        self.loading = True
        self.queue_draw()

        def work() -> None:
            try:
                info = phrases.wav_info(path)
                bins = phrases.waveform_bins(path, 1400)
            except Exception:  # noqa: BLE001
                info, bins = {"duration_s": 0.0}, []
            GLib.idle_add(self._loaded, token, info["duration_s"], bins)

        threading.Thread(target=work, daemon=True).start()

    def _loaded(self, token: int, duration: float, bins: list) -> bool:
        if token != self._load_token:
            return False
        self.loading = False
        self.bins, self.duration = bins, duration
        self.sel_start, self.sel_end = 0.0, duration
        self.queue_draw()
        self.emit("selection-changed", self.sel_start, self.sel_end)
        return False

    def set_selection(self, start: float, end: float) -> None:
        start = max(0.0, min(self.duration, start))
        end = max(start, min(self.duration, end))
        self.sel_start, self.sel_end = start, end
        self.queue_draw()
        self.emit("selection-changed", start, end)

    # ---- interaction ------------------------------------------------------------
    def _x_to_t(self, x: float) -> float:
        w = max(1, self.get_width())
        return max(0.0, min(self.duration, x / w * self.duration))

    def _t_to_x(self, t: float) -> float:
        return (t / self.duration * self.get_width()) if self.duration else 0.0

    def _on_drag_begin(self, gesture, x, y) -> None:
        if not self.duration:
            return
        xs, xe = self._t_to_x(self.sel_start), self._t_to_x(self.sel_end)
        if abs(x - xs) <= HANDLE_GRAB_PX and abs(x - xs) <= abs(x - xe):
            self._drag_handle = "start"
        elif abs(x - xe) <= HANDLE_GRAB_PX:
            self._drag_handle = "end"
        else:
            # click elsewhere: the nearest handle jumps there
            self._drag_handle = "start" if abs(x - xs) < abs(x - xe) else "end"
            self._apply(self._drag_handle, self._x_to_t(x))
        self._drag_x0 = x

    def _on_drag_update(self, gesture, dx, dy) -> None:
        if self._drag_handle:
            self._apply(self._drag_handle, self._x_to_t(self._drag_x0 + dx))

    def _on_drag_end(self, gesture, dx, dy) -> None:
        self._drag_handle = None

    def _apply(self, handle: str, t: float) -> None:
        if handle == "start":
            self.sel_start = min(t, self.sel_end - 0.01)
        else:
            self.sel_end = max(t, self.sel_start + 0.01)
        self.queue_draw()
        self.emit("selection-changed", self.sel_start, self.sel_end)

    # ---- drawing ----------------------------------------------------------------
    def _draw(self, area, cr, w, h) -> None:
        cr.set_source_rgba(0.09, 0.09, 0.11, 1.0)
        cr.rectangle(0, 0, w, h)
        cr.fill()
        cr.set_font_size(13)
        if self.loading:
            cr.set_source_rgba(0.7, 0.7, 0.75, 1.0)
            cr.move_to(12, h / 2)
            cr.show_text("Loading waveform...")
            return
        if not self.bins or not self.duration:
            cr.set_source_rgba(0.5, 0.5, 0.55, 1.0)
            cr.move_to(12, h / 2)
            cr.show_text("Select a phrase from the list to edit it")
            return
        wave_h = h - RULER_H
        mid = wave_h / 2
        n = len(self.bins)
        xs, xe = self._t_to_x(self.sel_start), self._t_to_x(self.sel_end)
        bar_w = max(1.0, w / n)
        # outside the selection: dim
        cr.set_source_rgba(0.35, 0.36, 0.42, 1.0)
        for i, (lo, hi) in enumerate(self.bins):
            cr.rectangle(i / n * w, mid - hi * mid * 0.92, bar_w, (hi - lo) * mid * 0.92)
        cr.fill()
        # inside the selection: bright
        cr.save()
        cr.rectangle(xs, 0, xe - xs, wave_h)
        cr.clip()
        cr.set_source_rgba(0.98, 0.45, 0.20, 1.0)
        for i, (lo, hi) in enumerate(self.bins):
            cr.rectangle(i / n * w, mid - hi * mid * 0.92, bar_w, (hi - lo) * mid * 0.92)
        cr.fill()
        cr.restore()
        # time ruler
        cr.set_source_rgba(0.16, 0.16, 0.19, 1.0)
        cr.rectangle(0, wave_h, w, RULER_H)
        cr.fill()
        step = next((st for st in _TICK_STEPS if w / (self.duration / st) >= 70), _TICK_STEPS[-1])
        cr.set_font_size(10)
        t = 0.0
        while t <= self.duration + 1e-9:
            x = self._t_to_x(t)
            cr.set_source_rgba(0.6, 0.6, 0.66, 1.0)
            cr.rectangle(x, wave_h, 1, 5)
            cr.fill()
            label = f"{t:.1f}s" if step < 1 else f"{t:.0f}s"
            cr.move_to(x + 3, h - 4)
            cr.show_text(label)
            t += step
        # handles (full height so they are easy to grab)
        for x in (xs, xe):
            cr.set_source_rgba(1.0, 0.85, 0.3, 1.0)
            cr.rectangle(x - 1.5, 0, 3, wave_h)
            cr.fill()
            cr.rectangle(x - 8, 0, 16, 14)
            cr.fill()
        # times next to the handles
        cr.set_font_size(11)
        cr.set_source_rgba(1.0, 0.92, 0.6, 1.0)
        cr.move_to(min(xs + 10, w - 60), 26)
        cr.show_text(f"{self.sel_start:.2f}s")
        cr.move_to(max(xe - 58, 4), 26)
        cr.show_text(f"{self.sel_end:.2f}s")
