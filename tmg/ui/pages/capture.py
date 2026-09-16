"""Phrases tab: Record / Stop on the system audio, the phrase bank and the trim editor."""

from __future__ import annotations

import os
import shutil

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gio, GLib, Gtk  # noqa: E402

from tmg import db as dbmod  # noqa: E402
from tmg import log, paths  # noqa: E402
from tmg.capture import phrases, pipewire  # noqa: E402
from tmg.ui.widgets.waveform import WaveformView  # noqa: E402

_log = log.get("ui.phrases")

LIST_LIMIT = 500   # rows on screen at once; the counts line says how many more there are

STATUS_LABEL = {
    "recording": "recording...",
    "raw": "raw",
    "processing": "cleaning up...",
    "ready": "ready",
    "failed": "failed",
}


def fmt_duration(seconds) -> str:
    if seconds is None:
        return "-"
    seconds = float(seconds)
    if seconds >= 60:
        m, s = divmod(seconds, 60)
        return f"{int(m)}:{s:04.1f}"
    return f"{seconds:.1f} s"


def esc(text: str) -> str:
    return GLib.markup_escape_text(str(text))


class PhraseRow(Adw.ActionRow):
    def __init__(self, phrase: dict, on_play, on_edit, on_delete) -> None:
        super().__init__(title=esc(phrase["name"]), activatable=True)
        self.phrase = phrase
        self.set_subtitle(esc(self._subtitle()))
        busy = phrase["status"] in ("recording", "processing")
        editable = phrase["status"] in ("ready", "raw")
        self.spinner = Adw.Spinner()
        self.spinner.set_visible(busy)
        self.play_btn = Gtk.Button(icon_name="media-playback-start-symbolic", valign=Gtk.Align.CENTER, tooltip_text="Play")
        self.play_btn.add_css_class("flat")
        self.play_btn.set_sensitive(editable)
        self.play_btn.connect("clicked", lambda *_: on_play(self))
        self.edit_btn = Gtk.Button(icon_name="edit-cut-symbolic", valign=Gtk.Align.CENTER, tooltip_text="Edit: cut the start and the end")
        self.edit_btn.add_css_class("flat")
        self.edit_btn.set_sensitive(editable)
        self.edit_btn.connect("clicked", lambda *_: on_edit(self))
        self.del_btn = Gtk.Button(icon_name="user-trash-symbolic", valign=Gtk.Align.CENTER, tooltip_text="Delete")
        self.del_btn.add_css_class("flat")
        self.del_btn.set_sensitive(not busy)
        self.del_btn.connect("clicked", lambda *_: on_delete(self))
        self.add_suffix(self.spinner)
        self.add_suffix(self.play_btn)
        self.add_suffix(self.edit_btn)
        self.add_suffix(self.del_btn)

    def _subtitle(self) -> str:
        p = self.phrase
        parts = [fmt_duration(p.get("duration_s")), dbmod.to_local(p.get("created_utc")), STATUS_LABEL.get(p["status"], p["status"])]
        if p.get("source") == "library":
            parts.append("from the library")
        if p.get("peak_db") is not None and p["status"] == "ready":
            parts.append(f"peak {p['peak_db']:.1f} dB")
        if p["status"] in ("ready", "raw"):
            parts.append("click to edit")
        return "  ·  ".join(x for x in parts if x)

    def set_playing(self, playing: bool) -> None:
        self.play_btn.set_icon_name("media-playback-stop-symbolic" if playing else "media-playback-start-symbolic")
        self.play_btn.set_tooltip_text("Stop" if playing else "Play")


class CapturePage(Gtk.Box):
    def __init__(self, app) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        # The whole page scrolls, so the editor below the list is never cut off on a small window.
        self.content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=14)
        for side in ("top", "bottom", "start", "end"):
            getattr(self.content, f"set_margin_{side}")(18)
        self.page_scroller = Gtk.ScrolledWindow(vexpand=True, hexpand=True)
        self.page_scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self.page_scroller.set_child(self.content)
        # MINIMUM: the page scrolls only when its minimum height does not fit; otherwise the content is given the
        # whole viewport, so the phrase list (vexpand) grows with the window instead of stopping at a fixed height
        self.page_scroller.get_child().set_vscroll_policy(Gtk.ScrollablePolicy.MINIMUM)
        self.append(self.page_scroller)
        self.app = app
        self.settings = app.settings
        self.db = app.db
        self.recorder = pipewire.Recorder()
        self.player = phrases.Player()
        self._tick_id: int | None = None
        self._rec_id: str | None = None
        self._selected: dict | None = None
        self._playing_row: PhraseRow | None = None
        self._reloading = False
        self._auto_open: str | None = None   # phrase id to open in the editor as soon as it is ready
        self._syncing = False                # guards waveform <-> Start/End fields feedback

        self._build_recorder()
        self._build_list()
        self._build_editor()
        app.add_job_listener(self._on_job_event)
        self.reload()
        self.refresh_source()

    # ---- building ----------------------------------------------------------------
    def _build_recorder(self) -> None:
        group = Adw.PreferencesGroup(
            title="Capture from system audio",
            description="Press Record, play the video or movie that contains the phrase, then press Stop. "
                        "Silence is trimmed and the level normalised automatically (see Settings).",
        )
        self.source_row = Adw.ActionRow(title="Source", subtitle="-")
        refresh = Gtk.Button(icon_name="view-refresh-symbolic", valign=Gtk.Align.CENTER, tooltip_text="Refresh source")
        refresh.add_css_class("flat")
        refresh.connect("clicked", self.refresh_source)
        self.source_row.add_suffix(refresh)
        self.name_row = Adw.EntryRow(title="Phrase name (optional)")
        group.add(self.source_row)
        group.add(self.name_row)
        self.content.append(group)

        controls = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=28, halign=Gtk.Align.CENTER)
        self.rec_button = Gtk.Button(label="●  Record", tooltip_text="Start recording the system audio")
        for cls in ("rec-button", "pill", "destructive-action"):
            self.rec_button.add_css_class(cls)
        self.rec_button.connect("clicked", self._toggle)
        self.elapsed = Gtk.Label(label="00:00.0", width_chars=8)
        self.elapsed.add_css_class("elapsed")
        meter = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4, valign=Gtk.Align.CENTER)
        self.level = Gtk.LevelBar()
        self.level.set_min_value(0.0)
        self.level.set_max_value(1.0)
        self.level.set_size_request(240, 12)
        self.level.add_offset_value(Gtk.LEVEL_BAR_OFFSET_LOW, 0.55)
        self.level.add_offset_value(Gtk.LEVEL_BAR_OFFSET_HIGH, 0.85)
        self.level.add_offset_value(Gtk.LEVEL_BAR_OFFSET_FULL, 0.98)
        self.level_label = Gtk.Label(label="-inf dB")
        self.level_label.add_css_class("dim-label")
        self.level_label.add_css_class("mono")
        meter.append(self.level)
        meter.append(self.level_label)
        controls.append(self.rec_button)
        controls.append(self.elapsed)
        controls.append(meter)
        self.content.append(controls)

    def _build_list(self) -> None:
        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        title = Gtk.Label(label="Phrase bank", xalign=0)
        title.add_css_class("heading")
        self.search = Gtk.SearchEntry(placeholder_text="Search phrases...", hexpand=True)
        self.search.connect("search-changed", lambda *_: self.reload())
        self.source_model = Gtk.StringList.new(["All", "Captured", "From library"])
        self.source_filter = Gtk.DropDown(model=self.source_model)
        self.source_filter.connect("notify::selected", lambda *_: None if self._reloading else self.reload())
        open_btn = Gtk.Button(icon_name="folder-open-symbolic", tooltip_text="Open the phrase folder")
        open_btn.add_css_class("flat")
        open_btn.connect("clicked", self._open_folder)
        header.append(title)
        header.append(self.search)
        header.append(self.source_filter)
        header.append(open_btn)
        self.content.append(header)
        self.count_label = Gtk.Label(xalign=0, wrap=True)
        self.count_label.add_css_class("dim-label")
        self.count_label.add_css_class("caption")
        self.content.append(self.count_label)

        self.listbox = Gtk.ListBox()
        self.listbox.add_css_class("boxed-list")
        self.listbox.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self.listbox.connect("row-selected", self._on_row_selected)
        scrolled = Gtk.ScrolledWindow(min_content_height=160, vexpand=True)
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_child(self.listbox)
        self.empty = Adw.StatusPage(
            icon_name="audio-input-microphone-symbolic",
            title="No phrases yet",
            description="Record the first one with the button above.",
        )
        self.empty.add_css_class("compact")
        self.list_stack = Gtk.Stack(vexpand=True)
        self.list_stack.add_named(self.empty, "empty")
        self.list_stack.add_named(scrolled, "list")
        self.content.append(self.list_stack)

    def _build_editor(self) -> None:
        self.editor = Gtk.Revealer(transition_type=Gtk.RevealerTransitionType.SLIDE_UP)
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        box.add_css_class("card")
        box.add_css_class("card-pad")
        head = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self.editor_title = Gtk.Label(xalign=0, hexpand=True)
        self.editor_title.add_css_class("heading")
        hint = Gtk.Label(label="Drag the yellow handles or type Start / End, listen to the edges, then Cut.", xalign=1)
        hint.add_css_class("dim-label")
        head.append(self.editor_title)
        head.append(hint)
        self.waveform = WaveformView()
        self.waveform.connect("selection-changed", self._on_selection_changed)

        fields = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)

        def spin(label: str, tooltip: str) -> Gtk.SpinButton:
            lbl = Gtk.Label(label=label)
            lbl.add_css_class("dim-label")
            sb = Gtk.SpinButton.new_with_range(0.0, 1.0, 0.05)
            sb.set_digits(2)
            sb.set_width_chars(8)
            sb.set_tooltip_text(tooltip)
            fields.append(lbl)
            fields.append(sb)
            return sb

        self.start_spin = spin("Start (s)", "Where the phrase begins - everything before it is cut")
        self.end_spin = spin("End (s)", "Where the phrase ends - everything after it is cut")
        self.start_spin.connect("value-changed", self._on_spin_changed)
        self.end_spin.connect("value-changed", self._on_spin_changed)
        self.sel_label = Gtk.Label(xalign=1, hexpand=True, label="")
        self.sel_label.add_css_class("dim-label")
        self.sel_label.add_css_class("mono")
        fields.append(self.sel_label)

        buttons = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)

        def button(label: str, icon: str, cb, tooltip: str = "", style: str | None = None) -> Gtk.Button:
            b = Gtk.Button(tooltip_text=tooltip or None)
            b.set_child(Adw.ButtonContent(icon_name=icon, label=label))
            b.connect("clicked", cb)
            if style:
                b.add_css_class(style)
            buttons.append(b)
            return b

        button("Play selection", "media-playback-start-symbolic", self._play_selection, "Play only the highlighted part")
        button("Start edge", "media-skip-backward-symbolic", self._play_start_edge, "Play the first 1.5 s of the selection - check where it begins")
        button("End edge", "media-skip-forward-symbolic", self._play_end_edge, "Play the last 1.5 s of the selection - check where it ends")
        button("Stop", "media-playback-stop-symbolic", self._stop_playback)
        button("Cut to selection", "edit-cut-symbolic", self._trim, "Keep only the highlighted part, cut the rest", "suggested-action")
        button("Restore original", "edit-undo-symbolic", self._restore, "Bring back the untouched recording")
        button("Rename", "document-edit-symbolic", self._rename)
        box.append(head)
        box.append(self.waveform)
        box.append(fields)
        box.append(buttons)
        self.editor.set_child(box)
        self.content.append(self.editor)

    # ---- source -------------------------------------------------------------------
    def refresh_source(self, *_) -> None:
        setting = self.settings.get("capture.sink", "default")
        name = pipewire.resolve_sink(setting)
        if not name:
            self.source_row.set_subtitle("no audio output found - is PipeWire running?")
            return
        prefix = "System default: " if setting == "default" else ""
        self.source_row.set_subtitle(esc(prefix + pipewire.describe_sink(name)))

    # ---- recording ----------------------------------------------------------------
    def _toggle(self, *_) -> None:
        if self.recorder.running:
            self._stop()
        else:
            self._start()

    def _start(self) -> None:
        sink = pipewire.resolve_sink(self.settings.get("capture.sink", "default"))
        if not sink:
            self.app.toast("No audio output found - is PipeWire running?")
            return
        pid = phrases.new_phrase_id()
        default_name = f"Phrase {pid[6:8]}/{pid[4:6]} {pid[9:11]}:{pid[11:13]}:{pid[13:15]}"
        name = phrases.safe_name(self.name_row.get_text(), default_name)
        path = str(paths.PHRASES / f"{pid}.wav")
        try:
            self.recorder.start(
                path, sink, int(self.settings.get("capture.rate", 48000)), int(self.settings.get("capture.channels", 2))
            )
        except RuntimeError as exc:
            _log.error("could not start recording: %s", exc)
            self.app.toast(f"Could not start recording: {exc}", 6)
            return
        self.db.add_phrase(pid, name, path, status="recording")
        self._rec_id = pid
        self.rec_button.set_label("■  Stop")
        self.rec_button.set_tooltip_text("Stop and save the phrase")
        self.name_row.set_sensitive(False)
        self.source_row.set_sensitive(False)
        self._tick_id = GLib.timeout_add(100, self._on_tick)
        self.reload()

    def _on_tick(self) -> bool:
        if not self.recorder.running:
            self._tick_id = None
            _log.warning("pw-record stopped on its own")
            self._stop()
            return False
        e = self.recorder.elapsed
        self.elapsed.set_label(f"{int(e // 60):02d}:{e % 60:04.1f}")
        _rms, peak = self.recorder.level()
        self.level.set_value(max(0.0, min(1.0, (peak + 60.0) / 60.0)))
        self.level_label.set_label(f"{peak:6.1f} dB" if peak > -99 else "-inf dB")
        return True

    def _reset_controls(self) -> None:
        self.rec_button.set_label("●  Record")
        self.rec_button.set_tooltip_text("Start recording the system audio")
        self.elapsed.set_label("00:00.0")
        self.level.set_value(0.0)
        self.level_label.set_label("-inf dB")
        self.name_row.set_sensitive(True)
        self.source_row.set_sensitive(True)

    def _stop(self) -> None:
        if self._tick_id is not None:
            GLib.source_remove(self._tick_id)
            self._tick_id = None
        pid, self._rec_id = self._rec_id, None
        path = self.recorder.path
        try:
            path = self.recorder.stop()
        except RuntimeError as exc:
            _log.warning("stop: %s", exc)
        self._reset_controls()
        phrase = self.db.get_phrase(pid) if pid else None
        if not phrase or not path:
            return
        try:
            info = phrases.wav_info(path)
        except Exception as exc:  # noqa: BLE001
            _log.error("recording unreadable: %s", exc)
            self.db.update_phrase(pid, status="failed")
            self.app.toast("Recording failed: the WAV file is not readable", 6)
            self.reload()
            return
        if info["frames"] == 0:
            self.db.update_phrase(pid, status="failed")
            self.app.toast("Nothing was recorded - was anything playing on that output?", 6)
            self.reload()
            return
        self.db.update_phrase(pid, duration_s=info["duration_s"], sample_rate=info["rate"], channels=info["channels"], status="raw")
        auto_trim = bool(self.settings.get("capture.auto_trim", True))
        normalize = bool(self.settings.get("capture.normalize", True))
        if auto_trim or normalize:
            raw_path = str(paths.PHRASES / f"{pid}.raw.wav")
            self.app.jobs.submit(
                "phrase_postprocess",
                {
                    "phrase_id": pid, "path": path, "raw_path": raw_path,
                    "trim": auto_trim,
                    "threshold_db": float(self.settings.get("capture.trim_threshold_db", -45.0)),
                    "pad_ms": float(self.settings.get("capture.trim_pad_ms", 120)),
                    "normalize": normalize,
                    "peak_db": float(self.settings.get("capture.normalize_peak_db", -1.0)),
                },
            )
            self.db.update_phrase(pid, status="processing", raw_path=raw_path)
            self._auto_open = pid
        else:
            self.db.update_phrase(pid, status="ready")
        self.app.toast(f"Recorded '{phrase['name']}' ({fmt_duration(info['duration_s'])})")
        self.name_row.set_text("")
        self.reload()
        if not (auto_trim or normalize):
            self.open_editor(pid)

    def _on_job_event(self, job_id, kind, params, event) -> None:
        if kind in ("library_analyze", "library_phrases") and event.get("event") in ("track_done", "finished") \
                and (event.get("phrases") or event.get("event") == "finished"):
            self.reload()   # phrases from the library land in the bank as each track is done
            return
        if kind != "phrase_postprocess" or event.get("event") != "finished":
            return
        pid = params.get("phrase_id")
        phrase = self.db.get_phrase(pid) if pid else None
        if not phrase:
            return
        if event.get("status") == "done":
            r = event.get("result") or {}
            self.db.update_phrase(pid, duration_s=r.get("duration_s"), peak_db=r.get("peak_db"), raw_path=r.get("raw_path"), status="ready")
            self.app.toast(f"'{phrase['name']}' is ready ({fmt_duration(r.get('duration_s'))})")
        else:
            self.db.update_phrase(pid, status="ready")  # the raw recording is still usable
            self.app.toast(f"Clean-up failed for '{phrase['name']}' - keeping the raw recording", 6)
        self.reload()
        if self._auto_open == pid:
            self._auto_open = None
            self.open_editor(pid)
        elif self._selected and self._selected["id"] == pid:
            self._select_phrase(self.db.get_phrase(pid))

    # ---- the list -------------------------------------------------------------------
    def _update_counts(self, source: str | None, search: str | None, shown: int) -> None:
        """Counts in the filter labels ('All (2093)') and a line saying how much of the bank is on screen."""
        n_cap, n_lib = self.db.count_phrases("capture"), self.db.count_phrases("library")
        labels = [f"All ({n_cap + n_lib})", f"Captured ({n_cap})", f"From library ({n_lib})"]
        if [self.source_model.get_string(i) for i in range(3)] != labels:
            selected = self.source_filter.get_selected()
            self.source_model.splice(0, 3, labels)
            self.source_filter.set_selected(selected)
        total = self.db.count_phrases(source, search=search)
        n = n_cap + n_lib
        text = f"{n} phrase{'s' if n != 1 else ''} in the bank: {n_cap} captured from the system audio, {n_lib} from the library."
        if search:
            text += f"  {total} match '{search}'."
        if total > shown:
            text += f"  Showing the newest {shown} of {total} - search by track name or phrase name to narrow."
        self.count_label.set_label(text)

    def reload(self) -> None:
        sel_id = self._selected["id"] if self._selected else None
        self._reloading = True
        try:
            while (row := self.listbox.get_row_at_index(0)) is not None:
                self.listbox.remove(row)
            source = {0: None, 1: "capture", 2: "library"}.get(self.source_filter.get_selected())
            search = self.search.get_text().strip() or None
            rows = self.db.list_phrases(source=source, search=search, limit=LIST_LIMIT)
            self._update_counts(source, search, len(rows))
            for p in rows:
                row = PhraseRow(p, self._on_play_row, self._on_edit_row, self._on_delete_row)
                self.listbox.append(row)
                if p["id"] == sel_id:
                    self.listbox.select_row(row)
            self.list_stack.set_visible_child_name("list" if rows else "empty")
        finally:
            self._reloading = False
        if sel_id and not any(p["id"] == sel_id for p in rows):
            self._hide_editor()

    def _hide_editor(self) -> None:
        self._selected = None
        self.editor.set_reveal_child(False)
        self.editor_title.set_label("")
        self.sel_label.set_label("")
        self.waveform.clear()
        self._syncing = True
        self.start_spin.set_value(0.0)
        self.end_spin.set_value(0.0)
        self._syncing = False

    def _on_row_selected(self, listbox, row) -> None:
        if self._reloading:
            return
        if row is None:
            self._hide_editor()
            return
        self._select_phrase(row.phrase)

    def _select_phrase(self, phrase: dict | None) -> None:
        if not phrase or phrase["status"] in ("recording", "processing", "failed"):
            self._hide_editor()
            return
        self._selected = phrase
        self.editor_title.set_label(phrase["name"])
        self.waveform.load(phrase["path"])
        self.editor.set_reveal_child(True)

    def open_editor(self, pid: str) -> None:
        """Select the phrase in the list (which opens the editor) and scroll the editor into view."""
        i = 0
        while (row := self.listbox.get_row_at_index(i)) is not None:
            if row.phrase["id"] == pid:
                self.listbox.select_row(row)
                if self._reloading:
                    self._select_phrase(row.phrase)
                break
            i += 1
        GLib.timeout_add(450, self._scroll_to_editor)

    def _scroll_to_editor(self) -> bool:
        adj = self.page_scroller.get_vadjustment()
        adj.set_value(adj.get_upper() - adj.get_page_size())
        return False

    def _on_edit_row(self, row: PhraseRow) -> None:
        self.open_editor(row.phrase["id"])

    def _on_selection_changed(self, view, start: float, end: float) -> None:
        self.sel_label.set_label(f"keeps {end - start:.2f} s of {view.duration:.2f} s")
        if self._syncing:
            return
        self._syncing = True
        try:
            for sb in (self.start_spin, self.end_spin):
                sb.set_range(0.0, max(view.duration, 0.01))
            self.start_spin.set_value(start)
            self.end_spin.set_value(end)
        finally:
            self._syncing = False

    def _on_spin_changed(self, *_) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            start, end = self.start_spin.get_value(), self.end_spin.get_value()
            if end <= start + 0.01:
                end = start + 0.01
                self.end_spin.set_value(end)
            self.waveform.set_selection(start, end)
        finally:
            self._syncing = False

    def _play_start_edge(self, *_) -> None:
        self._play_range(self.waveform.sel_start, min(self.waveform.sel_end, self.waveform.sel_start + 1.5))

    def _play_end_edge(self, *_) -> None:
        self._play_range(max(self.waveform.sel_start, self.waveform.sel_end - 1.5), self.waveform.sel_end)

    def _play_range(self, start: float, end: float) -> None:
        if not self._selected or end - start < 0.02:
            return
        try:
            tmp = phrases.slice_to_temp(self._selected["path"], start, end)
        except Exception as exc:  # noqa: BLE001
            self.app.toast(f"Could not play: {exc}", 6)
            return
        self._set_playing_row(None)
        self._play_file(tmp, temp=True)

    # ---- playback -------------------------------------------------------------------
    def _on_play_row(self, row: PhraseRow) -> None:
        if self._playing_row is row and self.player.playing:
            self._stop_playback()
            return
        self._set_playing_row(row)
        self._play_file(row.phrase["path"])

    def _play_file(self, path: str, temp: bool = False) -> None:
        self.player.play(path, on_finished=lambda: GLib.idle_add(self._on_playback_finished, path if temp else None))

    def _on_playback_finished(self, tmp_path: str | None) -> bool:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        self._set_playing_row(None)
        return False

    def _set_playing_row(self, row: PhraseRow | None) -> None:
        if self._playing_row is not None:
            self._playing_row.set_playing(False)
        self._playing_row = row
        if row is not None:
            row.set_playing(True)

    def _play_selection(self, *_) -> None:
        self._play_range(self.waveform.sel_start, self.waveform.sel_end)

    def _stop_playback(self, *_) -> None:
        self.player.stop()
        self._set_playing_row(None)

    # ---- editing ---------------------------------------------------------------------
    def _trim(self, *_) -> None:
        p = self._selected
        if not p:
            return
        start, end = self.waveform.sel_start, self.waveform.sel_end
        if end - start < 0.05:
            self.app.toast("The selection is too short")
            return
        if start <= 0.001 and end >= self.waveform.duration - 0.001:
            self.app.toast("The selection is the whole phrase - move the handles or type Start / End first")
            return
        self.player.stop()
        try:
            info = phrases.trim_wav(p["path"], p["path"], start, end)
        except Exception as exc:  # noqa: BLE001
            self.app.toast(f"Trim failed: {exc}", 6)
            return
        self.db.update_phrase(p["id"], duration_s=info["duration_s"])
        _log.info("trimmed %s to %.2f-%.2f s", p["id"], start, end)
        self.app.toast(f"Cut to {fmt_duration(info['duration_s'])} - cut again to refine, or Restore original")
        self.reload()
        self._select_phrase(self.db.get_phrase(p["id"]))

    def _restore(self, *_) -> None:
        p = self._selected
        if not p:
            return
        raw = p.get("raw_path")
        if not raw or not os.path.exists(raw):
            self.app.toast("There is no original to restore")
            return
        self.player.stop()
        shutil.copy2(raw, p["path"])
        info = phrases.wav_info(p["path"])
        self.db.update_phrase(p["id"], duration_s=info["duration_s"], peak_db=None)
        self.app.toast("Original restored")
        self.reload()
        self._select_phrase(self.db.get_phrase(p["id"]))

    def _rename(self, *_) -> None:
        p = self._selected
        if not p:
            return
        dialog = Adw.AlertDialog(heading="Rename phrase")
        entry = Gtk.Entry(text=p["name"], activates_default=True)
        dialog.set_extra_child(entry)
        dialog.add_response("cancel", "Cancel")
        dialog.add_response("rename", "Rename")
        dialog.set_default_response("rename")
        dialog.set_response_appearance("rename", Adw.ResponseAppearance.SUGGESTED)
        dialog.connect("response", self._on_rename_response, p["id"], entry)
        dialog.present(self.get_root())

    def _on_rename_response(self, dialog, response: str, pid: str, entry: Gtk.Entry) -> None:
        if response != "rename":
            return
        name = phrases.safe_name(entry.get_text())
        self.db.update_phrase(pid, name=name)
        self.editor_title.set_label(name)
        self.reload()

    def _on_delete_row(self, row: PhraseRow) -> None:
        p = row.phrase
        dialog = Adw.AlertDialog(
            heading=f"Delete '{p['name']}'?",
            body="The files are moved to data/trash inside the project folder - nothing is destroyed.",
        )
        dialog.add_response("cancel", "Cancel")
        dialog.add_response("delete", "Delete")
        dialog.set_response_appearance("delete", Adw.ResponseAppearance.DESTRUCTIVE)
        dialog.connect("response", self._on_delete_response, p)
        dialog.present(self.get_root())

    def _on_delete_response(self, dialog, response: str, p: dict) -> None:
        if response != "delete":
            return
        if self._selected and self._selected["id"] == p["id"]:
            self.player.stop()
            self._hide_editor()
        phrases.move_to_trash(p["path"], p.get("raw_path") or "")
        self.db.delete_phrase(p["id"])
        _log.info("deleted phrase %s (moved to trash)", p["id"])
        self.app.toast(f"'{p['name']}' moved to trash")
        self.reload()

    def _open_folder(self, *_) -> None:
        Gtk.FileLauncher.new(Gio.File.new_for_path(str(paths.PHRASES))).launch(self.get_root(), None, None)

    def shutdown(self) -> None:
        if self.recorder.running:
            self._stop()
        self.player.stop()
