"""Library tab: import a track / a CD / a collection, watch the analysis, see what was learned."""

from __future__ import annotations

import os
import threading
import time

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gdk, Gio, GLib, Gtk  # noqa: E402

from tmg import log  # noqa: E402
from tmg.library import scan  # noqa: E402
from tmg.ui.widgets.learned import LearnedView  # noqa: E402

_log = log.get("ui.library")

STATUS_LABEL = {"queued": "queued", "analyzing": "analysing...", "done": "analysed", "failed": "failed"}
MAX_ROWS = 300


def esc(text) -> str:
    return GLib.markup_escape_text(str(text))


def fmt_dur(seconds) -> str:
    if not seconds:
        return "-"
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


class TrackRow(Adw.ActionRow):
    def __init__(self, track: dict, on_stems, on_reanalyze, on_remove) -> None:
        super().__init__(title=esc(track.get("title") or os.path.basename(track["path"])))
        self.track = track
        parts = [track.get("album") or "", fmt_dur(track.get("duration_s"))]
        if track.get("bpm"):
            parts.append(f"{track['bpm']:.1f} BPM")
        if track.get("key_name"):
            parts.append(track["key_name"])
        if track.get("n_bars"):
            parts.append(f"{track['n_bars']} bars")
        status = STATUS_LABEL.get(track["status"], track["status"])
        if track["status"] == "failed" and track.get("error"):
            status += f": {track['error'][:80]}"
        parts.append(status)
        self.set_subtitle(esc("  ·  ".join(p for p in parts if p)))
        if track.get("stems_dir") and os.path.isdir(track["stems_dir"]):
            b = Gtk.Button(icon_name="folder-open-symbolic", valign=Gtk.Align.CENTER, tooltip_text="Open the separated stems (drums, bass, other, vocals)")
            b.add_css_class("flat")
            b.connect("clicked", lambda *_: on_stems(self))
            self.add_suffix(b)
        b = Gtk.Button(icon_name="view-refresh-symbolic", valign=Gtk.Align.CENTER, tooltip_text="Analyse again")
        b.add_css_class("flat")
        b.set_sensitive(track["status"] in ("done", "failed"))
        b.connect("clicked", lambda *_: on_reanalyze(self))
        self.add_suffix(b)
        b = Gtk.Button(icon_name="user-trash-symbolic", valign=Gtk.Align.CENTER, tooltip_text="Remove from the library (the file is not touched)")
        b.add_css_class("flat")
        b.connect("clicked", lambda *_: on_remove(self))
        self.add_suffix(b)


class LibraryPage(Gtk.Box):
    def __init__(self, app) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.app = app
        self.settings = app.settings
        self.db = app.db
        self._analyze_job: int | None = None
        self._last_reload = 0.0
        self._reload_pending = False
        self._profile_building = False
        self._last_profile = 0.0
        self._tracks_since_profile = 0

        self.content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=14)
        for side in ("top", "bottom", "start", "end"):
            getattr(self.content, f"set_margin_{side}")(18)
        scroller = Gtk.ScrolledWindow(vexpand=True, hexpand=True)
        scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scroller.set_child(self.content)
        self.append(scroller)

        self._build_import()
        self._build_progress()
        self._build_learned()
        self._build_tracks()

        drop = Gtk.DropTarget.new(Gdk.FileList, Gdk.DragAction.COPY)
        drop.connect("drop", self._on_drop)
        self.add_controller(drop)

        app.add_job_listener(self._on_job_event)
        self.reload()

    # ---- import -------------------------------------------------------------------
    def _build_import(self) -> None:
        group = Adw.PreferencesGroup(
            title="Learn from your library",
            description="Choose what to import - the choice is always yours: one track, one CD (a folder with tracks), "
                        "or a whole collection (a folder with many CD folders, read recursively). You can also drop files "
                        "or folders anywhere on this page. Every track is analysed: tempo, beat grid, kick/snare/hat/bass "
                        "patterns, structure (intro, build, drop, breakdown, outro), key and bass movement.",
        )
        self.content.append(group)
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12, homogeneous=True)

        def big(label: str, sub: str, icon: str, cb) -> None:
            b = Gtk.Button()
            inner = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4, margin_top=8, margin_bottom=8)
            inner.append(Gtk.Image.new_from_icon_name(icon))
            inner.get_last_child().set_pixel_size(32)
            t = Gtk.Label(label=label)
            t.add_css_class("heading")
            s = Gtk.Label(label=sub, wrap=True, justify=Gtk.Justification.CENTER)
            s.add_css_class("dim-label")
            inner.append(t)
            inner.append(s)
            b.set_child(inner)
            b.connect("clicked", cb)
            row.append(b)

        big("Import a track", "one audio file", "audio-x-generic-symbolic", self._pick_track)
        big("Import a CD", "one folder, only its tracks", "media-optical-symbolic", lambda *_: self._pick_folder("cd"))
        big("Import a collection", "a folder of CD folders, all of them", "folder-symbolic", lambda *_: self._pick_folder("collection"))
        self.content.append(row)

    def _file_filter(self) -> Gio.ListStore:
        f = Gtk.FileFilter()
        f.set_name("Audio files")
        for ext in sorted(scan.AUDIO_EXTENSIONS):
            f.add_suffix(ext[1:])
        store = Gio.ListStore.new(Gtk.FileFilter)
        store.append(f)
        return store

    def _initial_folder(self, dialog: Gtk.FileDialog) -> None:
        last = self.settings.get("library.last_folder") or ""
        if last and os.path.isdir(last):
            dialog.set_initial_folder(Gio.File.new_for_path(last))

    def _pick_track(self, *_) -> None:
        d = Gtk.FileDialog(title="Import a track")
        d.set_filters(self._file_filter())
        self._initial_folder(d)
        d.open(self.get_root(), None, self._on_track_chosen)

    def _on_track_chosen(self, dialog, result) -> None:
        try:
            f = dialog.open_finish(result)
        except GLib.Error:
            return
        if f is not None and f.get_path():
            self._import("track", f.get_path())

    def _pick_folder(self, kind: str) -> None:
        d = Gtk.FileDialog(title="Import a CD (one folder)" if kind == "cd" else "Import a collection (folder of folders)")
        self._initial_folder(d)
        d.select_folder(self.get_root(), None, lambda dlg, res: self._on_folder_chosen(dlg, res, kind))

    def _on_folder_chosen(self, dialog, result, kind: str) -> None:
        try:
            f = dialog.select_folder_finish(result)
        except GLib.Error:
            return
        if f is not None and f.get_path():
            self._import(kind, f.get_path())

    def _on_drop(self, target, value, x, y) -> bool:
        files = value.get_files() if hasattr(value, "get_files") else []
        for f in files:
            path = f.get_path()
            if path:
                self._import(scan.guess_kind_for_drop(path), path)
        return bool(files)

    def _import(self, kind: str, path: str) -> None:
        folder = path if os.path.isdir(path) else os.path.dirname(path)
        self.settings.set("library.last_folder", os.path.dirname(folder) if kind == "cd" else folder)
        self.app.jobs.submit("library_import", {"kind": kind, "root": path})
        _log.info("import %s: %s", kind, path)
        self.app.toast(f"Importing {kind}: {os.path.basename(path.rstrip('/')) or path} - scanning...")

    # ---- progress ------------------------------------------------------------------
    def _build_progress(self) -> None:
        card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        card.add_css_class("card")
        card.add_css_class("card-pad")
        self.counts_label = Gtk.Label(xalign=0)
        self.counts_label.add_css_class("heading")
        self.progress_label = Gtk.Label(xalign=0, label="Nothing running.", ellipsize=3)
        self.progress_label.add_css_class("dim-label")
        self.progress_bar = Gtk.ProgressBar()
        buttons = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)

        def button(label, icon, cb, tooltip="", style=None):
            b = Gtk.Button(tooltip_text=tooltip or None)
            b.set_child(Adw.ButtonContent(icon_name=icon, label=label))
            b.connect("clicked", cb)
            if style:
                b.add_css_class(style)
            buttons.append(b)
            return b

        self.analyze_btn = button("Analyse queued tracks", "media-playback-start-symbolic", self._start_analysis,
                                  "Analyse every queued track (resumes where it stopped)", "suggested-action")
        self.pause_btn = button("Pause", "media-playback-pause-symbolic", self._pause_analysis,
                                "Stop after the current track - the rest stays queued")
        button("Retry failed", "view-refresh-symbolic", self._retry_failed, "Put failed tracks back in the queue")
        button("Rebuild profile", "emblem-synchronizing-symbolic", self._rebuild_profile,
               "Recompute 'what it learned' from the analysed tracks")
        button("Re-extract phrases", "audio-input-microphone-symbolic", self._reextract_phrases,
               "Run the current phrase rule (speech score) over every analysed track and replace the library phrases")
        card.append(self.counts_label)
        card.append(self.progress_label)
        card.append(self.progress_bar)
        card.append(buttons)
        self.content.append(card)

    def _start_analysis(self, *_) -> None:
        if self._analyze_job is not None:
            self.app.toast("The analysis is already running")
            return
        counts = self.db.track_counts()
        if counts["queued"] + counts["analyzing"] == 0:
            self.app.toast("No queued tracks - import something first")
            return
        s = self.settings
        self._analyze_job = self.app.jobs.submit("library_analyze", {
            "demucs": bool(s.get("library.demucs", True)),
            "keep_stems_first": int(s.get("library.keep_stems_first", 3)),
            "extract_vocals": bool(s.get("library.extract_vocals", True)),
            "vocal_max_count": int(s.get("library.vocal_max_count", 3)),
            "vocal_threshold_db": float(s.get("library.vocal_threshold_db", -35.0)),
        })
        self.progress_label.set_label("starting the analysis worker...")
        self._update_buttons()

    def _pause_analysis(self, *_) -> None:
        if self._analyze_job is None:
            return
        self.app.jobs.cancel(self._analyze_job)
        self.app.toast("Pausing after the current track")

    def _retry_failed(self, *_) -> None:
        n = self.db.requeue_failed()
        self.app.toast(f"{n} failed track(s) queued again")
        self.reload()

    def _reextract_phrases(self, *_) -> None:
        if self._analyze_job is not None or self.app.jobs.busy:
            self.app.toast("A job is running - wait for it to finish (or pause the analysis) first")
            return
        counts = self.db.track_counts()
        n = counts.get("done", 0)
        if not n:
            self.app.toast("No analysed tracks yet")
            return
        s = self.settings
        dialog = Adw.AlertDialog(
            heading="Re-extract the library phrases?",
            body=f"Runs Demucs again over the {n} analysed tracks (about 15 s each, roughly {n * 15 / 60:.0f} min on this GPU) and "
                 "picks the phrases with the speech rule: gaps between syllables, consonants and vowels, energy in the voice "
                 "band, no beat-locked pulsing. The current library phrases are replaced; their files go to data/trash. "
                 "Tracks whose phrases already come from the current settings (minimum speech score, threshold, count) are "
                 "skipped, so after a change only the work that changed is done.",
        )
        dialog.add_response("cancel", "Cancel")
        dialog.add_response("go", "Re-extract")
        dialog.set_response_appearance("go", Adw.ResponseAppearance.SUGGESTED)
        dialog.set_default_response("go")

        def on_response(_d, response):
            if response != "go":
                return
            self._phrases_job = self.app.jobs.submit("library_phrases", {
                "vocal_threshold_db": float(s.get("library.vocal_threshold_db", -40.0)),
                "vocal_max_count": int(s.get("library.vocal_max_count", 3)),
                "min_score": float(s.get("library.vocal_min_score", 0.5)),
            })
            self.progress_label.set_label("re-extracting the phrases...")
            self.app.toast("Phrase re-extraction started - see the Jobs tab")

        dialog.connect("response", on_response)
        dialog.present(self.get_root())

    def _rebuild_profile(self, *_) -> None:
        self.app.toast("Rebuilding the profile...")
        self.build_profile_now(announce=True)

    def build_profile_now(self, announce: bool = False) -> None:
        """Aggregate the analyses into the profile right here, in a thread - the queue may be busy analysing."""
        if self._profile_building:
            return
        self._profile_building = True
        self._tracks_since_profile = 0

        def work() -> None:
            try:
                from tmg.library import profile as profile_mod

                prof = profile_mod.build_profile(self.db.analyses())
                self.db.save_profile(prof)
                GLib.idle_add(self._profile_done, prof.get("n_tracks", 0), announce, None)
            except Exception as exc:  # noqa: BLE001 - e.g. no numpy on this interpreter: fall back to the worker
                GLib.idle_add(self._profile_done, 0, announce, exc)

        threading.Thread(target=work, daemon=True).start()

    def _profile_done(self, n: int, announce: bool, error) -> bool:
        self._profile_building = False
        self._last_profile = time.monotonic()
        if error is not None:
            _log.warning("profile in-process failed (%s) - using the worker", error)
            self.app.jobs.submit("library_profile", {})
            return False
        self.learned.set_profile(self.db.get_profile())
        if announce:
            self.app.toast(f"Profile rebuilt from {n} analysed tracks")
        return False

    def _update_buttons(self) -> None:
        running = self._analyze_job is not None
        self.analyze_btn.set_sensitive(not running)
        self.pause_btn.set_sensitive(running)
        if not running:
            self.progress_bar.set_fraction(0.0)

    # ---- learned -------------------------------------------------------------------
    def _build_learned(self) -> None:
        head = Gtk.Label(label="What it learned", xalign=0)
        head.add_css_class("heading")
        self.content.append(head)
        self.learned = LearnedView()
        self.content.append(self.learned)

    # ---- tracks --------------------------------------------------------------------
    def _build_tracks(self) -> None:
        head = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        title = Gtk.Label(label="Tracks", xalign=0)
        title.add_css_class("heading")
        self.search = Gtk.SearchEntry(placeholder_text="Search title, album, artist...", hexpand=True)
        self.search.connect("search-changed", lambda *_: self.reload_tracks())
        self.status_filter = Gtk.DropDown.new_from_strings(["All", "Analysed", "Queued", "Failed"])
        self.status_filter.connect("notify::selected", lambda *_: self.reload_tracks())
        self.tracks_count = Gtk.Label(xalign=1)
        self.tracks_count.add_css_class("dim-label")
        head.append(title)
        head.append(self.search)
        head.append(self.status_filter)
        head.append(self.tracks_count)
        self.content.append(head)
        self.listbox = Gtk.ListBox()
        self.listbox.add_css_class("boxed-list")
        self.listbox.set_selection_mode(Gtk.SelectionMode.NONE)
        self.content.append(self.listbox)

    def reload_tracks(self) -> None:
        while (row := self.listbox.get_row_at_index(0)) is not None:
            self.listbox.remove(row)
        status = {0: None, 1: "done", 2: "queued", 3: "failed"}.get(self.status_filter.get_selected())
        search = self.search.get_text().strip() or None
        rows = self.db.list_tracks(status=status, search=search, limit=MAX_ROWS)
        for t in rows:
            self.listbox.append(TrackRow(t, self._open_stems, self._reanalyze, self._remove_track))
        total = self.db.track_counts()["total"]
        self.tracks_count.set_label(f"showing {len(rows)} of {total}" if len(rows) >= MAX_ROWS else f"{len(rows)} shown · {total} in the library")

    def reload(self) -> None:
        c = self.db.track_counts()
        self.counts_label.set_label(
            f"{c['total']} tracks in the library  ·  {c['done']} analysed  ·  {c['queued'] + c['analyzing']} waiting  ·  {c['failed']} failed"
        )
        self.learned.set_profile(self.db.get_profile())
        self.reload_tracks()
        self._update_buttons()
        self._last_reload = time.monotonic()

    def _reload_soon(self) -> None:
        if self._reload_pending:
            return
        self._reload_pending = True
        delay = max(0, int((2.0 - (time.monotonic() - self._last_reload)) * 1000))
        GLib.timeout_add(delay, self._reload_now)

    def _reload_now(self) -> bool:
        self._reload_pending = False
        self.reload()
        return False

    def _open_stems(self, row: TrackRow) -> None:
        Gtk.FileLauncher.new(Gio.File.new_for_path(row.track["stems_dir"])).launch(self.get_root(), None, None)

    def _reanalyze(self, row: TrackRow) -> None:
        self.db.update_track(row.track["id"], status="queued", error=None)
        self.app.toast("Queued for analysis again")
        self.reload()

    def _remove_track(self, row: TrackRow) -> None:
        self.db.delete_track(row.track["id"])
        self.app.toast("Removed from the library (the file was not touched)")
        self.reload()

    # ---- job events ------------------------------------------------------------------
    def _on_job_event(self, job_id, kind, params, event) -> None:
        if not kind.startswith("library_"):
            return
        name = event.get("event")
        if kind == "library_analyze":
            if name == "started":
                self._analyze_job = job_id
                self._update_buttons()
            elif name == "progress":
                self.progress_bar.set_fraction(float(event.get("fraction", 0.0)))
                self.progress_label.set_label(f"Analysing {event.get('message', '')}")
            elif name == "track_done":
                self._tracks_since_profile += 1
                # keep 'What it learned' alive while the analysis runs: every 5 tracks, at most every 45 s
                if self._tracks_since_profile >= 5 and time.monotonic() - self._last_profile > 45:
                    self.build_profile_now()
                elif self._tracks_since_profile == 1 and not self.db.get_profile():
                    self.build_profile_now()
                self._reload_soon()
            elif name == "finished":
                self._analyze_job = None
                r = event.get("result") or {}
                if event.get("status") == "done":
                    self.progress_label.set_label(f"Done: {r.get('analyzed', 0)} analysed, {r.get('failed', 0)} failed, {r.get('minutes', 0)} min.")
                    self.app.toast(f"Analysis finished: {r.get('analyzed', 0)} tracks")
                else:
                    self.progress_label.set_label(f"Analysis {event.get('status')}: {(event.get('message') or '').splitlines()[0]}")
                self.reload()
        elif kind == "library_import" and name == "finished":
            r = event.get("result") or {}
            if event.get("status") == "done":
                self.app.toast(f"Imported {r.get('added', 0)} new tracks ({r.get('skipped', 0)} already known)")
                if bool(self.settings.get("library.auto_analyze", True)):
                    self._start_analysis()
            else:
                self.app.toast(f"Import failed: {(event.get('message') or '').splitlines()[0]}", 6)
            self.reload()
        elif kind == "library_profile" and name == "finished":
            self.reload()
        elif kind == "library_phrases":
            if name == "progress":
                self.progress_label.set_label(f"Phrases: {event.get('message', '')}")
                self.progress_bar.set_fraction(float(event.get("fraction", 0.0)))
            elif name == "finished":
                r = event.get("result") or {}
                if event.get("status") == "done":
                    self.progress_label.set_label(f"Phrases re-extracted: {r.get('phrases', 0)} from {r.get('tracks', 0)} tracks, "
                                                  f"{r.get('failed', 0)} failed, {r.get('minutes', 0)} min.")
                    self.app.toast(f"{r.get('phrases', 0)} library phrases picked with the speech rule")
                else:
                    self.progress_label.set_label(f"Phrase re-extraction {event.get('status')}: {(event.get('message') or '').splitlines()[0]}")
                self.reload()
