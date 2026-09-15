"""Jobs tab: the queue with progress, and the application log."""

from __future__ import annotations

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gio, GLib, Gtk  # noqa: E402

from tmg import db as dbmod  # noqa: E402
from tmg import log, paths  # noqa: E402

KIND_LABELS = {
    "env_check": "Environment check",
    "phrase_postprocess": "Phrase clean-up (trim + normalise)",
    "echo": "Test job",
    "library_import": "Library import (scan + tags)",
    "library_analyze": "Library analysis (Demucs + beats, patterns, structure)",
    "library_profile": "Rebuild style profile",
    "compose": "Compose a track",
    "render_video": "Render the video (GPU visuals + NVENC)",
    "hf_check": "HuggingFace token and model access check",
    "model_test": "Neural model self-test (load on the GPU + short clip)",
    "library_phrases": "Re-extract the library phrases (Demucs + speech score)",
}


def esc(text) -> str:
    return GLib.markup_escape_text(str(text))


class JobRow(Adw.ActionRow):
    def __init__(self, job: dict, on_cancel) -> None:
        super().__init__(title=esc(f"#{job['id']}  {KIND_LABELS.get(job['kind'], job['kind'])}"))
        self.job = job
        self.progress = Gtk.ProgressBar(valign=Gtk.Align.CENTER)
        self.progress.set_size_request(160, -1)
        self.cancel_btn = Gtk.Button(icon_name="process-stop-symbolic", valign=Gtk.Align.CENTER, tooltip_text="Cancel")
        self.cancel_btn.add_css_class("flat")
        self.cancel_btn.connect("clicked", lambda *_: on_cancel(self))
        self.add_suffix(self.progress)
        self.add_suffix(self.cancel_btn)
        self.update(job)

    def update(self, job: dict) -> None:
        self.job = job
        status = job["status"]
        message = (job.get("message") or "").splitlines()[0] if job.get("message") else ""
        stamp = dbmod.to_local(job.get("finished_utc") or job.get("started_utc") or job.get("created_utc"), "%H:%M:%S")
        mark = {"done": "✓ ", "failed": "✗ ", "cancelled": "⊘ "}.get(status, "")
        self.set_subtitle(esc(f"{mark}{status}  ·  {message}  ·  {stamp}"))
        self.progress.set_fraction(float(job.get("progress") or 0.0))
        self.progress.set_visible(status in ("queued", "running"))
        self.cancel_btn.set_visible(status in ("queued", "running"))


class JobsPage(Gtk.Box):
    def __init__(self, app) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        for side in ("top", "bottom", "start", "end"):
            getattr(self, f"set_margin_{side}")(18)
        self.app = app
        self.db = app.db
        self._rows: dict[int, JobRow] = {}

        head = Gtk.Label(label="Jobs", xalign=0)
        head.add_css_class("heading")
        sub = Gtk.Label(
            label="Heavy work runs one job at a time in a separate worker process, so this window never freezes.",
            xalign=0, wrap=True,
        )
        sub.add_css_class("dim-label")
        self.append(head)
        self.append(sub)

        self.listbox = Gtk.ListBox()
        self.listbox.add_css_class("boxed-list")
        self.listbox.set_selection_mode(Gtk.SelectionMode.NONE)
        jobs_scroller = Gtk.ScrolledWindow(min_content_height=120, max_content_height=300, propagate_natural_height=True)
        jobs_scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        jobs_scroller.set_child(self.listbox)
        self.empty = Gtk.Label(label="No jobs yet.", xalign=0)
        self.empty.add_css_class("dim-label")
        self.jobs_stack = Gtk.Stack()
        self.jobs_stack.add_named(self.empty, "empty")
        self.jobs_stack.add_named(jobs_scroller, "list")
        self.append(self.jobs_stack)

        log_head = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8, margin_top=8)
        log_title = Gtk.Label(label="Log", xalign=0, hexpand=True)
        log_title.add_css_class("heading")
        clear_btn = Gtk.Button(label="Clear view")
        clear_btn.connect("clicked", lambda *_: self.buffer.set_text(""))
        open_btn = Gtk.Button(label="Open log folder")
        open_btn.connect("clicked", self._open_log_folder)
        log_head.append(log_title)
        log_head.append(clear_btn)
        log_head.append(open_btn)
        self.append(log_head)

        self.textview = Gtk.TextView(editable=False, cursor_visible=False, monospace=True, wrap_mode=Gtk.WrapMode.WORD_CHAR)
        self.textview.set_left_margin(8)
        self.textview.set_right_margin(8)
        self.buffer = self.textview.get_buffer()
        self.end_mark = self.buffer.create_mark(None, self.buffer.get_end_iter(), False)
        log_scroller = Gtk.ScrolledWindow(vexpand=True)
        log_scroller.add_css_class("card")
        log_scroller.set_child(self.textview)
        self.append(log_scroller)

        for line in log.recent():
            self._append_line(line)
        log.subscribe(lambda line: GLib.idle_add(self._append_line, line))
        app.add_job_listener(self._on_job_event)
        self.reload()

    def reload(self) -> None:
        while (row := self.listbox.get_row_at_index(0)) is not None:
            self.listbox.remove(row)
        self._rows.clear()
        jobs = self.db.list_jobs(100)
        for job in jobs:
            row = JobRow(job, self._on_cancel)
            self._rows[job["id"]] = row
            self.listbox.append(row)
        self.jobs_stack.set_visible_child_name("list" if jobs else "empty")

    def _on_job_event(self, job_id, kind, params, event) -> None:
        job = self.db.get_job(job_id)
        if not job:
            return
        row = self._rows.get(job_id)
        if row is None:
            self.reload()
        else:
            row.update(job)

    def _on_cancel(self, row: JobRow) -> None:
        self.app.jobs.cancel(row.job["id"])

    def _append_line(self, line: str) -> bool:
        self.buffer.insert(self.buffer.get_end_iter(), line + "\n")
        if self.buffer.get_line_count() > 4000:
            start = self.buffer.get_start_iter()
            cut = self.buffer.get_iter_at_line(500)[1]
            self.buffer.delete(start, cut)
        self.textview.scroll_to_mark(self.end_mark, 0.0, False, 0.0, 1.0)
        return False

    def _open_log_folder(self, *_) -> None:
        Gtk.FileLauncher.new(Gio.File.new_for_path(str(paths.LOGS))).launch(self.get_root(), None, None)
