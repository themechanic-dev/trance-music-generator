"""Application entry point. Runs on the system Python 3.14 - the only interpreter here with GTK bindings.

Heavy work never happens in this process: it goes through the job queue to workers in the .venv.
"""

from __future__ import annotations

import os
import sys

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gdk, Gio, GLib, Gtk  # noqa: E402

from tmg import APP_ID, APP_NAME, __version__, config, log, paths  # noqa: E402
from tmg import db as dbmod  # noqa: E402
from tmg.jobs.queue import JobQueue  # noqa: E402

CSS = """
.rec-button { font-size: 18px; font-weight: bold; padding: 10px 28px; }
.elapsed { font-family: monospace; font-size: 30px; font-weight: bold; }
.mono { font-family: monospace; }
.card-pad { padding: 14px; }
.signature-mark { letter-spacing: 3px; font-weight: bold; opacity: 0.75; }
.signature-line { opacity: 0.55; }
"""


class TmgApplication(Adw.Application):
    def __init__(self) -> None:
        # GTK apps are single-instance by app id: a second launch just raises the first window. For testing a
        # second instance next to a running one, set TMG_NON_UNIQUE=1 (both share the same database and folder).
        flags = Gio.ApplicationFlags.NON_UNIQUE if os.environ.get("TMG_NON_UNIQUE") else Gio.ApplicationFlags.DEFAULT_FLAGS
        super().__init__(application_id=APP_ID, flags=flags)
        self.settings: config.Settings | None = None
        self.db: dbmod.Database | None = None
        self.jobs: JobQueue | None = None
        self.window = None
        self.logger = None
        self._job_listeners: list = []

    # ---- lifecycle ---------------------------------------------------------------
    def do_startup(self) -> None:
        Adw.Application.do_startup(self)
        paths.ensure_dirs()
        self.logger = log.setup()
        self.logger.info("%s %s starting - project folder %s", APP_NAME, __version__, paths.ROOT)
        self.settings = config.Settings()
        self.db = dbmod.Database()
        leftover = self.db.reset_unfinished_jobs()
        if leftover:
            self.logger.warning("%d job(s) left over from a previous run were marked as failed", leftover)
        self.jobs = JobQueue(self.db, on_event=self._on_job_event_from_thread)
        self.jobs.start()

        provider = Gtk.CssProvider()
        provider.load_from_string(CSS)
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(), provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )
        self._add_action("about", self._on_about)
        self._add_action("quit", lambda *_: self.quit(), ["<Control>q"])

    def _add_action(self, name: str, callback, accels: list[str] | None = None) -> None:
        action = Gio.SimpleAction.new(name, None)
        action.connect("activate", callback)
        self.add_action(action)
        if accels:
            self.set_accels_for_action(f"app.{name}", accels)

    def do_activate(self) -> None:
        from tmg.ui.window import MainWindow

        if self.window is None:
            self.window = MainWindow(self)
        self.window.present()

    def do_shutdown(self) -> None:
        if self.jobs:
            self.jobs.shutdown()
        if self.db:
            self.db.close()
        if self.logger:
            self.logger.info("shutdown")
        Adw.Application.do_shutdown(self)

    # ---- job events: worker thread -> main loop ----------------------------------
    def add_job_listener(self, callback) -> None:
        self._job_listeners.append(callback)

    def _on_job_event_from_thread(self, job_id, kind, params, event) -> None:
        GLib.idle_add(self._dispatch_job_event, job_id, kind, params, event)

    def _dispatch_job_event(self, job_id, kind, params, event) -> bool:
        for cb in list(self._job_listeners):
            try:
                cb(job_id, kind, params, event)
            except Exception:  # noqa: BLE001
                self.logger.exception("job listener failed")
        return False

    # ---- helpers -------------------------------------------------------------------
    def toast(self, text: str, timeout: int = 3) -> None:
        if self.window is not None:
            self.window.toast(text, timeout)

    def _on_about(self, *_) -> None:
        dialog = Adw.AboutDialog(
            application_name=APP_NAME,
            version=__version__,
            application_icon="audio-x-generic",
            comments="Learns trance from your own library, composes new tracks, grabs phrases from the "
                     "system audio and renders MP3 or MP4 with GPU visuals. Everything runs locally.",
        )
        dialog.present(self.window)


def main(argv: list[str] | None = None) -> int:
    app = TmgApplication()
    return app.run(sys.argv if argv is None else argv)


if __name__ == "__main__":
    raise SystemExit(main())
