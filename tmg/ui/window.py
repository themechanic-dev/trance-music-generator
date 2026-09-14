"""Main window: header with the tab switcher, four pages, toasts."""

from __future__ import annotations

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gio, Gtk  # noqa: E402

from tmg import APP_NAME  # noqa: E402
from tmg.ui.pages.capture import CapturePage  # noqa: E402
from tmg.ui.pages.compose import ComposePage  # noqa: E402
from tmg.ui.pages.help import HelpPage  # noqa: E402
from tmg.ui.pages.jobs import JobsPage  # noqa: E402
from tmg.ui.pages.library import LibraryPage  # noqa: E402
from tmg.ui.pages.settings import SettingsPage  # noqa: E402


class MainWindow(Adw.ApplicationWindow):
    def __init__(self, app) -> None:
        super().__init__(application=app, title=APP_NAME)
        self.app = app
        self.set_default_size(
            int(app.settings.get("ui.window_width", 1100)), int(app.settings.get("ui.window_height", 760))
        )

        self.stack = Adw.ViewStack()
        self.capture_page = CapturePage(app)
        self.library_page = LibraryPage(app)
        self.compose_page = ComposePage(app)
        self.jobs_page = JobsPage(app)
        self.settings_page = SettingsPage(app)
        self.help_page = HelpPage(app)
        self.stack.add_titled_with_icon(self.capture_page, "phrases", "Phrases", "audio-input-microphone-symbolic")
        self.stack.add_titled_with_icon(self.library_page, "library", "Library", "folder-music-symbolic")
        self.stack.add_titled_with_icon(self.compose_page, "compose", "Compose", "media-record-symbolic")
        self.stack.add_titled_with_icon(self.jobs_page, "jobs", "Jobs", "view-list-symbolic")
        self.stack.add_titled_with_icon(self.settings_page, "settings", "Settings", "emblem-system-symbolic")
        self.stack.add_titled_with_icon(self.help_page, "help", "Help", "help-browser-symbolic")

        header = Adw.HeaderBar()
        switcher = Adw.ViewSwitcher(stack=self.stack, policy=Adw.ViewSwitcherPolicy.WIDE)
        header.set_title_widget(switcher)
        menu = Gio.Menu()
        menu.append("About", "app.about")
        menu.append("Quit", "app.quit")
        header.pack_end(Gtk.MenuButton(icon_name="open-menu-symbolic", menu_model=menu, tooltip_text="Menu"))

        toolbar = Adw.ToolbarView()
        toolbar.add_top_bar(header)
        toolbar.set_content(self.stack)
        self.toast_overlay = Adw.ToastOverlay()
        self.toast_overlay.set_child(toolbar)
        self.set_content(self.toast_overlay)
        self.connect("close-request", self._on_close_request)

    def toast(self, text: str, timeout: int = 3) -> None:
        self.toast_overlay.add_toast(Adw.Toast(title=text, timeout=timeout))

    def _on_close_request(self, *_) -> bool:
        width, height = self.get_default_size()
        self.app.settings.set("ui.window_width", int(width), save=False)
        self.app.settings.set("ui.window_height", int(height), save=True)
        self.capture_page.shutdown()
        self.compose_page.shutdown()
        return False
