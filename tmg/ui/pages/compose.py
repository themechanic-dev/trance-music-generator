"""Compose tab: draw new tracks from the profile, listen, keep or delete."""

from __future__ import annotations

import os
import subprocess
import threading

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gio, GLib, Gtk  # noqa: E402

from tmg import db as dbmod  # noqa: E402
from tmg import log, paths  # noqa: E402
from tmg.capture import phrases  # noqa: E402
from tmg.music.neural import MODELS as NEURAL_MODELS  # noqa: E402

_log = log.get("ui.compose")

LENGTHS = ["Like the library", "4 min", "5 min", "6 min", "7 min", "8 min", "9 min"]
WHERE = [("Breakdowns and before drops", "both"), ("Breakdowns only", "breakdown"), ("Before drops only", "build")]
SOURCES = [("Random from my captured phrases", "captured"), ("Random from the library phrases", "library"),
           ("Random from all phrases", "all"), ("Only the phrases I select", "selected")]
FORMATS = [("MP3", ("mp3",), False), ("MP3 + MP4 video", ("mp3",), True), ("WAV", ("wav",), False),
           ("WAV + MP4 video", ("wav",), True), ("MP3 + WAV", ("mp3", "wav"), False)]
FLAVORS = ["Auto (from tempo)", "uplifting", "progressive", "psy", "tech"]


def esc(text) -> str:
    return GLib.markup_escape_text(str(text))


def fmt_dur(seconds) -> str:
    if not seconds:
        return "-"
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


class ProductionRow(Adw.ActionRow):
    def __init__(self, prod: dict, on_play, on_folder, on_delete, on_video, on_play_video) -> None:
        super().__init__(title=esc(f"{prod['title']}  ·  seed {prod['seed']}"))
        self.prod = prod
        sub = f"{prod.get('bpm', 0):.1f} BPM  ·  {prod.get('key_name')}  ·  {fmt_dur(prod.get('duration_s'))}  ·  {prod.get('flavor')}  ·  " \
              f"from {prod.get('profile_tracks') or 0} library tracks  ·  {dbmod.to_local(prod.get('created_utc'))}\n{prod.get('structure') or ''}"
        used = prod.get("phrases") or []
        if used:
            sub += "\nphrases: " + "  ·  ".join(f"'{d.get('name')}' {d.get('slot')} at {fmt_dur(d.get('start_s'))}" for d in used)
        nn = prod.get("neural") or {}
        if nn.get("model"):
            sub += f"\nneural: {nn['model']} at {nn.get('level_db')} dB" + (" · dry copy kept for A/B" if nn.get("dry_mp3") else "")
        elif nn.get("error"):
            sub += f"\nneural: failed ({nn['error'][:60]})"
        v = prod.get("video") or {}
        if prod.get("mp4_path"):
            sub += f"\nvideo: {v.get('size', '')} {v.get('fps', '')} fps {v.get('codec', '')} · {v.get('shots', '?')} shots · rendered in {v.get('seconds', '?')} s"
        self.set_subtitle(esc(sub))
        self.set_subtitle_lines(0)
        self.play_btn = Gtk.Button(icon_name="media-playback-start-symbolic", valign=Gtk.Align.CENTER, tooltip_text="Play")
        self.play_btn.add_css_class("flat")
        self.play_btn.connect("clicked", lambda *_: on_play(self))
        b = Gtk.Button(icon_name="folder-open-symbolic", valign=Gtk.Align.CENTER, tooltip_text="Open the output folder")
        b.add_css_class("flat")
        b.connect("clicked", lambda *_: on_folder(self))
        d = Gtk.Button(icon_name="user-trash-symbolic", valign=Gtk.Align.CENTER, tooltip_text="Delete (files go to data/trash)")
        d.add_css_class("flat")
        d.connect("clicked", lambda *_: on_delete(self))
        self.add_suffix(self.play_btn)
        if prod.get("mp4_path") and os.path.exists(prod["mp4_path"]):
            pv = Gtk.Button(icon_name="video-display-symbolic", valign=Gtk.Align.CENTER, tooltip_text="Open the video (MP4)")
            pv.add_css_class("flat")
            pv.connect("clicked", lambda *_: on_play_video(self))
            self.add_suffix(pv)
        mv = Gtk.Button(icon_name="camera-video-symbolic", valign=Gtk.Align.CENTER,
                        tooltip_text="Make the video (again) with the current visual settings")
        mv.add_css_class("flat")
        mv.connect("clicked", lambda *_: on_video(self))
        self.add_suffix(mv)
        self.add_suffix(b)
        self.add_suffix(d)

    def set_playing(self, playing: bool) -> None:
        self.play_btn.set_icon_name("media-playback-stop-symbolic" if playing else "media-playback-start-symbolic")


class ComposePage(Gtk.Box):
    def __init__(self, app) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.app = app
        self.settings = app.settings
        self.db = app.db
        self.player = phrases.Player()
        self._playing_row: ProductionRow | None = None
        self._tmp_play: str | None = None

        self.content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=14)
        for side in ("top", "bottom", "start", "end"):
            getattr(self.content, f"set_margin_{side}")(18)
        scroller = Gtk.ScrolledWindow(vexpand=True, hexpand=True)
        scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scroller.set_child(self.content)
        self.append(scroller)

        self._build_controls()
        self._build_list()
        app.add_job_listener(self._on_job_event)
        self.reload()

    # ---- controls ----------------------------------------------------------------
    def _build_controls(self) -> None:
        group = Adw.PreferencesGroup(
            title="Compose",
            description="Every track is drawn from 'What it learned': tempo from the histogram, structure from the learned "
                        "transitions and lengths, rhythm patterns from the library's bars, key and bass movement from its "
                        "tracks. Same seed = same track. Each track takes about a minute of CPU.",
        )
        self.count_row = Adw.SpinRow.new_with_range(1, 10, 1)
        self.count_row.set_title("How many tracks")
        self.count_row.set_value(int(self.settings.get("compose.count", 5)))
        self.count_row.connect("notify::value", lambda r, _p: self.settings.set("compose.count", int(r.get_value())))
        group.add(self.count_row)
        self.length_row = Adw.ComboRow(title="Length", subtitle="'Like the library' draws the length from the analysed tracks")
        self.length_row.set_model(Gtk.StringList.new(LENGTHS))
        self.length_row.set_selected(int(self.settings.get("compose.length_index", 0)))
        self.length_row.connect("notify::selected", lambda r, _p: self.settings.set("compose.length_index", int(r.get_selected())))
        group.add(self.length_row)
        self.flavor_row = Adw.ComboRow(title="Sound flavor", subtitle="Timbre only (kick punch, lead, pad) - the rhythm and structure always come from the profile")
        self.flavor_row.set_model(Gtk.StringList.new(FLAVORS))
        self.flavor_row.set_selected(int(self.settings.get("compose.flavor_index", 0)))
        self.flavor_row.connect("notify::selected", lambda r, _p: self.settings.set("compose.flavor_index", int(r.get_selected())))
        group.add(self.flavor_row)
        self.format_row = Adw.ComboRow(title="Output")
        self.format_row.set_model(Gtk.StringList.new([f[0] for f in FORMATS]))
        self.format_row.set_selected(int(self.settings.get("compose.format_index", 0)))
        self.format_row.connect("notify::selected", lambda r, _p: self.settings.set("compose.format_index", int(r.get_selected())))
        group.add(self.format_row)
        self.bitrate_row = Adw.SpinRow.new_with_range(128, 320, 32)
        self.bitrate_row.set_title("MP3 bitrate (kbit/s)")
        self.bitrate_row.set_value(int(self.settings.get("compose.bitrate", 192)))
        self.bitrate_row.connect("notify::value", lambda r, _p: self.settings.set("compose.bitrate", int(r.get_value())))
        group.add(self.bitrate_row)
        self.seed_row = Adw.EntryRow(title="Seed (optional - empty = a new random track each time)")
        group.add(self.seed_row)
        self.content.append(group)
        self._build_phrase_group()
        self._build_neural_group()

        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        self.compose_btn = Gtk.Button()
        self.compose_btn.set_child(Adw.ButtonContent(icon_name="media-record-symbolic", label="Compose"))
        for cls in ("suggested-action", "pill"):
            self.compose_btn.add_css_class(cls)
        self.compose_btn.connect("clicked", self._compose)
        self.status_label = Gtk.Label(xalign=0, hexpand=True, ellipsize=3)
        self.status_label.add_css_class("dim-label")
        row.append(self.compose_btn)
        row.append(self.status_label)
        self.content.append(row)
        self.progress = Gtk.ProgressBar()
        self.content.append(self.progress)

    # ---- phrases in the music ----------------------------------------------------
    def _build_phrase_group(self) -> None:
        s = self.settings
        g = Adw.PreferencesGroup(
            title="Phrases in the music",
            description="Spoken bits from the phrase bank, dropped where the structure asks for them: opening a breakdown, "
                        "or landing right before a drop at the end of a build. The music steps back 4 dB while they speak.",
        )
        self.ph_enabled = Adw.SwitchRow(title="Add phrases from the bank")
        self.ph_enabled.set_active(bool(s.get("phrases.enabled", True)))
        self.ph_enabled.connect("notify::active", lambda r, _p: s.set("phrases.enabled", r.get_active()))
        g.add(self.ph_enabled)
        self.ph_count = Adw.SpinRow.new_with_range(1, 4, 1)
        self.ph_count.set_title("Phrases per track")
        self.ph_count.set_value(int(s.get("phrases.count", 2)))
        self.ph_count.connect("notify::value", lambda r, _p: s.set("phrases.count", int(r.get_value())))
        g.add(self.ph_count)
        self.ph_where = Adw.ComboRow(title="Where")
        self.ph_where.set_model(Gtk.StringList.new([w[0] for w in WHERE]))
        self.ph_where.set_selected(int(s.get("phrases.where_index", 0)))
        self.ph_where.connect("notify::selected", lambda r, _p: s.set("phrases.where_index", int(r.get_selected())))
        g.add(self.ph_where)
        self.ph_intro = Adw.SwitchRow(title="Also allow the intro", subtitle="A phrase two bars into the intro")
        self.ph_intro.set_active(bool(s.get("phrases.intro", False)))
        self.ph_intro.connect("notify::active", lambda r, _p: s.set("phrases.intro", r.get_active()))
        g.add(self.ph_intro)
        self.ph_source = Adw.ComboRow(title="Which phrases")
        self._source_model = Gtk.StringList.new(self._source_labels())
        self.ph_source.set_model(self._source_model)
        self.ph_source.set_selected(int(s.get("phrases.source_index", 2)))
        self.ph_source.connect("notify::selected", self._on_source_changed)
        self.connect("map", lambda *_: self._refresh_source_labels())
        pick = Gtk.Button(label="Select...", valign=Gtk.Align.CENTER)
        pick.connect("clicked", self._pick_phrases)
        self.ph_source.add_suffix(pick)
        g.add(self.ph_source)
        self.ph_selected = Adw.ActionRow(title="Selected phrases")
        g.add(self.ph_selected)
        self.ph_level = Adw.SpinRow.new_with_range(-18, 0, 1)
        self.ph_level.set_title("Phrase level (dB)")
        self.ph_level.set_subtitle("0 = as loud as the music peaks; -6 sits on top without shouting")
        self.ph_level.set_value(float(s.get("phrases.level_db", -6.0)))
        self.ph_level.connect("notify::value", lambda r, _p: s.set("phrases.level_db", float(r.get_value())))
        g.add(self.ph_level)
        self.ph_tel = Adw.SwitchRow(title="Telephone / radio EQ", subtitle="Band-limit the voice (300-3400 Hz) for the classic sampled sound")
        self.ph_tel.set_active(bool(s.get("phrases.telephone", False)))
        self.ph_tel.connect("notify::active", lambda r, _p: s.set("phrases.telephone", r.get_active()))
        g.add(self.ph_tel)
        self.ph_echo = Adw.SpinRow.new_with_range(0, 6, 1)
        self.ph_echo.set_title("Echo repeats")
        self.ph_echo.set_subtitle("Dotted-eighth echoes in time with the beat")
        self.ph_echo.set_value(int(s.get("phrases.echo", 2)))
        self.ph_echo.connect("notify::value", lambda r, _p: s.set("phrases.echo", int(r.get_value())))
        g.add(self.ph_echo)
        self.ph_fit = Adw.SwitchRow(title="Fit to the beat", subtitle="Stretch up to 15 % so the phrase ends on a beat")
        self.ph_fit.set_active(bool(s.get("phrases.fit", True)))
        self.ph_fit.connect("notify::active", lambda r, _p: s.set("phrases.fit", r.get_active()))
        g.add(self.ph_fit)
        self.content.append(g)
        self._refresh_selected_label()

    # ---- neural sound ----------------------------------------------------------------
    def _build_neural_group(self) -> None:
        s = self.settings
        g = Adw.PreferencesGroup(
            title="Neural sound (MusicGen / Stable Audio Open)",
            description="Real timbre under the arrangement: two ~30 s clips per track are generated from the plan (tempo, key, "
                        "flavor), stretched to the tempo when close enough, looped bar by bar under every section, high-passed "
                        "so the kick and bass stay ours, ducked by the kick. On this GPU: about 35 s per clip (small), 70 s (medium, "
                        "melody) or 27 s (Stable Audio Open). Settings > Neural model shows whether the chosen model is downloaded and working.",
        )
        self.nn_enabled = Adw.SwitchRow(title="Add neural textures")
        self.nn_enabled.set_active(bool(s.get("neural.enabled", False)))
        self.nn_enabled.connect("notify::active", lambda r, _p: s.set("neural.enabled", r.get_active()))
        g.add(self.nn_enabled)
        keys = list(NEURAL_MODELS)
        self.nn_model = Adw.ComboRow(title="Model")
        self.nn_model.set_model(Gtk.StringList.new([NEURAL_MODELS[k]["label"] for k in keys]))
        self.nn_model.set_selected(keys.index(s.get("neural.model", "stereo-small")) if s.get("neural.model", "stereo-small") in keys else 0)
        self.nn_model.connect("notify::selected", lambda r, _p: s.set("neural.model", keys[r.get_selected()]))
        g.add(self.nn_model)
        self.nn_level = Adw.SpinRow.new_with_range(-24, 0, 1)
        self.nn_level.set_title("Neural level (dB)")
        self.nn_level.set_subtitle("Relative to the mix; -10 sits underneath, -4 competes with the arrangement")
        self.nn_level.set_value(float(s.get("neural.level_db", -10.0)))
        self.nn_level.connect("notify::value", lambda r, _p: s.set("neural.level_db", float(r.get_value())))
        g.add(self.nn_level)
        self.nn_energy = Adw.SwitchRow(title="Energy layer in builds and drops too", subtitle="Off = atmosphere only, in intros, breakdowns and outros")
        self.nn_energy.set_active(bool(s.get("neural.energy", True)))
        self.nn_energy.connect("notify::active", lambda r, _p: s.set("neural.energy", r.get_active()))
        g.add(self.nn_energy)
        self.nn_ab = Adw.SwitchRow(title="Keep a dry copy for A/B", subtitle="Writes '[track]-dry' next to the track: the same music without the neural layer")
        self.nn_ab.set_active(bool(s.get("neural.ab", True)))
        self.nn_ab.connect("notify::active", lambda r, _p: s.set("neural.ab", r.get_active()))
        g.add(self.nn_ab)
        self.content.append(g)

    def _neural_params(self) -> dict:
        s = self.settings
        return {"enabled": bool(s.get("neural.enabled", False)), "model": s.get("neural.model", "stereo-small"),
                "level_db": float(s.get("neural.level_db", -10.0)), "energy": bool(s.get("neural.energy", True)),
                "ab": bool(s.get("neural.ab", True))}

    def _source_labels(self) -> list[str]:
        """The source names with how many phrases each one draws from, so an empty choice is visible."""
        n_cap, n_lib = self.db.count_phrases("capture", "ready"), self.db.count_phrases("library", "ready")
        counts = {"captured": n_cap, "library": n_lib, "all": n_cap + n_lib}
        return [f"{label} ({counts[key]})" if key in counts else label for label, key in SOURCES]

    def _refresh_source_labels(self) -> None:
        labels = self._source_labels()
        if [self._source_model.get_string(i) for i in range(len(SOURCES))] != labels:
            self._labels_refreshing = True
            try:
                selected = self.ph_source.get_selected()
                self._source_model.splice(0, len(SOURCES), labels)
                self.ph_source.set_selected(selected)
            finally:
                self._labels_refreshing = False

    def _on_source_changed(self, row, _p) -> None:
        if getattr(self, "_labels_refreshing", False):
            return
        self.settings.set("phrases.source_index", int(row.get_selected()))
        if SOURCES[row.get_selected()][1] == "selected" and not self.settings.get("phrases.ids"):
            self._pick_phrases()

    def _refresh_selected_label(self) -> None:
        ids = self.settings.get("phrases.ids") or []
        rows = self.db.get_phrases_by_ids([str(i) for i in ids]) if ids else []
        names = [r["name"] for r in rows]
        self.ph_selected.set_subtitle(esc(", ".join(names)) if names else "none - press Select... (used only with 'Only the phrases I select')")

    def _pick_phrases(self, *_) -> None:
        """A dialog with a check box per phrase in the bank."""
        dialog = Adw.Dialog(title="Select phrases", content_width=560, content_height=640)
        toolbar = Adw.ToolbarView()
        header = Adw.HeaderBar()
        toolbar.add_top_bar(header)
        listbox = Gtk.ListBox()
        listbox.add_css_class("boxed-list")
        listbox.set_selection_mode(Gtk.SelectionMode.NONE)
        chosen = {str(i) for i in (self.settings.get("phrases.ids") or [])}
        checks: list[tuple[str, Gtk.CheckButton]] = []
        for p in self.db.list_phrases(limit=1000):
            if p.get("status") != "ready":
                continue
            row = Adw.ActionRow(title=esc(p["name"]), subtitle=esc(f"{fmt_dur(p.get('duration_s'))} · {'from the library' if p.get('source') == 'library' else 'captured'}"))
            cb = Gtk.CheckButton(active=p["id"] in chosen, valign=Gtk.Align.CENTER)
            row.add_prefix(cb)
            row.set_activatable_widget(cb)
            listbox.append(row)
            checks.append((p["id"], cb))
        scroller = Gtk.ScrolledWindow(vexpand=True)
        scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        for side in ("top", "bottom", "start", "end"):
            getattr(box, f"set_margin_{side}")(12)
        box.append(Gtk.Label(label="Tick the phrases that may be used. Untick everything to use none.", xalign=0, wrap=True))
        box.append(listbox)
        scroller.set_child(box)
        toolbar.set_content(scroller)
        done = Gtk.Button(label="Done")
        done.add_css_class("suggested-action")

        def finish(*_) -> None:
            ids = [pid for pid, cb in checks if cb.get_active()]
            self.settings.set("phrases.ids", ids)
            self._refresh_selected_label()
            dialog.close()

        done.connect("clicked", finish)
        header.pack_end(done)
        dialog.set_child(toolbar)
        dialog.present(self.get_root())

    def _phrase_params(self) -> dict:
        s = self.settings
        return {
            "enabled": bool(s.get("phrases.enabled", True)), "count": int(s.get("phrases.count", 2)),
            "where": WHERE[int(s.get("phrases.where_index", 0))][1], "intro": bool(s.get("phrases.intro", False)),
            "source": SOURCES[int(s.get("phrases.source_index", 2))][1], "ids": list(s.get("phrases.ids") or []),
            "level_db": float(s.get("phrases.level_db", -6.0)), "telephone": bool(s.get("phrases.telephone", False)),
            "echo": int(s.get("phrases.echo", 2)), "fit": bool(s.get("phrases.fit", True)),
        }

    def _compose(self, *_) -> None:
        n = int(self.count_row.get_value())
        length_i = self.length_row.get_selected()
        minutes = None if length_i == 0 else float(LENGTHS[length_i].split()[0])
        formats = FORMATS[self.format_row.get_selected()][1]
        want_video = FORMATS[self.format_row.get_selected()][2]
        flavor_i = self.flavor_row.get_selected()
        flavor = None if flavor_i == 0 else FLAVORS[flavor_i]
        seed_text = self.seed_row.get_text().strip()
        seed = int(seed_text) if seed_text.isdigit() else None
        if seed is not None:
            n = 1
        for _ in range(n):
            self.app.jobs.submit("compose", {"seed": seed, "minutes": minutes, "formats": list(formats),
                                             "bitrate": int(self.bitrate_row.get_value()), "flavor": flavor,
                                             "phrases": self._phrase_params(), "neural": self._neural_params(), "video": want_video})
        profile = self.db.get_profile()
        src = f"from {profile['n_tracks']} analysed tracks" if profile and profile.get("n_tracks") else "from the built-in profile (library empty)"
        self.app.toast(f"Composing {n} track(s) {src}")
        self.status_label.set_label(f"{n} track(s) queued - watch the Jobs tab")
        _log.info("compose: %d track(s), minutes=%s, formats=%s, flavor=%s, seed=%s", n, minutes, formats, flavor, seed)

    # ---- list ----------------------------------------------------------------------
    def _build_list(self) -> None:
        head = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        title = Gtk.Label(label="Productions", xalign=0, hexpand=True)
        title.add_css_class("heading")
        open_btn = Gtk.Button(icon_name="folder-open-symbolic", tooltip_text="Open the output folder")
        open_btn.add_css_class("flat")
        open_btn.connect("clicked", lambda *_: Gtk.FileLauncher.new(Gio.File.new_for_path(str(paths.OUTPUT))).launch(self.get_root(), None, None))
        head.append(title)
        head.append(open_btn)
        self.content.append(head)
        self.listbox = Gtk.ListBox()
        self.listbox.add_css_class("boxed-list")
        self.listbox.set_selection_mode(Gtk.SelectionMode.NONE)
        self.empty = Gtk.Label(label="No tracks yet. Press Compose.", xalign=0)
        self.empty.add_css_class("dim-label")
        self.stack = Gtk.Stack()
        self.stack.add_named(self.empty, "empty")
        self.stack.add_named(self.listbox, "list")
        self.content.append(self.stack)

    def reload(self) -> None:
        while (row := self.listbox.get_row_at_index(0)) is not None:
            self.listbox.remove(row)
        prods = self.db.list_productions()
        for p in prods:
            self.listbox.append(ProductionRow(p, self._play, self._folder, self._delete, self._make_video, self._play_video))
        self.stack.set_visible_child_name("list" if prods else "empty")

    # ---- playback (MP3 decoded through our ffmpeg to a temp WAV, then pw-play) ------------
    def _play(self, row: ProductionRow) -> None:
        if self._playing_row is row and self.player.playing:
            self._stop()
            return
        path = row.prod.get("wav_path") or row.prod.get("mp3_path")
        if not path or not os.path.exists(path):
            self.app.toast("The audio file is missing")
            return
        self._stop()
        self._playing_row = row
        row.set_playing(True)
        if path.endswith(".wav"):
            self.player.play(path, on_finished=lambda: GLib.idle_add(self._finished))
            return

        def decode() -> None:
            paths.TMP.mkdir(parents=True, exist_ok=True)
            tmp = str(paths.TMP / f"play-{row.prod['id']}.wav")
            r = subprocess.run([str(paths.FFMPEG), "-v", "error", "-y", "-i", path, tmp], capture_output=True, text=True, check=False)
            GLib.idle_add(self._decoded, row, tmp if r.returncode == 0 else None)

        threading.Thread(target=decode, daemon=True).start()

    def _decoded(self, row: ProductionRow, tmp: str | None) -> bool:
        if self._playing_row is not row:
            return False
        if tmp is None:
            self.app.toast("Could not decode the MP3 for playback")
            self._finished()
            return False
        self._tmp_play = tmp
        self.player.play(tmp, on_finished=lambda: GLib.idle_add(self._finished))
        return False

    def _finished(self) -> bool:
        if self._playing_row is not None:
            self._playing_row.set_playing(False)
        self._playing_row = None
        if self._tmp_play:
            try:
                os.remove(self._tmp_play)
            except OSError:
                pass
            self._tmp_play = None
        return False

    def _stop(self) -> None:
        self.player.stop()
        self._finished()

    def _folder(self, row: ProductionRow) -> None:
        Gtk.FileLauncher.new(Gio.File.new_for_path(str(paths.OUTPUT))).launch(self.get_root(), None, None)

    def _delete(self, row: ProductionRow) -> None:
        p = row.prod
        dialog = Adw.AlertDialog(heading=f"Delete '{p['title']}'?", body="The files are moved to data/trash inside the project folder.")
        dialog.add_response("cancel", "Cancel")
        dialog.add_response("delete", "Delete")
        dialog.set_response_appearance("delete", Adw.ResponseAppearance.DESTRUCTIVE)
        dialog.connect("response", self._on_delete_response, p)
        dialog.present(self.get_root())

    def _on_delete_response(self, dialog, response: str, p: dict) -> None:
        if response != "delete":
            return
        if self._playing_row is not None and self._playing_row.prod["id"] == p["id"]:
            self._stop()
        files = [p.get("mp3_path"), p.get("wav_path"), p.get("timeline_path"), p.get("mp4_path")]
        nn = p.get("neural") or {}
        files += [nn.get("dry_mp3"), nn.get("dry_wav")]
        tl = p.get("timeline_path") or ""
        if tl.endswith(".timeline.json"):
            files.append(tl.replace(".timeline.json", ".plan.json"))
        phrases.move_to_trash(*[f for f in files if f])
        self.db.delete_production(p["id"])
        self.app.toast(f"'{p['title']}' moved to trash")
        self.reload()

    # ---- video -----------------------------------------------------------------------
    def _video_params(self, production_id: int) -> dict:
        s = self.settings
        return {"production_id": production_id, "resolution": s.get("video.resolution", "1080p"), "fps": int(s.get("video.fps", 30)),
                "codec": s.get("video.codec", "h264"), "bitrate_k": int(s.get("video.bitrate_k", 10000)),
                "generators": dict(s.get("video.generators") or {}), "ai_stills": bool(s.get("video.ai_stills", False)),
                "ai_count": int(s.get("video.ai_count", 8))}

    def _make_video(self, row: ProductionRow) -> None:
        self.app.jobs.submit("render_video", self._video_params(row.prod["id"]))
        self.app.toast(f"Rendering the video for '{row.prod['title']}' - watch the Jobs tab")

    def _play_video(self, row: ProductionRow) -> None:
        Gtk.FileLauncher.new(Gio.File.new_for_path(row.prod["mp4_path"])).launch(self.get_root(), None, None)

    # ---- job events ----------------------------------------------------------------
    def _on_job_event(self, job_id, kind, params, event) -> None:
        if kind == "render_video":
            name = event.get("event")
            if name == "progress":
                self.progress.set_fraction(float(event.get("fraction", 0.0)))
                self.status_label.set_label(f"Video #{job_id}: {event.get('message', '')}")
            elif name == "finished":
                self.progress.set_fraction(0.0)
                r = event.get("result") or {}
                if event.get("status") == "done":
                    self.status_label.set_label(f"Video done: {os.path.basename(r.get('mp4', ''))} · {r.get('shots')} shots · {r.get('fps_achieved')} fps · {r.get('seconds')} s")
                    self.app.toast("Video ready - press the screen icon on the production to open it")
                else:
                    self.status_label.set_label(f"Video {event.get('status')}: {(event.get('message') or '').splitlines()[0]}")
                self.reload()
            return
        if kind != "compose":
            return
        name = event.get("event")
        if name == "progress":
            self.progress.set_fraction(float(event.get("fraction", 0.0)))
            self.status_label.set_label(f"Composing #{job_id}: {event.get('message', '')}")
        elif name == "finished":
            self.progress.set_fraction(0.0)
            r = event.get("result") or {}
            if event.get("status") == "done" and params.get("video") and r.get("production_id"):
                self.app.jobs.submit("render_video", self._video_params(int(r["production_id"])))
            if event.get("status") == "done":
                used = r.get("phrases") or []
                self.status_label.set_label(f"Done: {r.get('title')} · {r.get('bpm')} BPM · {r.get('key')} · {fmt_dur(r.get('duration_s'))} · "
                                            f"rendered in {r.get('render_s')} s" + (f" · phrases: {', '.join(used)}" if used else ""))
                self.app.toast(f"'{r.get('title')}' is ready ({r.get('bpm')} BPM, {r.get('key')})")
            else:
                self.status_label.set_label(f"Compose {event.get('status')}: {(event.get('message') or '').splitlines()[0]}")
            self.reload()

    def shutdown(self) -> None:
        self._stop()
