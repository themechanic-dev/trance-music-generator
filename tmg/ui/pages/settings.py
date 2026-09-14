"""Settings tab: capture, library, video, HuggingFace token, neural model self-test, environment check, about."""

from __future__ import annotations

import os

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gio, GLib, Gtk  # noqa: E402

from tmg import SIGNATURE_LINE, SIGNATURE_MARK, __version__, log, paths  # noqa: E402
from tmg.capture import pipewire  # noqa: E402
from tmg.db import utc_now as dbmod_utc_now  # noqa: E402
from tmg.jobs.kinds import model_test  # noqa: E402  (stdlib-only at import time; the heavy work runs in the worker)
from tmg.music.neural import MODELS as NEURAL_MODELS  # noqa: E402
from tmg.visuals.sequencer import GENERATORS  # noqa: E402

_log = log.get("ui.settings")


def esc(text) -> str:
    return GLib.markup_escape_text(str(text))


def format_env_report(r: dict) -> str:
    lines = []
    t = r.get("torch", {})
    if "error" in t:
        lines.append(f"PyTorch: ERROR - {t['error']}")
    else:
        line = f"PyTorch {t.get('version')} - CUDA {'available' if t.get('cuda_available') else 'NOT available'}"
        if t.get("cuda_available"):
            line += f": {t.get('device')}, {t.get('vram_free_gb')} of {t.get('vram_total_gb')} GB VRAM free"
        lines.append(line)
    f = r.get("ffmpeg", {})
    if f.get("present"):
        nv = "OK (real test encode)" if f.get("nvenc_h264") else f"FAILED {f.get('nvenc_error', '')}".strip()
        lines.append(f"ffmpeg: {f.get('version')} - NVENC h264: {nv}")
    else:
        lines.append(f"ffmpeg: MISSING at {f.get('path')}")
    lines.append("Libraries: " + ", ".join(f"{k} {v}" for k, v in r.get("libs", {}).items()))
    models = r.get("models", [])
    lines.append("Models: " + (", ".join(f"{m['name']} ({m['size_gb']} GB)" for m in models) if models else "none downloaded yet"))
    lines.append(f"HuggingFace token: {'saved' if r.get('hf_token_present') else 'not set'}")
    pw = r.get("pipewire_tools", {})
    lines.append("PipeWire tools: " + ", ".join(f"{k} {'ok' if v else 'MISSING'}" for k, v in pw.items()))
    lines.append(f"Worker Python {r.get('python', {}).get('version')}  |  disk free {r.get('disk_free_gb')} GB")
    return "\n".join(lines)


class SettingsPage(Adw.PreferencesPage):
    def __init__(self, app) -> None:
        super().__init__()
        self.app = app
        self.s = app.settings
        self._sinks: list[pipewire.Sink] = []
        self._env_job: int | None = None
        self._build_capture()
        self._build_library()
        self._build_video()
        self._build_huggingface()
        self._build_neural_model()
        self._build_environment()
        self._build_about()
        app.add_job_listener(self._on_job_event)

    # ---- capture -------------------------------------------------------------------
    def _build_capture(self) -> None:
        g = Adw.PreferencesGroup(title="Capture", description="Where phrases are recorded from, and how they are cleaned up after Stop.")
        self.sink_row = Adw.ComboRow(title="Audio source", subtitle="'System default' follows whatever output the system uses right now")
        refresh = Gtk.Button(icon_name="view-refresh-symbolic", valign=Gtk.Align.CENTER, tooltip_text="Refresh the list of outputs")
        refresh.add_css_class("flat")
        refresh.connect("clicked", lambda *_: self._fill_sinks())
        self.sink_row.add_suffix(refresh)
        self._fill_sinks()
        self.sink_row.connect("notify::selected", self._on_sink_changed)
        g.add(self.sink_row)

        self.trim_row = Adw.SwitchRow(title="Trim silence automatically", subtitle="Cut leading and trailing silence right after Stop")
        self.trim_row.set_active(bool(self.s.get("capture.auto_trim", True)))
        self.trim_row.connect("notify::active", lambda r, _p: self.s.set("capture.auto_trim", r.get_active()))
        g.add(self.trim_row)

        self.thr_row = Adw.SpinRow.new_with_range(-80, -10, 1)
        self.thr_row.set_title("Silence threshold (dB)")
        self.thr_row.set_subtitle("Anything quieter than this counts as silence")
        self.thr_row.set_value(float(self.s.get("capture.trim_threshold_db", -45.0)))
        self.thr_row.connect("notify::value", lambda r, _p: self.s.set("capture.trim_threshold_db", float(r.get_value())))
        g.add(self.thr_row)

        self.pad_row = Adw.SpinRow.new_with_range(0, 1000, 10)
        self.pad_row.set_title("Padding (ms)")
        self.pad_row.set_subtitle("Air kept before and after the phrase")
        self.pad_row.set_value(float(self.s.get("capture.trim_pad_ms", 120)))
        self.pad_row.connect("notify::value", lambda r, _p: self.s.set("capture.trim_pad_ms", int(r.get_value())))
        g.add(self.pad_row)

        self.norm_row = Adw.SwitchRow(title="Normalise peak", subtitle="Scale the phrase so its loudest sample hits the target level")
        self.norm_row.set_active(bool(self.s.get("capture.normalize", True)))
        self.norm_row.connect("notify::active", lambda r, _p: self.s.set("capture.normalize", r.get_active()))
        g.add(self.norm_row)

        self.peak_row = Adw.SpinRow.new_with_range(-12, 0, 0.5)
        self.peak_row.set_digits(1)
        self.peak_row.set_title("Peak target (dB)")
        self.peak_row.set_value(float(self.s.get("capture.normalize_peak_db", -1.0)))
        self.peak_row.connect("notify::value", lambda r, _p: self.s.set("capture.normalize_peak_db", float(r.get_value())))
        g.add(self.peak_row)

        folder = Adw.ActionRow(title="Phrase folder", subtitle=esc(str(paths.PHRASES)))
        open_btn = Gtk.Button(icon_name="folder-open-symbolic", valign=Gtk.Align.CENTER, tooltip_text="Open")
        open_btn.add_css_class("flat")
        open_btn.connect("clicked", lambda *_: Gtk.FileLauncher.new(Gio.File.new_for_path(str(paths.PHRASES))).launch(self.get_root(), None, None))
        folder.add_suffix(open_btn)
        g.add(folder)
        self.add(g)

    def _fill_sinks(self) -> None:
        self._sinks = pipewire.list_sinks()
        names = ["System default"] + [s.description for s in self._sinks]
        self.sink_row.set_model(Gtk.StringList.new(names))
        setting = self.s.get("capture.sink", "default")
        index = 0
        for i, s in enumerate(self._sinks):
            if s.name == setting:
                index = i + 1
        self.sink_row.set_selected(index)

    def _on_sink_changed(self, row, _p) -> None:
        i = row.get_selected()
        value = "default" if i == 0 or i - 1 >= len(self._sinks) else self._sinks[i - 1].name
        if value != self.s.get("capture.sink"):
            self.s.set("capture.sink", value)
            _log.info("audio source set to %s", value)
            if self.app.window is not None:
                self.app.window.capture_page.refresh_source()

    # ---- library -------------------------------------------------------------------
    def _build_library(self) -> None:
        g = Adw.PreferencesGroup(title="Library analysis", description="How imported tracks are analysed. One track takes about "
                                 "30 s on this GPU with stem separation; the analysis runs in the background and resumes where it stopped.")
        auto = Adw.SwitchRow(title="Analyse right after import", subtitle="Otherwise press 'Analyse queued tracks' yourself")
        auto.set_active(bool(self.s.get("library.auto_analyze", True)))
        auto.connect("notify::active", lambda r, _p: self.s.set("library.auto_analyze", r.get_active()))
        g.add(auto)
        demucs = Adw.SwitchRow(title="Separate stems with Demucs (GPU)", subtitle="Drums, bass, other, vocals - needed for clean patterns and for vocal phrases. Off = faster but rougher")
        demucs.set_active(bool(self.s.get("library.demucs", True)))
        demucs.connect("notify::active", lambda r, _p: self.s.set("library.demucs", r.get_active()))
        g.add(demucs)
        keep = Adw.SpinRow.new_with_range(0, 20, 1)
        keep.set_title("Keep stems of the first N tracks")
        keep.set_subtitle("Saved as MP3 in data/library/stems so you can listen and judge the separation")
        keep.set_value(int(self.s.get("library.keep_stems_first", 3)))
        keep.connect("notify::value", lambda r, _p: self.s.set("library.keep_stems_first", int(r.get_value())))
        g.add(keep)
        vocals = Adw.SwitchRow(title="Extract vocal phrases into the phrase bank", subtitle="Spoken or sung bits found in the vocals stem, normalised, tagged 'library'")
        vocals.set_active(bool(self.s.get("library.extract_vocals", True)))
        vocals.connect("notify::active", lambda r, _p: self.s.set("library.extract_vocals", r.get_active()))
        g.add(vocals)
        vmax = Adw.SpinRow.new_with_range(0, 10, 1)
        vmax.set_title("Vocal phrases per track (max)")
        vmax.set_value(int(self.s.get("library.vocal_max_count", 3)))
        vmax.connect("notify::value", lambda r, _p: self.s.set("library.vocal_max_count", int(r.get_value())))
        g.add(vmax)
        vthr = Adw.SpinRow.new_with_range(-60, -10, 1)
        vthr.set_title("Vocal detection threshold (dB)")
        vthr.set_subtitle("Lower = more (and quieter) segments count as vocals")
        vthr.set_value(float(self.s.get("library.vocal_threshold_db", -35.0)))
        vthr.connect("notify::value", lambda r, _p: self.s.set("library.vocal_threshold_db", float(r.get_value())))
        g.add(vthr)
        self.add(g)

    # ---- video & visuals -------------------------------------------------------------
    def _build_video(self) -> None:
        s = self.s
        g = Adw.PreferencesGroup(title="Video (MP4)", description="How the video of a production is rendered: GPU visuals that follow "
                                 "the timeline (kick, drop, breakdown, phrases) and the real spectrum, encoded with NVENC.")
        res = Adw.ComboRow(title="Resolution")
        res_keys = ["720p", "1080p", "1440p", "2160p"]
        res.set_model(Gtk.StringList.new(["1280 x 720", "1920 x 1080 (default)", "2560 x 1440", "3840 x 2160 (4K)"]))
        res.set_selected(res_keys.index(s.get("video.resolution", "1080p")) if s.get("video.resolution", "1080p") in res_keys else 1)
        res.connect("notify::selected", lambda r, _p: s.set("video.resolution", res_keys[r.get_selected()]))
        g.add(res)
        fps = Adw.ComboRow(title="Frames per second")
        fps.set_model(Gtk.StringList.new(["30", "60"]))
        fps.set_selected(1 if int(s.get("video.fps", 30)) == 60 else 0)
        fps.connect("notify::selected", lambda r, _p: s.set("video.fps", 60 if r.get_selected() == 1 else 30))
        g.add(fps)
        codec = Adw.ComboRow(title="Codec", subtitle="NVENC on the GPU; x264 is the slow software fallback")
        codec_keys = ["h264", "hevc", "x264"]
        codec.set_model(Gtk.StringList.new(["H.264 NVENC (default)", "H.265 / HEVC NVENC", "H.264 software (x264)"]))
        codec.set_selected(codec_keys.index(s.get("video.codec", "h264")) if s.get("video.codec", "h264") in codec_keys else 0)
        codec.connect("notify::selected", lambda r, _p: s.set("video.codec", codec_keys[r.get_selected()]))
        g.add(codec)
        br = Adw.SpinRow.new_with_range(2000, 40000, 1000)
        br.set_title("Video bitrate (kbit/s)")
        br.set_value(int(s.get("video.bitrate_k", 10000)))
        br.connect("notify::value", lambda r, _p: s.set("video.bitrate_k", int(r.get_value())))
        g.add(br)
        ai = Adw.SwitchRow(title="AI stills (SD-Turbo on the GPU)", subtitle="Generates abstract images per production for the 'stills' "
                           "generator. Downloads ~2.5 GB the first time; a few seconds per image after that.")
        ai.set_active(bool(s.get("video.ai_stills", False)))
        ai.connect("notify::active", lambda r, _p: s.set("video.ai_stills", r.get_active()))
        g.add(ai)
        aic = Adw.SpinRow.new_with_range(2, 24, 1)
        aic.set_title("AI stills per production")
        aic.set_value(int(s.get("video.ai_count", 8)))
        aic.connect("notify::value", lambda r, _p: s.set("video.ai_count", int(r.get_value())))
        g.add(aic)
        imgs = Adw.ActionRow(title="Your own images for the 'stills' generator", subtitle=esc(f"Put JPG / PNG files in {paths.DATA / 'images'}"))
        ob = Gtk.Button(icon_name="folder-open-symbolic", valign=Gtk.Align.CENTER, tooltip_text="Open")
        ob.add_css_class("flat")

        def open_images(*_):
            (paths.DATA / "images").mkdir(parents=True, exist_ok=True)
            Gtk.FileLauncher.new(Gio.File.new_for_path(str(paths.DATA / "images"))).launch(self.get_root(), None, None)

        ob.connect("clicked", open_images)
        imgs.add_suffix(ob)
        g.add(imgs)
        self.add(g)

        gg = Adw.PreferencesGroup(title="Visual generators", description="Each one on or off, with a weight (how often it is picked). "
                                  "Calm generators play in intros, breakdowns and outros; medium ones in builds; intense ones in drops. "
                                  "Changes apply to the next video you render.")
        current = dict(s.get("video.generators") or {})
        for name, (mood, default_w, desc) in GENERATORS.items():
            row = Adw.SpinRow.new_with_range(0, 3, 0.5)
            row.set_digits(1)
            row.set_title(f"{name}  ·  {mood}")
            row.set_subtitle(desc + "  (0 = off)")
            row.set_value(float(current.get(name, 0.0 if name == "stills" else default_w)))
            row.connect("notify::value", lambda r, _p, n=name: self._set_generator_weight(n, float(r.get_value())))
            gg.add(row)
        self.add(gg)

    def _set_generator_weight(self, name: str, weight: float) -> None:
        current = dict(self.s.get("video.generators") or {})
        if not current:   # first change: materialise the defaults so one edit does not turn everything else off
            current = {n: (0.0 if n == "stills" else spec[1]) for n, spec in GENERATORS.items()}
        current[name] = weight
        self.s.set("video.generators", current)

    # ---- HuggingFace ---------------------------------------------------------------
    def _build_huggingface(self) -> None:
        g = Adw.PreferencesGroup(
            title="HuggingFace",
            description="Only needed for gated models such as Stable Audio Open. Create a free account at huggingface.co, "
                        "open the model's page and accept access, create a token under Settings > Access Tokens (Read), "
                        "then paste it here. It is stored in models/hf/token inside the project folder and never shown again.",
        )
        self.token_row = Adw.PasswordEntryRow(title="Access token")
        g.add(self.token_row)
        self.token_status = Adw.ActionRow(title="Status")
        g.add(self.token_status)
        save = Adw.ButtonRow(title="Save token")
        save.connect("activated", self._save_token)
        g.add(save)
        remove = Adw.ButtonRow(title="Remove saved token")
        remove.connect("activated", self._remove_token)
        g.add(remove)
        check = Adw.ButtonRow(title="Check token and model access")
        check.connect("activated", lambda *_: (self.app.jobs.submit("hf_check", {}), self.hf_result.set_subtitle("checking...")))
        g.add(check)
        self.hf_result = Adw.ActionRow(title="Last check", subtitle="not run yet")
        self.hf_result.set_subtitle_lines(0)
        g.add(self.hf_result)
        self._update_token_status()
        self.add(g)

    def _update_token_status(self) -> None:
        present = paths.HF_TOKEN_FILE.exists() and paths.HF_TOKEN_FILE.stat().st_size > 0
        self.token_status.set_subtitle("Saved" if present else "Not set")

    def _save_token(self, *_) -> None:
        text = self.token_row.get_text().strip()
        if not text:
            self.app.toast("Paste a token first")
            return
        paths.HF_HOME.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(paths.HF_TOKEN_FILE), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        self.token_row.set_text("")
        self._update_token_status()
        _log.info("HuggingFace token saved to %s", paths.HF_TOKEN_FILE)
        self.app.toast("Token saved" + ("" if text.startswith("hf_") else " (note: HuggingFace tokens usually start with hf_)"), 5)

    def _remove_token(self, *_) -> None:
        try:
            os.remove(paths.HF_TOKEN_FILE)
            self.app.toast("Token removed")
            _log.info("HuggingFace token removed")
        except FileNotFoundError:
            self.app.toast("No token was saved")
        self._update_token_status()

    # ---- neural model ----------------------------------------------------------------
    def _build_neural_model(self) -> None:
        g = Adw.PreferencesGroup(
            title="Neural model",
            description="The model the composer uses for neural textures (chosen in Compose > Neural sound): is it downloaded, "
                        "does it load on the GPU, does it make sound? The self-test loads it, generates a short clip and reports "
                        "load time, clip time and peak GPU memory. It runs as a job - the Jobs tab shows it running.",
        )
        self.model_row = Adw.ActionRow(title="Selected model")
        self.model_row.set_subtitle_lines(0)
        g.add(self.model_row)
        test = Adw.ButtonRow(title="Test the selected model")
        test.connect("activated", self._run_model_test)
        g.add(test)
        self.model_test_row = Adw.ActionRow(title="Self-test", subtitle="not run yet")
        self.model_test_row.set_subtitle_lines(0)
        g.add(self.model_test_row)
        self.connect("map", lambda *_: self._refresh_model_rows())
        self._refresh_model_rows()
        self.add(g)

    def _refresh_model_rows(self) -> None:
        key = self.s.get("neural.model", "stereo-small")
        spec = NEURAL_MODELS.get(key)
        if spec is None:
            self.model_row.set_subtitle(esc(f"{key}: unknown model - pick one in Compose > Neural sound"))
            return
        try:
            downloaded, size_gb = model_test.cache_info(spec["id"])
        except OSError:
            downloaded, size_gb = False, 0.0
        state = f"downloaded ({size_gb} GB in models/hf)" if downloaded else "not downloaded yet (fetched on first use or by the self-test)"
        self.model_row.set_subtitle(esc(f"{spec['label']}\n{spec['id']} - {state}"))
        last = self.s.get("neural.last_test") or {}
        if last.get("model") == key and last.get("summary"):
            self.model_test_row.set_subtitle(esc(last["summary"]))
        elif last.get("model"):
            self.model_test_row.set_subtitle(esc(f"not run for this model yet (last tested: {last['model']})"))
        else:
            self.model_test_row.set_subtitle("not run yet")

    def _run_model_test(self, *_) -> None:
        key = self.s.get("neural.model", "stereo-small")
        self.app.jobs.submit("model_test", {"model": key})
        self.model_test_row.set_subtitle(esc(f"{key}: queued..."))
        self.app.toast(f"Self-test of {key} started - see the Jobs tab")

    # ---- environment -----------------------------------------------------------------
    def _build_environment(self) -> None:
        g = Adw.PreferencesGroup(
            title="Environment",
            description="Verifies the GPU and CUDA, ffmpeg with NVENC (a real test encode), the libraries and the "
                        "downloaded models. Runs as a job - watch the Jobs tab.",
        )
        run = Adw.ButtonRow(title="Run environment check")
        run.connect("activated", self._run_env_check)
        g.add(run)
        self.env_row = Adw.ActionRow(title="Last result", subtitle="not run yet")
        self.env_row.set_subtitle_lines(0)
        g.add(self.env_row)
        self.add(g)

    def _run_env_check(self, *_) -> None:
        self._env_job = self.app.jobs.submit("env_check", {})
        self.env_row.set_subtitle("running...")
        self.app.toast("Environment check started - see the Jobs tab")

    def _on_job_event(self, job_id, kind, params, event) -> None:
        if kind == "model_test":
            self._on_model_test_event(params, event)
            return
        if kind == "hf_check" and event.get("event") == "finished":
            r = event.get("result") or {}
            if event.get("status") != "done":
                self.hf_result.set_subtitle(esc(f"failed: {(event.get('message') or '').splitlines()[0]}"))
                return
            lines = [f"token: {r.get('token')}" + (f" ('{r.get('token_name')}', {r.get('role')}, permissions: {', '.join(r.get('permissions') or []) or 'none'})" if r.get("token") == "valid" else "")]
            for repo, status in (r.get("models") or {}).items():
                lines.append(f"{repo}: {status}")
            self.hf_result.set_subtitle(esc("\n".join(lines)))
            return
        if kind != "env_check" or event.get("event") != "finished":
            return
        if event.get("status") == "done":
            report = format_env_report(event.get("result") or {})
            self.env_row.set_subtitle(esc(report))
            _log.info("environment check:\n%s", report)
            dialog = Adw.AlertDialog(heading="Environment check", body=report)
            dialog.add_response("ok", "OK")
            dialog.present(self.get_root())
        else:
            self.env_row.set_subtitle(esc(f"failed: {event.get('message')}"))

    def _on_model_test_event(self, params: dict, event: dict) -> None:
        key = params.get("model", "?")
        name = event.get("event")
        if name == "started":
            self.model_test_row.set_subtitle(esc(f"{key}: starting the worker..."))
        elif name == "progress":
            self.model_test_row.set_subtitle(esc(f"{key}: {event.get('message', '')}"))
        elif name == "finished":
            r = dict(event.get("result") or {})
            if event.get("status") == "done" and r.get("working"):
                text = model_test.summary(r)
                self.app.toast(f"{key}: working")
            else:
                spec = NEURAL_MODELS.get(key, {})
                r.update(model=key, model_id=spec.get("id", "?"), working=False,
                         error=(event.get("message") or "failed").splitlines()[0])
                r.setdefault("tested_utc", dbmod_utc_now())
                text = model_test.summary(r)
                self.app.toast(f"{key}: NOT working - see the Jobs tab", 6)
            self.s.set("neural.last_test", {"model": key, "summary": text, "utc": r.get("tested_utc", "")})
            _log.info("neural model self-test:\n%s", text)
            self._refresh_model_rows()

    # ---- about ---------------------------------------------------------------------
    def _build_about(self) -> None:
        g = Adw.PreferencesGroup(title="About")
        g.add(Adw.ActionRow(title="Version", subtitle=__version__))
        g.add(Adw.ActionRow(title="Project folder", subtitle=esc(str(paths.ROOT))))
        g.add(Adw.ActionRow(title="Installation", subtitle="Python environment, ffmpeg, models and data all live inside the project folder - nothing system-wide."))
        # The HOME LAB signature: at the very end of Settings, under a thin line, below the version.
        signature = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2, halign=Gtk.Align.CENTER, margin_top=24, margin_bottom=12)
        signature.append(Gtk.Separator(margin_bottom=14, width_request=160))
        mark = Gtk.Label(label=SIGNATURE_MARK)
        mark.add_css_class("signature-mark")
        line = Gtk.Label(label=SIGNATURE_LINE)
        line.add_css_class("signature-line")
        version = Gtk.Label(label=f"v{__version__}")
        version.add_css_class("signature-line")
        signature.append(mark)
        signature.append(line)
        signature.append(version)
        g.add(signature)
        self.add(g)
