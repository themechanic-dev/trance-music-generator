"""In-app user guide (English). Written together with each feature, not after."""

from __future__ import annotations

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gtk  # noqa: E402

from tmg import __version__  # noqa: E402

SECTIONS: list[tuple[str, str, list[tuple[str, str]]]] = [
    (
        "What this app does",
        "Trance Music Generator is a studio that learns how trance is built from your own music library, "
        "composes brand-new tracks with different tempo, rhythm and structure every time, drops spoken phrases "
        "captured from the system audio into the music, and renders MP3 (audio) or MP4 (audio + GPU visuals). "
        "Everything runs locally on this machine and its NVIDIA GPU; nothing is installed system-wide.",
        [
            ("Available now (all phases)", "Capturing phrases from the system audio, the phrase bank with the trim editor, library import and analysis, the style profile ('What it learned'), composing new tracks from the profile with phrases inside them, optional MusicGen textures with A/B, MP3 / WAV export, and MP4 videos with GPU visuals that follow the music."),
            ("Later", "Fine-tuning MusicGen on your library once the whole collection is analysed; Stable Audio Open once a HuggingFace token is saved."),
        ],
    ),
    (
        "Recording a phrase from the system audio",
        "The app listens to what your speakers play (the monitor of the audio output), so anything you can hear "
        "can become a phrase: a line from a movie, a YouTube video, a stream, an advert.",
        [
            ("1. Open the Phrases tab", "Check the Source row: it shows the output that will be recorded. By default it follows the system output."),
            ("2. Optionally type a name", "In 'Phrase name'. Leave it empty and the phrase is named after the time it was recorded."),
            ("3. Press Record", "The button turns into Stop, the timer runs and the level meter moves when audio is actually being captured. "
                               "If the meter stays at -inf while something plays, the wrong output is selected - fix it in Settings."),
            ("4. Play the phrase", "Start the video or movie where the phrase is. Do not worry about silence before it: it is trimmed automatically."),
            ("5. Press Stop", "The recording is saved to the phrase bank. A clean-up job trims leading/trailing silence and normalises the volume "
                             "(both can be switched off in Settings). The untouched original is kept as a .raw.wav file next to it."),
        ],
    ),
    (
        "Editing a phrase - cutting the start and the end",
        "The editor opens by itself as soon as a recording is cleaned up. You can also open it any time with the "
        "scissors button on a phrase, or by clicking the phrase in the list. Only the highlighted (orange) part is kept.",
        [
            ("1. Find the phrase on the waveform", "The time ruler under the waveform shows seconds. The two yellow handles mark the start and the end of what you keep."),
            ("2. Move the handles", "Drag them, click anywhere on the waveform to jump the nearest handle there, or type exact seconds in the "
                                    "Start / End fields (the arrows move them in 0.05 s steps)."),
            ("3. Listen to the edges", "'Start edge' plays the first 1.5 s of the selection and 'End edge' the last 1.5 s, so you can hear "
                                        "exactly where the phrase begins and ends. 'Play selection' plays the whole highlighted part."),
            ("4. Cut", "'Cut to selection' keeps only the highlighted part and throws the rest away. Cut as many times as you need "
                       "to get closer - each cut reloads the waveform so you can refine it again."),
            ("Restore original", "Brings back the untouched recording (the .raw.wav file), undoing every cut and the automatic clean-up."),
            ("Listen, rename, delete", "The play button on each row plays the whole phrase. Delete moves the files to the data/trash folder "
                                      "inside the project - nothing is destroyed silently."),
        ],
    ),
    (
        "Learning from your library",
        "The Library tab is where the app learns how your trance is built. Nothing is copied or altered - "
        "each track is analysed once and only numbers are kept: tempo, beat grid, rhythm patterns, structure, key.",
        [
            ("Three ways to import - your choice", "'Import a track' = one file. 'Import a CD' = one folder, only the tracks directly in it. "
                                                    "'Import a collection' = a folder that contains many CD folders (200 CDs = 200 folders), all of them, recursively. "
                                                    "Dropping files or folders on the page also works (a folder with CD sub-folders counts as a collection)."),
            ("What happens next", "Tracks are added to the library and, by default, the analysis starts right away in the background. "
                                  "Each track is split into drums / bass / other / vocals on the GPU (Demucs), then beats, bars, kick/snare/hat/bass "
                                  "patterns per bar, sections (intro, build, drop, breakdown, outro), key and bass movement are measured."),
            ("How long it takes", "About 30 s per track on this machine. A collection of 2000 tracks is a matter of hours - leave it running. "
                                  "Every track is saved as soon as it is done, so Pause, a crash or a reboot lose nothing: 'Analyse queued tracks' resumes."),
            ("What it learned", "Updates while the analysis runs (every few tracks) and at the end; 'Rebuild profile' refreshes it at any time. Not averages but distributions: the tempo histogram, the most common section sequences and their lengths in bars, "
                                "what usually follows a breakdown, the most common kick / snare / hat / bass patterns, keys and bass movement. "
                                "This profile is what the composer will draw from - a different library gives different music."),
            ("Listening to the separation", "The stems of the first tracks of each run are kept as MP3 (Settings > Library) - use the folder button on the track."),
            ("Vocal phrases from the library", "Spoken or sung bits found in the vocals stem are saved into the phrase bank, tagged 'from the library' "
                                              "(Settings > Library: on/off, how many per track, threshold). Filter the phrase bank with 'From library'."),
            ("Removing", "'Remove from the library' forgets the track; the audio file is never touched. 'Analyse again' re-queues it."),
        ],
    ),
    (
        "Composing new tracks",
        "The Compose tab draws brand-new tracks from 'What it learned'. Nothing is sampled or copied from the library - "
        "only its habits are used: how fast, how long, how the sections follow each other, which rhythms, which keys.",
        [
            ("How a track is drawn", "Tempo comes from the tempo histogram, the length from the library's lengths, the structure from a walk over "
                                     "the learned transitions ('after a breakdown, 62 % drop') with learned section lengths, the kick / snare / hat / bass "
                                     "rhythms from the most common bar patterns, the key from the keys found, the chords from the bass movement."),
            ("Why every track is different", "Each draw is independent: one track may be 128 BPM with a long breakdown and a rolling bass, the next 145 "
                                             "with short drops - because that is what the library contains. Import a different library and the music changes."),
            ("Seed", "Every track shows its seed. Type a seed to get exactly that track again (same profile, same settings). Leave it empty for new ones."),
            ("Sound flavor", "Only the timbre: kick punch, lead type (supersaw or acid), pad darkness. 'Auto' picks it from the tempo. Rhythm and structure never come from here."),
            ("Output", "MP3 (tagged with BPM, key, seed and structure), WAV, or both, in data/output. A .timeline.json next to each track lists every "
                       "section, bar, beat, kick, fill and riser - that is what the visuals (phase 5) will follow."),
            ("Listening", "The play button plays the track through the default output; the folder button opens data/output. "
                          "Delete moves the files to data/trash."),
        ],
    ),
    (
        "Phrases inside the music",
        "The Compose tab can drop phrases from the bank into every new track. The timeline knows where each "
        "breakdown and build is, so a phrase opens a breakdown or lands right before a drop - no guessing.",
        [
            ("Which phrases", "Random from your captured phrases (default), from the library phrases, from all, or only the ones you tick "
                              "under 'Select...'. A phrase is never used twice in the same track while others are available."),
            ("Where", "Breakdowns: one bar after the breakdown starts. Before drops: the phrase ends half a beat before the drop hits, "
                      "at the end of a build. 'Also allow the intro' adds a third option two bars into the intro."),
            ("How it sounds", "Level relative to the music (-6 dB is a good start), optional telephone / radio EQ, echoes in time with the "
                              "beat (dotted eighths), a little reverb, and 'Fit to the beat' which stretches up to 15 % so the phrase ends "
                              "exactly on a beat. The music steps back 4 dB while the words are spoken."),
            ("Where to see it", "Each production lists the phrases it used and where. The .timeline.json carries them too, so the "
                                "visuals (phase 5) can react to the words."),
            ("Tip", "Trim your phrases tight in the Phrases tab first: what you keep is what gets dropped in."),
        ],
    ),
    (
        "Video: visuals that follow the music",
        "Choose 'MP3 + MP4 video' (or 'WAV + MP4 video') in the Compose tab and every new track also gets a video. "
        "For an existing production press the camera icon on its row. The visuals never analyse the audio blindly: "
        "the composer's timeline tells them where every kick, fill, build, drop, breakdown and phrase is, and the "
        "real spectrum of the track drives the rest.",
        [
            ("What you will see", "A pulse on every kick, a flash and a hard cut on every drop, calm liquid pictures in the breakdowns, "
                                  "a picture change on downbeats inside long sections, a warm lift while a phrase is spoken."),
            ("Generators", "Fourteen, all on the GPU: plasma, waves, tunnel, domainwarp, flow, reaction (Gray-Scott), kaleidoscope, "
                           "julia, mandelbulb, menger, metaballs, particles, spectrum, stills. Settings > Visual generators: each on/off "
                           "with a weight. Calm ones go to intros, breakdowns and outros; medium to builds; intense to drops."),
            ("Stills", "The 'stills' generator moves through images with a slow camera and a liquid warp: your own JPG/PNG files in "
                       "data/images, or AI stills made by SD-Turbo on the GPU (Settings > Video: switch on, ~2.5 GB download once)."),
            ("Palettes", "data/palettes.json - edit or add palettes; a palette is picked per section."),
            ("Output", "MP4 next to the MP3 in data/output: H.264 or H.265 with NVENC (or software x264), 1080p 30 fps by default, AAC audio. "
                       "A 6-minute track renders in roughly a minute or two on this GPU. Open it with the screen icon on the production."),
        ],
    ),
    (
        "Neural sound (MusicGen)",
        "The numpy composer stays the orchestrator - it knows every bar. MusicGen adds what it cannot: real timbre. "
        "Switch it on in the Compose tab under 'Neural sound'.",
        [
            ("What happens", "Two ~30 s clips are generated from the plan: an atmosphere (intros, breakdowns, outros) and an energy layer "
                             "(builds, drops). Each clip's tempo is measured; if it is within 12 % of the track's tempo it is stretched to match, "
                             "otherwise it is used as a free texture. Clips are looped bar by bar under each section with one-bar fades, "
                             "high-passed at 150 Hz so the kick and bass stay ours, and ducked by the kick."),
            ("Models", "stereo-small: fast (35 s per clip), stereo. medium: better timbre, 70 s per clip, mono. melody: follows the harmony of "
                       "our own rendered track (the model listens to the section it will play under), ~6 GB download the first time."),
            ("A/B", "'Keep a dry copy' writes the same track without the neural layer as '<track>-dry' in data/output. Listen to both; "
                    "if the neural layer does not earn its place, leave it off - nothing else changes."),
            ("Stable Audio Open", "Gated on HuggingFace: save your token in Settings and it becomes an option in a later update."),
            ("Fine-tuning on your library", "Not done yet: a real fine-tune (LoRA over thousands of 30 s clips with captions from the analysis) is "
                                            "hours of training and only worth it once the whole collection is analysed. The analysis already stores "
                                            "what such a dataset needs (tempo, key, sections per track)."),
        ],
    ),
    (
        "Jobs and log",
        "Heavy work (clean-up, analysis, separation, composition, rendering) runs one job at a time in a separate worker "
        "process on the Python environment inside the project folder, so this window never freezes.",
        [
            ("Jobs tab", "Shows every job with its progress and message. Queued and running jobs can be cancelled."),
            ("Log", "Everything the app and the workers report. The same text is written to data/logs/tmg.log."),
        ],
    ),
    (
        "Settings",
        "All settings are saved immediately to data/settings.json inside the project folder.",
        [
            ("Audio source", "Which output to record from. 'System default' follows whatever output the system currently uses "
                             "(HDMI, headphones, ...). Pick a specific one if you route audio elsewhere."),
            ("Trim silence automatically", "Cuts the quiet part before and after the phrase. The threshold is in dB below full scale "
                                           "(-45 dB is a good default); padding keeps a little air around the phrase."),
            ("Normalise peak", "Scales the phrase so that its loudest sample sits at the target (-1 dB by default), so all phrases have a similar level."),
            ("HuggingFace token", "Needed only for gated models such as Stable Audio Open. Create a free account at huggingface.co, "
                                  "open the model page, click to accept access, then create a token under Settings > Access Tokens (Read) "
                                  "and paste it here. It is stored in models/hf/token inside the project folder and never shown again."),
            ("Environment check", "Runs a job that verifies the GPU, CUDA, ffmpeg with NVENC (a real test encode), the libraries and the downloaded models."),
        ],
    ),
    (
        "Where the files live",
        "The whole installation is inside the project folder - delete the folder and nothing is left behind.",
        [
            ("data/phrases", "The phrase bank (WAV, 48 kHz stereo) plus the .raw.wav originals."),
            ("data/trash", "Deleted phrases."),
            ("data/logs", "The application log."),
            ("models", "Downloaded AI models (Demucs, MusicGen, ...)."),
            ("tools, .venv", "ffmpeg, the Python environment and every library."),
        ],
    ),
]


class HelpPage(Adw.PreferencesPage):
    def __init__(self, app) -> None:
        super().__init__()
        self.set_title("Help")
        for title, description, items in SECTIONS:
            group = Adw.PreferencesGroup(title=title, description=description)
            for item_title, item_text in items:
                row = Adw.ActionRow(title=item_title, subtitle=item_text)
                row.set_subtitle_lines(0)
                row.set_title_lines(0)
                group.add(row)
            self.add(group)
        footer = Adw.PreferencesGroup()
        lbl = Gtk.Label(label=f"Trance Music Generator {__version__}", halign=Gtk.Align.CENTER)
        lbl.add_css_class("dim-label")
        footer.add(lbl)
        self.add(footer)
