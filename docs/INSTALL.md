# Installing Trance Music Generator

For anyone with a Linux desktop and an NVIDIA GPU who wants to run the app. Everything lives inside one
folder; the only things written outside it are an optional launcher and icon in your own user profile.

## 1. What you need

**Hardware**

| | |
|---|---|
| GPU | NVIDIA with 12 GB of memory (developed on an RTX 3060 12 GB). 8 GB runs Demucs, MusicGen small and the video; MusicGen medium and Stable Audio Open need about 6 GB free while they run. |
| Disk | About 25 GB after the first run (PyTorch with CUDA, Demucs, MusicGen small, SD-Turbo). Add 10 GB for Stable Audio Open, 8 GB for MusicGen medium, and room for your library's stems. |
| RAM | 16 GB is comfortable. |
| Audio | PipeWire (the default on current Ubuntu and Fedora) - phrases are captured from its output monitor. |

**Software** - a distribution with GTK 4.14+, libadwaita 1.6+ and PipeWire: Ubuntu 25.04 or newer, Debian 13,
Fedora 41 or newer. Ubuntu 24.04 ships libadwaita 1.5 and is too old.

- NVIDIA driver **580 or newer** (CUDA 13). `nvidia-smi` must work before you start.
- Python 3 with the GTK 4 / libadwaita bindings, PipeWire's command-line tools, `git`, `curl`, `tar`:

  ```bash
  sudo apt install python3-gi python3-gi-cairo gir1.2-gtk-4.0 gir1.2-adw-1 pipewire-bin git curl
  ```

  Fedora: `sudo dnf install python3-gobject gtk4 libadwaita pipewire-utils git curl`.

Nothing else is installed system-wide. Python packages, the second Python, ffmpeg and the models all go
into the project folder.

## 2. Get the code

```bash
git clone https://github.com/themechanic-dev/trance-music-generator.git
cd trance-music-generator
```

## 3. Run the installer

```bash
./install.sh
```

Step by step, all inside the folder:

1. **Checks the system** - python3 with GTK 4 / libadwaita, `pw-record`, `nvidia-smi`. It stops with a clear
   message if the bindings are missing.
2. **uv** - a static binary (about 15 MB) into `tools/`. It is the package manager for the next steps.
3. **Python 3.12** - installed by uv into `tools/python`. The GUI runs on the system Python (that is where the
   GTK bindings are); the audio, ML and video workers run in this second Python. Both are needed.
4. **The worker environment** - `.venv/` with the pinned packages from `requirements.txt`: PyTorch with the
   CUDA 13 libraries, Demucs, transformers, diffusers, librosa, moderngl and friends. About 6 GB of downloads;
   ten minutes on a fast connection.
5. **ffmpeg** - a static build with NVENC (BtbN, about 150 MB) into `tools/ffmpeg`.
6. **Self-check** - PyTorch must see the GPU, and two quick test files run.

Re-running `./install.sh` is safe: it only fills in what is missing.

### Launcher and icon

```bash
./install.sh --desktop
```

Adds the app to the applications menu and the dock with its own icon. It writes exactly two things into your
profile: `~/.local/share/applications/org.tmg.TranceMusicGenerator.desktop` and the icon under
`~/.local/share/icons/hicolor/`. The launcher is named after the application id on purpose - that is how
GNOME matches the window to the entry; any other name shows a generic gear in the dock.

## 4. First start

```bash
./trance-music-generator
```

or from the applications menu. The app is single-instance: launching it twice raises the open window.

- **Settings > Environment > Run environment check** verifies the GPU and CUDA, ffmpeg with NVENC (a real test
  encode), the libraries, the downloaded models and the PipeWire tools. Do this once.
- The in-app **Help** tab explains every screen.
- Models download into `models/` the first time each feature is used:

| Model | When | Size |
|---|---|---|
| Demucs htdemucs | first library analysis | 80 MB |
| MusicGen stereo-small | first compose with neural textures | 1.2 GB |
| MusicGen medium / stereo-melody | if you pick them | 7.5 GB / 6 GB |
| SD-Turbo | AI stills for the video, if switched on | 2.5 GB |
| Stable Audio Open 1.0 | if you pick it (see below) | 9.5 GB |

## 5. Optional: Stable Audio Open

The model is gated on HuggingFace; the app needs your own free account and token.

1. Create an account at huggingface.co.
2. Open the model page `stabilityai/stable-audio-open-1.0`, fill in the short form and press
   **Agree and access repository**.
3. Settings > Access Tokens > **New token**, type **Read**. (A fine-grained token also needs
   "Read access to contents of all public gated repos you can access", or it fails with 403.)
4. In the app: **Settings > HuggingFace**, paste the token, **Save token**. It is stored in `models/hf/token`
   inside the project folder and never shown again.
5. Press **Check token and model access** - the model must say `ok`.
6. **Compose > Neural sound > Model**: pick *Stable Audio Open 1.0*.
7. **Settings > Neural model > Test the selected model** - downloads it (9.5 GB) the first time, loads it on
   the GPU, generates a short clip and reports load time, clip time and memory: `WORKING` or the reason.

## 6. Where things live

| Path | Contents |
|---|---|
| `tools/` | uv, Python 3.12, ffmpeg |
| `.venv/` | the worker Python environment |
| `models/` | HuggingFace cache (`models/hf`), Demucs checkpoints, the token |
| `data/settings.json` | every setting |
| `data/tmg.sqlite` | phrases, library, profile, productions, jobs |
| `data/phrases/` | captured and library phrases (WAV) |
| `data/library/` | stems of the analysed tracks |
| `data/output/` | the tracks and videos you make, with their timeline and plan files |
| `data/neural/` | cached neural clips |
| `data/logs/tmg.log` | the log |
| `env.sh` | the environment the launcher and workers use (`source env.sh` before running anything by hand) |

## 7. Updating

```bash
git pull
./install.sh
```

then restart the app. The installer adds any new packages; settings, data and models stay.

## 8. Uninstalling

Delete the folder. If you used `--desktop`, also remove the launcher and the icon:

```bash
rm ~/.local/share/applications/org.tmg.TranceMusicGenerator.desktop
rm ~/.local/share/icons/hicolor/*/apps/org.tmg.TranceMusicGenerator.*
```

Nothing else was written outside the folder.

## 9. If something goes wrong

| Symptom | Cause and fix |
|---|---|
| `GTK 4 / libadwaita bindings for python3 are missing` | Install the packages from section 1. |
| `libadwaita 1.6 or newer is needed` | The distribution is too old (Ubuntu 24.04 has 1.5). Use Ubuntu 25.04+, Debian 13 or Fedora 41+. |
| Self-check prints `CUDA False` | Driver older than 580, or no NVIDIA GPU visible. `nvidia-smi` has to work first. |
| Environment check: `NVENC h264: FAILED` | The driver lacks `libnvidia-encode`. Pick **x264** in Settings > Video; everything else still runs on the GPU. |
| `pw-record not found` | Install PipeWire's tools (`pipewire-bin` / `pipewire-utils`). Everything but phrase capture works without them. |
| The dock shows a gear instead of the icon | Run `./install.sh --desktop` again, then close and reopen the app. |
| Stable Audio: `gated` or 403 | Access not accepted on the model page, or the token cannot read gated repos (section 5). |
| A job fails | Open the **Jobs** tab: every job keeps its message. The full trace is in `data/logs/tmg.log`. |

Tests, if you want to run them yourself:

```bash
source env.sh && .venv/bin/python -m pytest -q
```
