#!/usr/bin/env bash
# Trance Music Generator - one-shot setup on a fresh Linux box. Everything lands INSIDE this folder:
#   tools/uv, tools/python (Python 3.12), .venv (torch + CUDA + Demucs + MusicGen + librosa + moderngl ...),
#   tools/ffmpeg (static build with NVENC), data/, models/ (downloaded on first use).
# Nothing is installed system-wide. Re-running is safe; it only fills in what is missing.
#
# System requirements (already present on Ubuntu 24.04+ with GNOME):
#   - python3 with PyGObject for GTK 4 + libadwaita  (packages: python3-gi gir1.2-gtk-4.0 gir1.2-adw-1)
#   - PipeWire (pw-record / pw-play) for capturing the system audio
#   - an NVIDIA GPU with a recent driver (CUDA 12.x/13.x capable) - for Demucs, MusicGen, SD-Turbo, NVENC
#
# Usage:  ./install.sh            set up
#         ./install.sh --desktop  also add a launcher to the applications menu (user-level, one .desktop file)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
mkdir -p tools models data

step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

step "checking the system"
for tool in python3 curl tar; do command -v "$tool" >/dev/null || { echo "missing: $tool"; exit 1; }; done
python3 -c "import gi; gi.require_version('Gtk','4.0'); gi.require_version('Adw','1'); from gi.repository import Gtk, Adw" 2>/dev/null \
  || { echo "GTK 4 / libadwaita bindings for python3 are missing (python3-gi gir1.2-gtk-4.0 gir1.2-adw-1)"; exit 1; }
command -v pw-record >/dev/null || echo "warning: pw-record not found - the phrase capture will not work without PipeWire"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "warning: no NVIDIA driver found - GPU features will fail"

step "uv (Python package manager, static binary)"
if [ ! -x tools/uv ]; then
  curl -sL --max-time 300 -o tools/uv.tar.gz https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz
  tar -xzf tools/uv.tar.gz -C tools && mv tools/uv-x86_64-unknown-linux-gnu/uv tools/uv-x86_64-unknown-linux-gnu/uvx tools/ && rm -rf tools/uv-x86_64-unknown-linux-gnu tools/uv.tar.gz
fi
tools/uv --version

# shellcheck source=env.sh
source env.sh

step "Python 3.12 (managed by uv, inside tools/python)"
tools/uv python install 3.12
[ -x .venv/bin/python ] || tools/uv venv --python 3.12 .venv
.venv/bin/python --version

step "worker environment (.venv) - several GB the first time (PyTorch + CUDA libraries)"
tools/uv pip install --python .venv/bin/python -r requirements.txt

step "ffmpeg (static build with NVENC, inside tools/ffmpeg)"
if [ ! -x tools/ffmpeg/bin/ffmpeg ]; then
  asset="$(curl -s --max-time 60 https://api.github.com/repos/BtbN/FFmpeg-Builds/releases/latest | grep -o 'ffmpeg-n[0-9.]*-latest-linux64-gpl-[0-9.]*\.tar\.xz' | head -1)"
  curl -sL --max-time 600 -o tools/ffmpeg.tar.xz "https://github.com/BtbN/FFmpeg-Builds/releases/latest/download/$asset"
  tar -xJf tools/ffmpeg.tar.xz -C tools && rm -f tools/ffmpeg.tar.xz && mv tools/ffmpeg-n*-linux64-gpl-* tools/ffmpeg
fi
tools/ffmpeg/bin/ffmpeg -version | head -1

step "self-check"
.venv/bin/python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available())"
.venv/bin/python -m pytest -q -p no:warnings tests/test_config_db.py tests/test_phrases.py 2>&1 | tail -1

if [ "${1:-}" = "--desktop" ]; then
  step "applications-menu launcher (user-level)"
  mkdir -p "$HOME/.local/share/applications"
  cat > "$HOME/.local/share/applications/trance-music-generator.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=Trance Music Generator
Comment=Learns trance from your library, composes new tracks, captures phrases from the system audio, renders MP3/MP4
Exec="$HERE/trance-music-generator"
Path=$HERE
Icon=audio-x-generic
Terminal=false
Categories=AudioVideo;Audio;Music;
DESKTOP
  update-desktop-database "$HOME/.local/share/applications" 2>/dev/null || true
  echo "launcher written to ~/.local/share/applications/trance-music-generator.desktop"
fi

step "done"
echo "Start the app with:  ./trance-music-generator"
echo "Models (Demucs, MusicGen, SD-Turbo) download into models/ the first time each feature is used."
