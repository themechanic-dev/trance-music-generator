# Περιβάλλον του Trance Music Generator — ΟΛΑ μέσα στον φάκελο του project.
# Χρήση: source env.sh
TMG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TMG_ROOT
export UV_PYTHON_INSTALL_DIR="$TMG_ROOT/tools/python"
export UV_CACHE_DIR="$TMG_ROOT/tools/uv-cache"
export HF_HOME="$TMG_ROOT/models/hf"
export TORCH_HOME="$TMG_ROOT/models/torch"
export XDG_CACHE_HOME="$TMG_ROOT/tools/cache"
export TMG_FFMPEG="$TMG_ROOT/tools/ffmpeg/bin/ffmpeg"
export TMG_FFPROBE="$TMG_ROOT/tools/ffmpeg/bin/ffprobe"
export PATH="$TMG_ROOT/tools/ffmpeg/bin:$TMG_ROOT/tools:$PATH"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$TMG_ROOT${PYTHONPATH:+:$PYTHONPATH}"
