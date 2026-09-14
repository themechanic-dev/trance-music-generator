"""System-audio capture with PipeWire: the monitor of the (default) sink through `pw-record`.

Verified in phase 0 (see the measurements file). Standard library only - runs in the GUI process.
"""

from __future__ import annotations

import json
import math
import os
import re
import signal
import struct
import subprocess
import time
import wave
from array import array
from dataclasses import dataclass

from tmg import log

_log = log.get("capture")


@dataclass(frozen=True)
class Sink:
    id: int
    name: str
    description: str


def _run(cmd: list[str], timeout: float = 5.0) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        _log.warning("%s failed: %s", cmd[0], exc)
        return ""
    return out.stdout


# ---- pure parsers (unit-tested) --------------------------------------------------------

def parse_default_sink(metadata_text: str) -> str | None:
    """`pw-metadata 0 default.audio.sink` -> node.name. The value is JSON inside single quotes."""
    m = re.search(r'"name"\s*:\s*"([^"]+)"', metadata_text)
    return m.group(1) if m else None


def parse_wpctl_inspect(text: str) -> str | None:
    """`wpctl inspect @DEFAULT_AUDIO_SINK@` -> node.name (the line carries a leading `*`)."""
    m = re.search(r'node\.name\s*=\s*"([^"]+)"', text)
    return m.group(1) if m else None


def parse_sinks(pw_dump_json: str) -> list[Sink]:
    try:
        objs = json.loads(pw_dump_json or "[]")
    except json.JSONDecodeError:
        return []
    sinks: list[Sink] = []
    for o in objs:
        props = (o.get("info") or {}).get("props") or {}
        if props.get("media.class") == "Audio/Sink" and props.get("node.name"):
            sinks.append(Sink(int(o.get("id", 0)), props["node.name"], props.get("node.description") or props["node.name"]))
    return sinks


# ---- live queries ------------------------------------------------------------------------

def default_sink_name() -> str | None:
    name = parse_default_sink(_run(["pw-metadata", "0", "default.audio.sink"]))
    if not name:
        name = parse_wpctl_inspect(_run(["wpctl", "inspect", "@DEFAULT_AUDIO_SINK@"]))
    return name


def list_sinks() -> list[Sink]:
    return parse_sinks(_run(["pw-dump"], timeout=8))


def describe_sink(name: str | None) -> str:
    if not name:
        return "-"
    for s in list_sinks():
        if s.name == name:
            return s.description
    return name


def resolve_sink(setting: str) -> str | None:
    """Settings value ('default' or a node.name) -> a node.name that exists right now."""
    if setting and setting != "default":
        if any(s.name == setting for s in list_sinks()):
            return setting
        _log.warning("configured sink %r not found - falling back to the system default", setting)
    return default_sink_name()


# ---- WAV helpers -------------------------------------------------------------------------

def wav_is_valid(path: str) -> bool:
    try:
        with wave.open(path, "rb") as w:
            return w.getnframes() > 0
    except (wave.Error, EOFError, OSError):
        return False


def repair_wav_header(path: str) -> bool:
    """If pw-record died hard the header says size 0. Rewrite the sizes from the file size."""
    size = os.path.getsize(path)
    with open(path, "r+b") as f:
        hdr = f.read(12)
        if len(hdr) < 12 or hdr[:4] != b"RIFF" or hdr[8:12] != b"WAVE":
            return False
        pos = 12
        while pos + 8 <= size:
            f.seek(pos)
            cid, csz = struct.unpack("<4sI", f.read(8))
            if cid == b"data":
                data_size = size - (pos + 8)
                f.seek(pos + 4)
                f.write(struct.pack("<I", data_size))
                f.seek(4)
                f.write(struct.pack("<I", size - 8))
                return True
            pos += 8 + csz + (csz & 1)
    return False


def tail_level_db(path: str, rate: int, channels: int, window_s: float = 0.1) -> tuple[float, float]:
    """RMS and peak (dBFS) of the last `window_s` of a file that is being written right now."""
    frame_bytes = 2 * channels
    want = int(rate * window_s) * frame_bytes
    try:
        size = os.path.getsize(path)
    except OSError:
        return -100.0, -100.0
    n = min(size - 44, want)
    n -= n % frame_bytes
    if n <= 0:
        return -100.0, -100.0
    with open(path, "rb") as f:
        f.seek(size - n)
        data = f.read(n)
    samples = array("h")
    samples.frombytes(data)
    if not samples:
        return -100.0, -100.0
    peak = max(abs(s) for s in samples) / 32768.0
    rms = math.sqrt(sum(s * s for s in samples) / len(samples)) / 32768.0
    to_db = lambda v: 20 * math.log10(v) if v > 1e-6 else -100.0  # noqa: E731
    return to_db(rms), to_db(peak)


# ---- the recorder ------------------------------------------------------------------------

class Recorder:
    """Record / Stop on the monitor of a sink."""

    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.path: str | None = None
        self.sink: str | None = None
        self.rate = 48000
        self.channels = 2
        self.started_at = 0.0

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at if self.running else 0.0

    def start(self, path: str, sink_name: str, rate: int = 48000, channels: int = 2) -> None:
        if self.running:
            raise RuntimeError("a recording is already running")
        cmd = [
            "pw-record",
            "--target", sink_name,
            "-P", "{ stream.capture.sink = true }",
            "--rate", str(rate),
            "--channels", str(channels),
            "--format", "s16",
            path,
        ]
        _log.info("record: sink=%s -> %s", sink_name, path)
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        self.path, self.sink, self.rate, self.channels = path, sink_name, rate, channels
        self.started_at = time.monotonic()
        time.sleep(0.15)
        if self.proc.poll() is not None:
            err = (self.proc.stderr.read() if self.proc.stderr else "").strip()
            self.proc = None
            raise RuntimeError(f"pw-record exited immediately: {err or 'unknown error'}")

    def level(self) -> tuple[float, float]:
        if not self.running or not self.path:
            return -100.0, -100.0
        return tail_level_db(self.path, self.rate, self.channels)

    def stop(self) -> str:
        """Stop cleanly (SIGINT -> SIGTERM -> SIGKILL) and return the WAV path, header repaired if needed."""
        proc, path = self.proc, self.path
        if proc is None or path is None:
            raise RuntimeError("no recording is running")
        duration = time.monotonic() - self.started_at
        for sig, wait in ((signal.SIGINT, 3.0), (signal.SIGTERM, 2.0), (signal.SIGKILL, 2.0)):
            if proc.poll() is not None:
                break
            try:
                proc.send_signal(sig)
                proc.wait(timeout=wait)
            except subprocess.TimeoutExpired:
                continue
            except OSError:
                break
        self.proc = None
        if not wav_is_valid(path):
            fixed = repair_wav_header(path)
            _log.warning("WAV header needed repair: %s", "ok" if fixed else "FAILED")
        _log.info("stop: %.1f s -> %s", duration, path)
        return path
