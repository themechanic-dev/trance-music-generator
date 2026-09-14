"""GUI <-> worker protocol: one JSON line per event on the worker's stdout.

  {"event": "progress", "fraction": 0.42, "message": "..."}
  {"event": "log", "line": "..."}
  {"event": "done", "result": {...}}
  {"event": "error", "message": "...", "traceback": "..."}
"""

from __future__ import annotations

import json
import sys
from typing import Any


def emit(event: str, **fields: Any) -> None:
    sys.stdout.write(json.dumps({"event": event, **fields}, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def progress(fraction: float, message: str = "") -> None:
    emit("progress", fraction=max(0.0, min(1.0, float(fraction))), message=message)


def log(line: str) -> None:
    emit("log", line=line)


def parse_line(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line.startswith("{"):
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) and "event" in obj else None
