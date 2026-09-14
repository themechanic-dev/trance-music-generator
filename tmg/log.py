"""One log for the whole app: rotating file (data/logs/tmg.log) + in-memory ring + subscribers (the GUI)."""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable
from logging.handlers import RotatingFileHandler

from tmg import paths

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_lock = threading.Lock()
_ring: deque[str] = deque(maxlen=3000)
_subscribers: list[Callable[[str], None]] = []
_configured = False


class _RingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        line = self.format(record)
        with _lock:
            _ring.append(line)
            subs = list(_subscribers)
        for cb in subs:
            try:
                cb(line)
            except Exception:  # noqa: BLE001 - a broken subscriber must not break logging
                pass


def setup(level: int = logging.INFO) -> logging.Logger:
    global _configured
    root = logging.getLogger("tmg")
    if _configured:
        return root
    paths.LOGS.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(_FORMAT, _DATEFMT)
    fh = RotatingFileHandler(paths.LOGS / "tmg.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    rh = _RingHandler()
    rh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(rh)
    root.setLevel(level)
    root.propagate = False
    _configured = True
    return root


def get(name: str) -> logging.Logger:
    return logging.getLogger(f"tmg.{name}")


def subscribe(cb: Callable[[str], None]) -> None:
    with _lock:
        _subscribers.append(cb)


def unsubscribe(cb: Callable[[str], None]) -> None:
    with _lock:
        if cb in _subscribers:
            _subscribers.remove(cb)


def recent() -> list[str]:
    with _lock:
        return list(_ring)
