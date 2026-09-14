"""Job queue: one job at a time (the VRAM cannot hold two models), each in a .venv subprocess.

The window never freezes: the queue lives in its own thread, events arrive through a callback and
the GUI hands them to the main loop with GLib.idle_add. Standard library only.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
from collections.abc import Callable
from typing import Any

from tmg import db as dbmod
from tmg import log, paths
from tmg.jobs import protocol

_log = log.get("jobs")

EventCallback = Callable[[int, str, dict[str, Any], dict[str, Any]], None]  # (job_id, kind, params, event)


class JobQueue(threading.Thread):
    def __init__(self, database: dbmod.Database, worker_python: str | None = None, on_event: EventCallback | None = None):
        super().__init__(name="tmg-jobs", daemon=True)
        self.db = database
        self.worker_python = worker_python or str(paths.VENV_PYTHON)
        self.on_event = on_event
        self._q: queue.Queue[int | None] = queue.Queue()
        self._cancelled: set[int] = set()
        self._current: subprocess.Popen | None = None
        self._current_id: int | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()

    # ---- API for the GUI ------------------------------------------------------
    def submit(self, kind: str, params: dict[str, Any] | None = None) -> int:
        job_id = self.db.add_job(kind, params or {})
        _log.info("job #%d %s queued", job_id, kind)
        self._q.put(job_id)
        self._emit(job_id, kind, params or {}, {"event": "queued"})
        return job_id

    def cancel(self, job_id: int) -> None:
        with self._lock:
            if self._current_id == job_id and self._current is not None and self._current.poll() is None:
                self._current.terminate()
                _log.info("job #%d: cancel requested (terminate)", job_id)
                return
        self._cancelled.add(job_id)
        self.db.update_job(job_id, status="cancelled", message="cancelled", finished_utc=dbmod.utc_now())
        job = self.db.get_job(job_id)
        if job:
            self._emit(job_id, job["kind"], job["params"], {"event": "finished", "status": "cancelled", "message": "cancelled", "result": {}})

    def shutdown(self) -> None:
        self._stop.set()
        with self._lock:
            if self._current is not None and self._current.poll() is None:
                self._current.terminate()
        self._q.put(None)

    @property
    def busy(self) -> bool:
        return self._current_id is not None

    # ---- the loop -------------------------------------------------------------
    def run(self) -> None:
        while not self._stop.is_set():
            job_id = self._q.get()
            if job_id is None:
                break
            if job_id in self._cancelled:
                continue
            job = self.db.get_job(job_id)
            if not job:
                continue
            self._run_job(job)

    def _run_job(self, job: dict[str, Any]) -> None:
        job_id, kind, params = job["id"], job["kind"], job["params"]
        self.db.update_job(job_id, status="running", started_utc=dbmod.utc_now(), message="started")
        self._emit(job_id, kind, params, {"event": "started"})

        if not os.path.exists(self.worker_python):
            self._finish(job_id, kind, params, "failed", f"worker Python not found: {self.worker_python} (is the .venv missing?)")
            return

        payload = json.dumps({"id": job_id, "kind": kind, "params": params}, ensure_ascii=False)
        cmd = [self.worker_python, "-m", "tmg.jobs.worker", "--job", payload]
        try:
            proc = subprocess.Popen(
                cmd, cwd=str(paths.ROOT), env=paths.worker_env(),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
            )
        except OSError as exc:
            self._finish(job_id, kind, params, "failed", f"could not start the worker: {exc}")
            return
        with self._lock:
            self._current, self._current_id = proc, job_id

        stderr_lines: list[str] = []
        t_err = threading.Thread(target=lambda: stderr_lines.extend(proc.stderr.readlines()), daemon=True)  # type: ignore[union-attr]
        t_err.start()

        result: dict[str, Any] | None = None
        error: str | None = None
        assert proc.stdout is not None
        for line in proc.stdout:
            ev = protocol.parse_line(line)
            if ev is None:
                if line.strip():
                    _log.info("#%d %s | %s", job_id, kind, line.rstrip())
                continue
            name = ev.get("event")
            if name == "progress":
                self.db.update_job(job_id, progress=ev.get("fraction", 0.0), message=ev.get("message", ""))
            elif name == "log":
                _log.info("#%d %s | %s", job_id, kind, ev.get("line", ""))
            elif name == "done":
                result = ev.get("result") or {}
            elif name == "error":
                error = ev.get("message", "error")
                if ev.get("traceback"):
                    _log.error("#%d %s traceback:\n%s", job_id, kind, ev["traceback"])
            self._emit(job_id, kind, params, ev)
        rc = proc.wait()
        t_err.join(timeout=2)
        with self._lock:
            self._current, self._current_id = None, None

        if error is None and rc != 0:
            tail = "".join(stderr_lines[-15:]).strip()
            error = f"worker exited with code {rc}" + (f":\n{tail}" if tail else "")
        if rc < 0 or job_id in self._cancelled:
            self._finish(job_id, kind, params, "cancelled", "cancelled")
        elif error is not None:
            self._finish(job_id, kind, params, "failed", error)
        else:
            self._finish(job_id, kind, params, "done", "finished", result)

    def _finish(self, job_id: int, kind: str, params: dict[str, Any], status: str, message: str, result: dict[str, Any] | None = None) -> None:
        fields: dict[str, Any] = {"status": status, "message": message, "finished_utc": dbmod.utc_now()}
        if status == "done":
            fields["progress"] = 1.0
            fields["result"] = result or {}
        self.db.update_job(job_id, **fields)
        first = message.splitlines()[0] if message else ""
        (_log.info if status == "done" else _log.warning)("job #%d %s: %s - %s", job_id, kind, status, first)
        self._emit(job_id, kind, params, {"event": "finished", "status": status, "message": message, "result": result or {}})

    def _emit(self, job_id: int, kind: str, params: dict[str, Any], event: dict[str, Any]) -> None:
        if self.on_event:
            try:
                self.on_event(job_id, kind, params, event)
            except Exception:  # noqa: BLE001
                _log.exception("error in job event callback")
