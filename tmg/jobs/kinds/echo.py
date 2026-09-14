"""Test job: returns its own parameters (or fails on purpose). Used by tests and for diagnostics."""

from __future__ import annotations

import time

from tmg.jobs import protocol


def run(params: dict) -> dict:
    steps = int(params.get("steps", 3))
    for i in range(steps):
        protocol.progress((i + 1) / steps, f"step {i + 1}/{steps}")
        time.sleep(float(params.get("sleep", 0.01)))
    if params.get("fail"):
        raise RuntimeError("failed on request")
    protocol.log("echo: done")
    return {"echo": params}
