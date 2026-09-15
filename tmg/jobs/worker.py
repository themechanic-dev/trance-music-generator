"""The worker: runs INSIDE the .venv (Python 3.12) as a subprocess, one job per process.

    .venv/bin/python -m tmg.jobs.worker --job '{"id": 1, "kind": "env_check", "params": {}}'

Every job kind is a module in tmg/jobs/kinds/ exposing run(params) -> dict.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import traceback

from tmg.jobs import protocol

KINDS = {
    "echo": "tmg.jobs.kinds.echo",
    "env_check": "tmg.jobs.kinds.env_check",
    "phrase_postprocess": "tmg.jobs.kinds.phrase_postprocess",
    "library_import": "tmg.jobs.kinds.library_import",
    "library_analyze": "tmg.jobs.kinds.library_analyze",
    "library_profile": "tmg.jobs.kinds.library_profile",
    "compose": "tmg.jobs.kinds.compose",
    "render_video": "tmg.jobs.kinds.render_video",
    "hf_check": "tmg.jobs.kinds.hf_check",
    "model_test": "tmg.jobs.kinds.model_test",
    "library_phrases": "tmg.jobs.kinds.library_phrases",
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, help="JSON: {id, kind, params}")
    args = ap.parse_args(argv)
    try:
        job = json.loads(args.job)
        kind = job["kind"]
        params = job.get("params") or {}
        module_name = KINDS[kind]
    except (json.JSONDecodeError, KeyError) as exc:
        protocol.emit("error", message=f"invalid job: {exc}")
        return 2
    try:
        mod = importlib.import_module(module_name)
        result = mod.run(params)
        protocol.emit("done", result=result or {})
        return 0
    except Exception as exc:  # noqa: BLE001 - whatever breaks, the GUI must hear about it
        protocol.emit("error", message=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
