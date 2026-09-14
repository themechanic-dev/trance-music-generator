"""Import job: scan a track / a CD / a collection, read tags, add the tracks to the library (queued)."""

from __future__ import annotations

import os

from tmg import db as dbmod
from tmg.jobs import protocol
from tmg.library import scan


def run(params: dict) -> dict:
    kind, root = params["kind"], params["root"]
    protocol.progress(0.0, f"scanning {os.path.basename(root) or root}")
    files = scan.scan(kind, root)
    if not files:
        raise RuntimeError(f"no audio files found in {root}")
    db = dbmod.Database()
    import_id = db.add_import(kind, root)
    added, skipped = 0, 0
    for i, path in enumerate(files):
        st = os.stat(path)
        info = scan.probe(path)
        new_id = db.add_track(path, import_id, info["album"], info["title"], info["artist"], info["duration_s"], st.st_size, st.st_mtime)
        if new_id is None:
            skipped += 1
        else:
            added += 1
        if i % 5 == 0 or i == len(files) - 1:
            protocol.progress((i + 1) / len(files), f"{i + 1}/{len(files)} · {info['title']}")
    db.set_import_count(import_id, added)
    protocol.log(f"import {kind} {root}: {added} new tracks, {skipped} already in the library")
    db.close()
    return {"import_id": import_id, "kind": kind, "root": root, "found": len(files), "added": added, "skipped": skipped}
