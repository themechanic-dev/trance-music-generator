"""Re-extract the library phrases with the current rule (the speech score), track by track.

For every analysed track: decode, Demucs, vocal_segments with the track's tempo, then replace the track's
library phrases in the bank - the old files go to data/trash, nothing is lost silently - and store the new
segments in the track's analysis. Resumable: a track whose phrases already come from these settings is
skipped, so a crash or a Pause picks up where it stopped. Force=True redoes every track.
"""

from __future__ import annotations

import json
import time

from tmg import db as dbmod
from tmg.capture import phrases as phrase_files
from tmg.jobs import protocol
from tmg.jobs.kinds.library_analyze import _save_vocal_phrases
from tmg.library import analysis

DEFAULT_RULE = {"version": analysis.ANALYSIS_VERSION, "threshold_db": -40.0, "min_len": 1.2, "max_len": 8.0, "max_count": 3,
                "min_score": analysis.SPEECH_MIN_SCORE}


def rule_of(vocal_opts: dict) -> dict:
    """The settings that decide which phrases a track gets; stored with the track so a change reruns it."""
    return {"version": analysis.ANALYSIS_VERSION, **{k: vocal_opts[k] for k in ("threshold_db", "min_len", "max_len", "max_count", "min_score")}}


def needs_redo(track_analysis: dict, rule: dict) -> bool:
    """A track is done when its phrases came from exactly these settings. Tracks re-extracted before the rule was
    recorded (analysis version 2 without a 'phrase_rule') count as done with the default settings."""
    stored = track_analysis.get("phrase_rule")
    if stored is None and int(track_analysis.get("version", 1)) >= analysis.ANALYSIS_VERSION:
        stored = DEFAULT_RULE
    return stored != rule


def _analysis_of(track: dict) -> dict:
    a = track.get("analysis")
    if isinstance(a, str):
        try:
            return json.loads(a)
        except json.JSONDecodeError:
            return {}
    return dict(a or {})


def run(params: dict) -> dict:
    vocal_opts = {
        "threshold_db": float(params.get("vocal_threshold_db", -40.0)),
        "min_len": float(params.get("vocal_min_len", 1.2)),
        "max_len": float(params.get("vocal_max_len", 8.0)),
        "max_count": int(params.get("vocal_max_count", 3)),
        "min_score": float(params.get("min_score", analysis.SPEECH_MIN_SCORE)),
    }
    force = bool(params.get("force", False))
    only = {int(i) for i in (params.get("track_ids") or [])}
    rule = rule_of(vocal_opts)

    db = dbmod.Database()
    tracks = db.tracks_done()
    todo = []
    for t in tracks:
        if only and t["id"] not in only:
            continue
        if not force and not needs_redo(_analysis_of(t), rule):
            continue
        todo.append(t)
    if not todo:
        db.close()
        protocol.log("every analysed track already has phrases from these settings - change the minimum speech score, the "
                     "threshold or the count in Settings > Library analysis to pick them again")
        return {"tracks": 0, "phrases": 0, "removed": 0, "minutes": 0.0}

    protocol.progress(0.0, "loading Demucs")
    model = analysis.load_demucs("htdemucs")
    started = time.perf_counter()
    saved_total = removed_total = failed = 0
    for i, t in enumerate(todo):
        title = t.get("title") or t["path"]
        elapsed = time.perf_counter() - started
        eta = (elapsed / i * (len(todo) - i) / 60.0) if i else 0.0
        protocol.progress(i / len(todo), f"{i}/{len(todo)} · {title}" + (f" · ~{eta:.0f} min left" if i else ""))
        try:
            wav = analysis.load_audio(t["path"], analysis.SR)
            stems = analysis.separate(model, wav)
            vocals22 = analysis.to_mono_22k(stems["vocals"])
            a = _analysis_of(t)
            bpm = float(t.get("bpm") or a.get("bpm") or 0.0) or None
            segs = analysis.vocal_segments(vocals22, bpm=bpm, **vocal_opts)
            # out with the old picks of this track (files to the trash), in with the new
            old_files: list[str] = []
            for p in db.phrases_for_track(t["id"]):
                if p.get("source") != "library":
                    continue
                old_files += [x for x in (p.get("path"), p.get("raw_path")) if x]
                db.delete_phrase(p["id"])
            if old_files:
                phrase_files.move_to_trash(*old_files)
            removed_total += len(old_files)
            saved = _save_vocal_phrases(db, t, stems, segs, vocal_opts["max_count"]) if segs else 0
            saved_total += saved
            a["vocal_segments"] = segs
            a["version"] = analysis.ANALYSIS_VERSION
            a["phrase_rule"] = rule
            db.update_track(t["id"], analysis=a)
            protocol.emit("track_done", track_id=t["id"], phrases=saved)
            if saved:
                protocol.log(f"{title}: {saved} phrase(s), best score {segs[0]['score']:.2f}")
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the rest
            failed += 1
            protocol.log(f"FAILED {title}: {type(exc).__name__}: {exc}")
        if (i + 1) % 10 == 0:
            import torch

            torch.cuda.empty_cache()
    minutes = round((time.perf_counter() - started) / 60.0, 1)
    db.close()
    protocol.log(f"phrases re-extracted: {saved_total} new from {len(todo)} tracks, {removed_total} old files moved to the trash, "
                 f"{failed} failed, {minutes} min")
    protocol.progress(1.0, "done")
    return {"tracks": len(todo), "phrases": saved_total, "removed": removed_total, "failed": failed, "minutes": minutes}
