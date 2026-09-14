"""Compose job: profile -> plan -> audio + timeline -> files -> a 'productions' row."""

from __future__ import annotations

import random
import time

import numpy as np

from tmg import db as dbmod
from tmg.jobs import protocol
from tmg.music import composer, neural, phrases_mix, render
from tmg.music import plan as planmod


def _pick_bank(db: dbmod.Database, opts: dict) -> list[dict]:
    """Which phrases may be used: chosen ids, or everything ready from the requested source."""
    ids = opts.get("ids") or []
    if opts.get("source") == "selected" and ids:
        return db.get_phrases_by_ids([str(i) for i in ids])
    source = {"captured": "capture", "library": "library"}.get(opts.get("source") or "captured")
    rows = [p for p in db.list_phrases(source=source, limit=5000) if p.get("status") == "ready"]
    if not rows and source == "capture":
        rows = [p for p in db.list_phrases(limit=5000) if p.get("status") == "ready"]
    return rows


def run(params: dict) -> dict:
    seed = params.get("seed")
    seed = int(seed) if seed not in (None, "", 0) else random.randrange(1, 10**8)
    minutes = params.get("minutes")
    minutes = float(minutes) if minutes else None
    formats = tuple(params.get("formats") or ("mp3",))
    bitrate = int(params.get("bitrate", 192))
    flavor = params.get("flavor") or None

    db = dbmod.Database()
    profile = db.get_profile()
    protocol.progress(0.02, "planning from the profile" if profile and profile.get("n_tracks") else "planning (library empty - built-in profile)")
    spec = planmod.build_plan(profile, seed, minutes=minutes, flavor=flavor)
    protocol.log(f"plan: seed {seed} · {spec.bpm} BPM · {spec.key_name} · {spec.flavor.name} · {spec.duration_s / 60:.1f} min · {spec.structure}")

    t0 = time.perf_counter()
    audio, timeline = composer.render(spec, progress=protocol.progress)
    render_s = time.perf_counter() - t0
    tl = timeline.to_dict()

    # phrases from the bank, placed where the timeline says (phase 4)
    placed: list[dict] = []
    popts = params.get("phrases") or {}
    if popts.get("enabled") and int(popts.get("count", 0)) > 0:
        protocol.progress(0.86, "placing phrases")
        bank = _pick_bank(db, popts)
        if bank:
            rng = np.random.default_rng(seed + 7)
            placements = phrases_mix.choose_placements(
                tl, bank, count=int(popts.get("count", 2)), where=popts.get("where", "both"),
                allow_intro=bool(popts.get("intro", False)), rng=rng,
            )
            audio, placed = phrases_mix.place(
                audio, tl, placements, level_db=float(popts.get("level_db", -6.0)), telephone=bool(popts.get("telephone", False)),
                echo_repeats=int(popts.get("echo", 2)), fit_to_beat=bool(popts.get("fit", True)), rng=rng,
            )
            for d in placed:
                protocol.log(f"phrase '{d['name']}' -> {d['slot']} at {d['start_s']:.1f} s (stretch {d['stretch']})")
        else:
            protocol.log("no phrases in the bank match the selection - track left without phrases")
    tl["phrases"] = placed

    # neural textures under the arrangement (phase 6), optionally keeping a dry copy for A/B
    nopts = params.get("neural") or {}
    neural_info: dict | None = None
    dry_files = None
    if nopts.get("enabled"):
        if nopts.get("ab"):
            protocol.progress(0.88, "writing the dry copy (A/B)")
            dry_files = render.write_outputs(audio, spec, tl, formats=formats, bitrate_k=bitrate, suffix="-dry")
        try:
            audio, neural_info = neural.apply(
                audio, tl, spec, model_key=nopts.get("model", "stereo-small"), level_db=float(nopts.get("level_db", -10.0)),
                energy_too=bool(nopts.get("energy", True)), seed=seed, log=protocol.log,
                progress=lambda f, m: protocol.progress(0.88 + 0.08 * f, m),
            )
        except Exception as exc:  # noqa: BLE001 - the track is still a track without the neural layer
            protocol.log(f"neural layer failed ({type(exc).__name__}: {exc}) - track written without it")
            neural_info = {"error": f"{type(exc).__name__}: {exc}"}
    tl["neural"] = neural_info

    protocol.progress(0.96, "writing files")
    files = render.write_outputs(audio, spec, tl, formats=formats, bitrate_k=bitrate)
    if dry_files:
        files["dry_mp3"] = dry_files.get("mp3")
        files["dry_wav"] = dry_files.get("wav")
    production_id = db.add_production(
        seed=seed, title=files["title"], bpm=spec.bpm, key_name=spec.key_name, duration_s=spec.duration_s,
        flavor=spec.flavor.name, structure=spec.structure, mp3_path=files["mp3"], wav_path=files["wav"],
        timeline_path=files["timeline"], plan=spec.to_dict(), profile_tracks=spec.profile_tracks, phrases=placed,
        neural={**(neural_info or {}), "dry_mp3": files.get("dry_mp3"), "dry_wav": files.get("dry_wav")} if nopts.get("enabled") else None,
    )
    db.close()
    protocol.log(f"rendered in {render_s:.0f} s ({spec.duration_s / max(render_s, 0.01):.1f}x realtime)")
    return {"production_id": production_id, "seed": seed, "title": files["title"], "bpm": spec.bpm, "key": spec.key_name,
            "duration_s": round(spec.duration_s, 1), "structure": spec.structure, "flavor": spec.flavor.name,
            "mp3": files["mp3"], "wav": files["wav"], "render_s": round(render_s, 1),
            "phrases": [f"{d['name']} @ {d['start_s']:.0f}s" for d in placed],
            "neural": (neural_info or {}).get("model") if neural_info and "error" not in neural_info else None,
            "dry_mp3": files.get("dry_mp3")}
