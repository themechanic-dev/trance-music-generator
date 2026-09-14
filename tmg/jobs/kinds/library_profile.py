"""Rebuild the style profile from the analyses already in the database (fast, no audio work)."""

from __future__ import annotations

from tmg import db as dbmod
from tmg.jobs import protocol
from tmg.library import profile


def run(params: dict) -> dict:
    db = dbmod.Database()
    prof = profile.build_profile(db.analyses())
    db.save_profile(prof)
    db.close()
    protocol.log(profile.describe(prof))
    return {"profile_tracks": prof.get("n_tracks", 0)}
