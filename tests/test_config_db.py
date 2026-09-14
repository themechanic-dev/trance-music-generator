import json

from tmg import config
from tmg import db as dbmod


def test_settings_defaults_and_roundtrip(tmp_path):
    path = tmp_path / "settings.json"
    s = config.Settings(path)
    assert s.get("capture.sink") == "default"
    assert s.get("nope.missing", 7) == 7
    s.set("capture.sink", "alsa_output.x")
    s.set("ui.window_width", 900)
    s2 = config.Settings(path)
    assert s2.get("capture.sink") == "alsa_output.x" and s2.get("ui.window_width") == 900
    assert s2.get("capture.auto_trim") is True  # defaults merged in


def test_settings_broken_file(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{not json", encoding="utf-8")
    s = config.Settings(path)
    assert s.get("capture.sink") == "default"
    assert (tmp_path / "settings.json.broken").exists()


def test_db_phrases_and_jobs(tmp_path):
    db = dbmod.Database(tmp_path / "t.sqlite")
    db.add_phrase("p1", "one", "/x/p1.wav")
    db.add_phrase("p2", "two", "/x/p2.wav", status="ready")
    db.update_phrase("p1", duration_s=2.5, status="ready", peak_db=-1.0)
    rows = db.list_phrases()
    assert {r["id"] for r in rows} == {"p1", "p2"}
    assert db.get_phrase("p1")["duration_s"] == 2.5
    db.delete_phrase("p2")
    assert db.get_phrase("p2") is None

    jid = db.add_job("echo", {"a": 1})
    db.update_job(jid, status="running", progress=0.5)
    db.update_job(jid, status="done", result={"ok": True})
    job = db.get_job(jid)
    assert job["params"] == {"a": 1} and job["result"] == {"ok": True} and job["progress"] == 0.5
    j2 = db.add_job("echo", {})
    assert db.reset_unfinished_jobs() == 1
    assert db.get_job(j2)["status"] == "failed"
    assert dbmod.to_local("2026-09-13T20:00:00Z", "%Y") == "2026"
    db.close()


def test_job_params_are_json(tmp_path):
    db = dbmod.Database(tmp_path / "t.sqlite")
    jid = db.add_job("x", {"nested": {"k": [1, 2]}})
    raw = db._rows("SELECT params FROM jobs WHERE id = ?", (jid,))[0]["params"]
    assert json.loads(raw) == {"nested": {"k": [1, 2]}}
    db.close()
