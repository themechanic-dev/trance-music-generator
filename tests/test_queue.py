import sys
import threading
import time

from tmg import db as dbmod
from tmg.jobs.queue import JobQueue


def _run(db, kind, params, timeout=30):
    events = []
    finished = threading.Event()

    def on_event(job_id, k, p, ev):
        events.append(ev)
        if ev.get("event") == "finished":
            finished.set()

    q = JobQueue(db, worker_python=sys.executable, on_event=on_event)
    q.start()
    jid = q.submit(kind, params)
    assert finished.wait(timeout), "job did not finish"
    q.shutdown()
    return jid, events


def test_echo_job_succeeds(tmp_path):
    db = dbmod.Database(tmp_path / "q.sqlite")
    jid, events = _run(db, "echo", {"steps": 2, "hello": "world"})
    job = db.get_job(jid)
    assert job["status"] == "done"
    assert job["result"] == {"echo": {"steps": 2, "hello": "world"}}
    assert job["progress"] == 1.0
    kinds = [e["event"] for e in events]
    assert kinds[0] == "queued" and "progress" in kinds and kinds[-1] == "finished"
    db.close()


def test_echo_job_fails(tmp_path):
    db = dbmod.Database(tmp_path / "q.sqlite")
    jid, events = _run(db, "echo", {"fail": True})
    job = db.get_job(jid)
    assert job["status"] == "failed"
    assert "RuntimeError" in job["message"]
    db.close()


def test_unknown_kind(tmp_path):
    db = dbmod.Database(tmp_path / "q.sqlite")
    jid, _ = _run(db, "no_such_kind", {})
    assert db.get_job(jid)["status"] == "failed"
    db.close()


def test_missing_worker(tmp_path):
    db = dbmod.Database(tmp_path / "q.sqlite")
    finished = threading.Event()
    q = JobQueue(db, worker_python=str(tmp_path / "nope"), on_event=lambda *a: finished.set() if a[3].get("event") == "finished" else None)
    q.start()
    jid = q.submit("echo", {})
    assert finished.wait(10)
    assert "not found" in db.get_job(jid)["message"]
    q.shutdown()
    db.close()


def test_cancel_running(tmp_path):
    db = dbmod.Database(tmp_path / "q.sqlite")
    finished = threading.Event()
    started = threading.Event()

    def on_event(job_id, k, p, ev):
        if ev.get("event") == "started":
            started.set()
        if ev.get("event") == "finished":
            finished.set()

    q = JobQueue(db, worker_python=sys.executable, on_event=on_event)
    q.start()
    jid = q.submit("echo", {"steps": 50, "sleep": 0.2})
    assert started.wait(10)
    time.sleep(0.5)
    q.cancel(jid)
    assert finished.wait(15)
    assert db.get_job(jid)["status"] == "cancelled"
    q.shutdown()
    db.close()
