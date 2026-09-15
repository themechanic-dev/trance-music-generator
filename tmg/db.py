"""SQLite inside the project folder: phrases + jobs (later: library, productions).

Lesson from AutoDJ: SQLite loses the timezone, so everything is stored as UTC ISO ("...Z") and
converted to local time only for display.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any

from tmg import paths

SCHEMA = """
CREATE TABLE IF NOT EXISTS phrases (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    path        TEXT NOT NULL,
    raw_path    TEXT,
    duration_s  REAL,
    sample_rate INTEGER,
    channels    INTEGER,
    source      TEXT NOT NULL DEFAULT 'capture',
    status      TEXT NOT NULL DEFAULT 'recording',
    peak_db     REAL,
    created_utc TEXT NOT NULL,
    notes       TEXT
);
CREATE TABLE IF NOT EXISTS library_imports (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL,            -- track | cd | collection
    root        TEXT NOT NULL,
    n_tracks    INTEGER NOT NULL DEFAULT 0,
    added_utc   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS library_tracks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    path          TEXT NOT NULL UNIQUE,
    import_id     INTEGER,
    album         TEXT,                   -- the folder name (one CD = one folder)
    title         TEXT,
    artist        TEXT,
    duration_s    REAL,
    size_bytes    INTEGER,
    mtime         REAL,
    status        TEXT NOT NULL DEFAULT 'queued',   -- queued | analyzing | done | failed
    error         TEXT,
    bpm           REAL,
    key_name      TEXT,
    n_bars        INTEGER,
    analysis      TEXT,                   -- JSON, the full per-track analysis
    stems_dir     TEXT,
    added_utc     TEXT NOT NULL,
    analyzed_utc  TEXT
);
CREATE INDEX IF NOT EXISTS idx_library_status ON library_tracks(status);
CREATE TABLE IF NOT EXISTS profiles (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL DEFAULT 'current',
    n_tracks    INTEGER NOT NULL,
    json        TEXT NOT NULL,
    built_utc   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS productions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    seed           INTEGER NOT NULL,
    title          TEXT NOT NULL,
    bpm            REAL,
    key_name       TEXT,
    duration_s     REAL,
    flavor         TEXT,
    structure      TEXT,
    mp3_path       TEXT,
    wav_path       TEXT,
    timeline_path  TEXT,
    plan           TEXT,
    profile_tracks INTEGER,
    created_utc    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    kind         TEXT NOT NULL,
    params       TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'queued',
    progress     REAL NOT NULL DEFAULT 0,
    message      TEXT,
    result       TEXT,
    created_utc  TEXT NOT NULL,
    started_utc  TEXT,
    finished_utc TEXT
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def to_local(iso_utc: str | None, fmt: str = "%d/%m/%Y %H:%M") -> str:
    if not iso_utc:
        return ""
    try:
        dt = datetime.strptime(iso_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return iso_utc
    return dt.astimezone().strftime(fmt)


class Database:
    def __init__(self, path=None):
        self.path = str(path or paths.DB_FILE)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=15000")   # the worker process writes too
            self._conn.executescript(SCHEMA)
            self._migrate()
            self._conn.commit()

    def _migrate(self) -> None:
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(phrases)")}
        if "track_id" not in cols:
            self._conn.execute("ALTER TABLE phrases ADD COLUMN track_id INTEGER")
        pcols = {r[1] for r in self._conn.execute("PRAGMA table_info(productions)")}
        if "phrases" not in pcols:
            self._conn.execute("ALTER TABLE productions ADD COLUMN phrases TEXT")
        if "mp4_path" not in pcols:
            self._conn.execute("ALTER TABLE productions ADD COLUMN mp4_path TEXT")
            self._conn.execute("ALTER TABLE productions ADD COLUMN video TEXT")
        if "neural" not in pcols:
            self._conn.execute("ALTER TABLE productions ADD COLUMN neural TEXT")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---- generic ------------------------------------------------------------
    def _rows(self, sql: str, args: tuple = ()) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(sql, args)
            return [dict(r) for r in cur.fetchall()]

    def _exec(self, sql: str, args: tuple = ()) -> int:
        with self._lock:
            cur = self._conn.execute(sql, args)
            self._conn.commit()
            return cur.lastrowid or 0

    # ---- phrases ------------------------------------------------------------
    def add_phrase(self, id: str, name: str, path: str, source: str = "capture", status: str = "recording") -> None:
        self._exec(
            "INSERT INTO phrases (id, name, path, source, status, created_utc) VALUES (?,?,?,?,?,?)",
            (id, name, path, source, status, utc_now()),
        )

    def update_phrase(self, id: str, **fields: Any) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k} = ?" for k in fields)
        self._exec(f"UPDATE phrases SET {cols} WHERE id = ?", (*fields.values(), id))

    def delete_phrase(self, id: str) -> None:
        self._exec("DELETE FROM phrases WHERE id = ?", (id,))

    def get_phrase(self, id: str) -> dict[str, Any] | None:
        rows = self._rows("SELECT * FROM phrases WHERE id = ?", (id,))
        return rows[0] if rows else None

    def list_phrases(self, source: str | None = None, search: str | None = None, limit: int = 2000) -> list[dict[str, Any]]:
        sql, args = "SELECT * FROM phrases", []
        where = []
        if source:
            where.append("source = ?")
            args.append(source)
        if search:
            where.append("name LIKE ?")
            args.append(f"%{search}%")
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_utc DESC, id DESC LIMIT ?"
        args.append(limit)
        return self._rows(sql, tuple(args))

    def add_phrase_full(self, id: str, name: str, path: str, source: str, status: str, duration_s: float,
                        sample_rate: int, channels: int, peak_db: float | None, track_id: int | None) -> None:
        self._exec(
            "INSERT INTO phrases (id, name, path, source, status, duration_s, sample_rate, channels, peak_db, track_id, created_utc)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (id, name, path, source, status, duration_s, sample_rate, channels, peak_db, track_id, utc_now()),
        )

    def get_phrases_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        return self._rows(f"SELECT * FROM phrases WHERE id IN ({marks}) AND status = 'ready'", tuple(ids))

    def count_phrases_for_track(self, track_id: int) -> int:
        return self._rows("SELECT count(*) AS n FROM phrases WHERE track_id = ?", (track_id,))[0]["n"]

    def count_phrases(self, source: str | None = None, status: str | None = None, search: str | None = None) -> int:
        sql, args = "SELECT count(*) AS n FROM phrases", []
        where = []
        if source:
            where.append("source = ?")
            args.append(source)
        if status:
            where.append("status = ?")
            args.append(status)
        if search:
            where.append("name LIKE ?")
            args.append(f"%{search}%")
        if where:
            sql += " WHERE " + " AND ".join(where)
        return self._rows(sql, tuple(args))[0]["n"]

    def phrases_for_track(self, track_id: int) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM phrases WHERE track_id = ? ORDER BY id", (track_id,))

    def tracks_done(self) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM library_tracks WHERE status = 'done' ORDER BY id")

    # ---- library ------------------------------------------------------------
    def add_import(self, kind: str, root: str) -> int:
        return self._exec("INSERT INTO library_imports (kind, root, added_utc) VALUES (?,?,?)", (kind, root, utc_now()))

    def set_import_count(self, import_id: int, n: int) -> None:
        self._exec("UPDATE library_imports SET n_tracks = ? WHERE id = ?", (n, import_id))

    def add_track(self, path: str, import_id: int, album: str, title: str, artist: str,
                  duration_s: float | None, size_bytes: int, mtime: float) -> int | None:
        """Insert one track; returns the new id, or None when the path is already in the library."""
        with self._lock:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO library_tracks (path, import_id, album, title, artist, duration_s, size_bytes, mtime, added_utc)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (path, import_id, album, title, artist, duration_s, size_bytes, mtime, utc_now()),
            )
            self._conn.commit()
            return cur.lastrowid if cur.rowcount else None

    def get_track(self, id: int) -> dict[str, Any] | None:
        rows = self._rows("SELECT * FROM library_tracks WHERE id = ?", (id,))
        return self._decode_track(rows[0]) if rows else None

    def update_track(self, id: int, **fields: Any) -> None:
        if "analysis" in fields and not isinstance(fields["analysis"], (str, type(None))):
            fields["analysis"] = json.dumps(fields["analysis"], ensure_ascii=False)
        cols = ", ".join(f"{k} = ?" for k in fields)
        self._exec(f"UPDATE library_tracks SET {cols} WHERE id = ?", (*fields.values(), id))

    def delete_track(self, id: int) -> None:
        self._exec("DELETE FROM library_tracks WHERE id = ?", (id,))

    def list_tracks(self, status: str | None = None, search: str | None = None, limit: int = 300,
                    with_analysis: bool = False) -> list[dict[str, Any]]:
        cols = "*" if with_analysis else ("id, path, import_id, album, title, artist, duration_s, status, error, bpm, key_name, "
                                           "n_bars, stems_dir, added_utc, analyzed_utc")
        sql, args = f"SELECT {cols} FROM library_tracks", []
        where = []
        if status:
            where.append("status = ?")
            args.append(status)
        if search:
            where.append("(title LIKE ? OR album LIKE ? OR artist LIKE ? OR path LIKE ?)")
            args.extend([f"%{search}%"] * 4)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY album, path LIMIT ?"
        args.append(limit)
        return [self._decode_track(r) for r in self._rows(sql, tuple(args))]

    def tracks_to_analyze(self) -> list[dict[str, Any]]:
        return self._rows("SELECT id, path, title, album FROM library_tracks WHERE status IN ('queued','analyzing') ORDER BY album, path")

    def analyses(self) -> list[dict[str, Any]]:
        out = []
        for r in self._rows("SELECT id, analysis FROM library_tracks WHERE status = 'done' AND analysis IS NOT NULL"):
            try:
                a = json.loads(r["analysis"])
                a["track_id"] = r["id"]
                out.append(a)
            except json.JSONDecodeError:
                continue
        return out

    def track_counts(self) -> dict[str, int]:
        counts = {"queued": 0, "analyzing": 0, "done": 0, "failed": 0, "total": 0}
        for r in self._rows("SELECT status, count(*) AS n FROM library_tracks GROUP BY status"):
            counts[r["status"]] = r["n"]
            counts["total"] += r["n"]
        return counts

    def requeue_failed(self) -> int:
        with self._lock:
            cur = self._conn.execute("UPDATE library_tracks SET status='queued', error=NULL WHERE status='failed'")
            self._conn.commit()
            return cur.rowcount

    def save_profile(self, profile: dict[str, Any], name: str = "current") -> int:
        with self._lock:
            self._conn.execute("DELETE FROM profiles WHERE name = ?", (name,))
            cur = self._conn.execute(
                "INSERT INTO profiles (name, n_tracks, json, built_utc) VALUES (?,?,?,?)",
                (name, int(profile.get("n_tracks", 0)), json.dumps(profile, ensure_ascii=False), utc_now()),
            )
            self._conn.commit()
            return cur.lastrowid or 0

    def get_profile(self, name: str = "current") -> dict[str, Any] | None:
        rows = self._rows("SELECT * FROM profiles WHERE name = ? ORDER BY id DESC LIMIT 1", (name,))
        if not rows:
            return None
        try:
            p = json.loads(rows[0]["json"])
        except json.JSONDecodeError:
            return None
        p["built_utc"] = rows[0]["built_utc"]
        return p

    # ---- productions --------------------------------------------------------
    def add_production(self, **f: Any) -> int:
        plan = f.get("plan")
        phrases = f.get("phrases")
        neural = f.get("neural")
        return self._exec(
            "INSERT INTO productions (seed, title, bpm, key_name, duration_s, flavor, structure, mp3_path, wav_path,"
            " timeline_path, plan, profile_tracks, phrases, neural, created_utc) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (f["seed"], f["title"], f.get("bpm"), f.get("key_name"), f.get("duration_s"), f.get("flavor"), f.get("structure"),
             f.get("mp3_path"), f.get("wav_path"), f.get("timeline_path"),
             json.dumps(plan, ensure_ascii=False) if plan is not None else None, f.get("profile_tracks"),
             json.dumps(phrases, ensure_ascii=False) if phrases is not None else None,
             json.dumps(neural, ensure_ascii=False) if neural is not None else None, utc_now()),
        )

    def list_productions(self, limit: int = 200) -> list[dict[str, Any]]:
        rows = self._rows("SELECT id, seed, title, bpm, key_name, duration_s, flavor, structure, mp3_path, wav_path, timeline_path,"
                          " profile_tracks, phrases, mp4_path, video, neural, created_utc FROM productions ORDER BY id DESC LIMIT ?", (limit,))
        for r in rows:
            for key in ("phrases", "video", "neural"):
                if isinstance(r.get(key), str):
                    try:
                        r[key] = json.loads(r[key])
                    except json.JSONDecodeError:
                        r[key] = None
        return rows

    def update_production_video(self, id: int, mp4_path: str | None, video: dict[str, Any] | None) -> None:
        self._exec("UPDATE productions SET mp4_path = ?, video = ? WHERE id = ?",
                   (mp4_path, json.dumps(video, ensure_ascii=False) if video is not None else None, id))

    def get_production(self, id: int) -> dict[str, Any] | None:
        rows = self._rows("SELECT * FROM productions WHERE id = ?", (id,))
        if not rows:
            return None
        row = rows[0]
        if isinstance(row.get("plan"), str):
            try:
                row["plan"] = json.loads(row["plan"])
            except json.JSONDecodeError:
                pass
        return row

    def delete_production(self, id: int) -> None:
        self._exec("DELETE FROM productions WHERE id = ?", (id,))

    @staticmethod
    def _decode_track(row: dict[str, Any]) -> dict[str, Any]:
        val = row.get("analysis")
        if isinstance(val, str):
            try:
                row["analysis"] = json.loads(val)
            except json.JSONDecodeError:
                pass
        return row

    # ---- jobs ---------------------------------------------------------------
    def add_job(self, kind: str, params: dict[str, Any]) -> int:
        return self._exec(
            "INSERT INTO jobs (kind, params, status, created_utc) VALUES (?,?,?,?)",
            (kind, json.dumps(params, ensure_ascii=False), "queued", utc_now()),
        )

    def update_job(self, id: int, **fields: Any) -> None:
        if "result" in fields and not isinstance(fields["result"], (str, type(None))):
            fields["result"] = json.dumps(fields["result"], ensure_ascii=False)
        cols = ", ".join(f"{k} = ?" for k in fields)
        self._exec(f"UPDATE jobs SET {cols} WHERE id = ?", (*fields.values(), id))

    def get_job(self, id: int) -> dict[str, Any] | None:
        rows = self._rows("SELECT * FROM jobs WHERE id = ?", (id,))
        return self._decode_job(rows[0]) if rows else None

    def list_jobs(self, limit: int = 200) -> list[dict[str, Any]]:
        return [self._decode_job(r) for r in self._rows("SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,))]

    def reset_unfinished_jobs(self) -> int:
        """At startup: whatever was left 'running'/'queued' by a previous run is not running anymore."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE jobs SET status='failed', message='interrupted (the app was closed)', finished_utc=? "
                "WHERE status IN ('queued','running')",
                (utc_now(),),
            )
            self._conn.commit()
            return cur.rowcount

    @staticmethod
    def _decode_job(row: dict[str, Any]) -> dict[str, Any]:
        for key in ("params", "result"):
            val = row.get(key)
            if isinstance(val, str):
                try:
                    row[key] = json.loads(val)
                except json.JSONDecodeError:
                    pass
        return row
