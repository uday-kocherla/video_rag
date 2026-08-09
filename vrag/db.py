"""SQLite source of truth: connection, migrations, stage tracking.

Two things here are load-bearing rather than incidental:

* `connect` refuses a network filesystem. Drive/GCS-mounted SQLite has broken
  advisory locking and corrupts silently under WAL. Failing loudly at open is
  cheaper than discovering it three passes later.
* `stage_run` wraps a pass in a transaction. On failure it rolls the pass's
  rows back, so a re-run appends to a clean slate instead of duplicating a
  half-written video. That rollback is what makes idempotency real.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, fields
from typing import Any, Iterable, Iterator

log = logging.getLogger(__name__)

# Append only. Index = PRAGMA user_version after it has been applied.
MIGRATIONS: list[str] = [
    """
    CREATE TABLE videos (
        video_id   TEXT PRIMARY KEY,
        url        TEXT NOT NULL,
        added_at   REAL NOT NULL,
        title      TEXT,
        duration_s REAL,
        fps        REAL,
        width      INTEGER,
        height     INTEGER,
        chapters   TEXT
    );

    CREATE TABLE shots (
        id       INTEGER PRIMARY KEY,
        video_id TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
        t_start  REAL NOT NULL,
        t_end    REAL NOT NULL,
        conf     REAL
    );

    CREATE TABLE words (
        id       INTEGER PRIMARY KEY,
        video_id TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
        t_start  REAL NOT NULL,
        t_end    REAL NOT NULL,
        text     TEXT NOT NULL,
        conf     REAL
    );

    CREATE TABLE sentences (
        id       INTEGER PRIMARY KEY,
        video_id TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
        t_start  REAL NOT NULL,
        t_end    REAL NOT NULL,
        text     TEXT NOT NULL,
        speaker  TEXT
    );

    CREATE TABLE units (
        id        INTEGER PRIMARY KEY,
        video_id  TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
        level     INTEGER NOT NULL,
        t_start   REAL NOT NULL,
        t_end     REAL NOT NULL,
        strategy  TEXT NOT NULL,
        parent_id INTEGER REFERENCES units(id) ON DELETE SET NULL,
        flags     TEXT
    );

    CREATE TABLE captions (
        id        INTEGER PRIMARY KEY,
        unit_id   INTEGER NOT NULL REFERENCES units(id) ON DELETE CASCADE,
        captioner TEXT NOT NULL,
        payload   TEXT NOT NULL
    );

    CREATE TABLE ocr_spans (
        id        INTEGER PRIMARY KEY,
        video_id  TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
        t_start   REAL NOT NULL,
        t_end     REAL NOT NULL,
        text      TEXT NOT NULL,
        conf      REAL,
        text_hash TEXT
    );

    CREATE TABLE stage_runs (
        id           INTEGER PRIMARY KEY,
        video_id     TEXT NOT NULL,
        pass_name    TEXT NOT NULL,
        status       TEXT NOT NULL,          -- running | ok | failed
        config_hash  TEXT NOT NULL,
        started_at   REAL NOT NULL,
        ended_at     REAL,
        rows_out     INTEGER,
        peak_vram_mb REAL,
        error        TEXT
    );

    -- (video_id, t_start) is the `seek` access path; nothing else needs an index yet.
    CREATE INDEX shots_t     ON shots(video_id, t_start);
    CREATE INDEX words_t     ON words(video_id, t_start);
    CREATE INDEX sentences_t ON sentences(video_id, t_start);
    CREATE INDEX units_t     ON units(video_id, level, t_start);
    CREATE INDEX captions_u  ON captions(unit_id);
    CREATE INDEX ocr_t       ON ocr_spans(video_id, t_start);
    CREATE INDEX stage_lookup ON stage_runs(video_id, pass_name, status);
    """,
]

# fuse.* is matched by prefix (fuse.drivefs, fuse.gcsfuse, fuse.sshfs, ...).
# fuseblk is deliberately absent: that's ntfs-3g on local disk.
_NETWORK_FS = {"nfs", "nfs4", "cifs", "smbfs", "smb3", "9p", "afs", "sshfs", "fuse"}


def _fstype(path: str) -> str | None:
    """Filesystem type of the mount that `path` lands on, or None if unknowable."""
    try:
        with open("/proc/mounts", encoding="utf-8") as f:
            mounts = f.read().splitlines()
    except OSError:  # not Linux, or /proc unavailable — caller degrades to a warning
        return None
    target = os.path.realpath(path)
    best_mp, best_type = "", None
    for line in mounts:
        parts = line.split()
        if len(parts) < 3:
            continue
        mp = parts[1].replace("\\040", " ")
        if (target == mp or target.startswith(mp.rstrip("/") + "/")) and len(mp) >= len(best_mp):
            best_mp, best_type = mp, parts[2]
    return best_type


def connect(path: str | os.PathLike[str]) -> sqlite3.Connection:
    """Open (creating if needed) the corpus DB in WAL mode, migrated to head."""
    path = os.fspath(path)
    fstype = _fstype(path)
    if fstype and (fstype in _NETWORK_FS or fstype.startswith("fuse.")):
        raise RuntimeError(
            f"refusing to open SQLite on {fstype!r} filesystem: {path}. "
            "Network/FUSE mounts have broken file locking and will corrupt the DB. "
            "Use local disk and sync the file as an artifact."
        )
    if fstype is None:
        log.warning("could not determine filesystem for %s; assuming local disk", path)

    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA busy_timeout = 30000")
    migrate(conn)
    return conn


def migrate(conn: sqlite3.Connection) -> int:
    """Apply pending MIGRATIONS. Returns the resulting schema version.

    Caption *content* never comes through here: it lives as JSON in
    captions.payload and is read with json_extract, so reshaping the caption
    schema costs nothing. This path is only for real structural change.
    """
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    for i, sql in enumerate(MIGRATIONS[version:], start=version):
        log.info("applying migration %d", i + 1)
        conn.executescript(sql)
        conn.execute(f"PRAGMA user_version = {i + 1}")
        conn.commit()
    return len(MIGRATIONS)


def insert(conn: sqlite3.Connection, rows: Iterable[Any]) -> int:
    """Bulk-insert homogeneous schema dataclasses. Returns the row count."""
    rows = list(rows)
    if not rows:
        return 0
    cls = type(rows[0])
    cols = [f.name for f in fields(cls) if f.name != "id"]
    sql = (
        f"INSERT INTO {cls.table} ({', '.join(cols)}) "
        f"VALUES ({', '.join('?' * len(cols))})"
    )
    conn.executemany(sql, [tuple(getattr(r, c) for c in cols) for r in rows])
    return len(rows)


def config_hash(config: Any) -> str:
    """Stable short hash of a pass's own config subtree.

    Part of the skip key, so changing pass 2's config re-runs pass 2 without
    invalidating ASR — that's what makes the M5 segmentation sweep possible.
    """
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def stage_done(conn: sqlite3.Connection, video_id: str, pass_name: str, cfg_hash: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM stage_runs "
        "WHERE video_id = ? AND pass_name = ? AND config_hash = ? AND status = 'ok' LIMIT 1",
        (video_id, pass_name, cfg_hash),
    ).fetchone()
    return row is not None


def _vram_peak_mb() -> float | None:
    try:
        import torch  # noqa: PLC0415 — optional; absent on CPU-only test runs
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 2**20


def _vram_reset() -> None:
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


@dataclass
class StageRun:
    """Handle yielded by stage_run; the pass sets rows_out as it writes."""

    rows_out: int = 0


@contextmanager
def stage_run(
    conn: sqlite3.Connection, video_id: str, pass_name: str, cfg_hash: str
) -> Iterator[StageRun]:
    """Run one (video, pass) as a transaction, logged to stage_runs.

    Re-raises on failure: the corpus loop is responsible for catching per video
    and continuing, so one bad video can't take down a 9-hour run.
    """
    conn.commit()  # don't let a caller's open transaction ride along into ours
    started = time.time()
    cur = conn.execute(
        "INSERT INTO stage_runs (video_id, pass_name, status, config_hash, started_at) "
        "VALUES (?, ?, 'running', ?, ?)",
        (video_id, pass_name, cfg_hash, started),
    )
    run_id = cur.lastrowid
    conn.commit()  # survives the rollback below, so a crashed run stays visible
    _vram_reset()

    handle = StageRun()
    try:
        yield handle
    except BaseException as exc:
        conn.rollback()
        conn.execute(
            "UPDATE stage_runs SET status='failed', ended_at=?, peak_vram_mb=?, error=? WHERE id=?",
            (time.time(), _vram_peak_mb(), f"{type(exc).__name__}: {exc}"[:2000], run_id),
        )
        conn.commit()
        log.exception("pass %s failed on %s", pass_name, video_id)
        raise
    else:
        conn.commit()
        conn.execute(
            "UPDATE stage_runs SET status='ok', ended_at=?, rows_out=?, peak_vram_mb=? WHERE id=?",
            (time.time(), handle.rows_out, _vram_peak_mb(), run_id),
        )
        conn.commit()
        log.info(
            "pass %s ok on %s: %d rows in %.1fs",
            pass_name, video_id, handle.rows_out, time.time() - started,
        )
