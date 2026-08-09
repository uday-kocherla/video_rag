"""CPU-only, no network. Covers the two things that silently rot: migration
idempotency and the stage_run rollback that idempotency depends on."""

from __future__ import annotations

import sqlite3

import pytest

from vrag import db
from vrag.schema import Caption, Sentence, Unit, Video


@pytest.fixture()
def conn(tmp_path):
    c = db.connect(tmp_path / "corpus.sqlite")
    db.insert(c, [Video(video_id="vid1", url="https://y/1", added_at=0.0)])
    c.commit()
    yield c
    c.close()


def test_connect_sets_wal_and_migrates(tmp_path):
    c = db.connect(tmp_path / "a.sqlite")
    assert c.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert c.execute("PRAGMA user_version").fetchone()[0] == len(db.MIGRATIONS)
    assert db.migrate(c) == len(db.MIGRATIONS)  # second call is a no-op, not a crash
    c.close()


def test_refuses_network_filesystem(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "_fstype", lambda p: "fuse.drivefs")
    with pytest.raises(RuntimeError, match="refusing to open SQLite"):
        db.connect(tmp_path / "drive.sqlite")


def test_insert_roundtrip(conn):
    n = db.insert(conn, [
        Sentence(video_id="vid1", t_start=0.0, t_end=2.5, text="hello"),
        Sentence(video_id="vid1", t_start=2.5, t_end=4.0, text="world", speaker="A"),
    ])
    assert n == 2
    rows = conn.execute("SELECT * FROM sentences ORDER BY t_start").fetchall()
    assert [r["text"] for r in rows] == ["hello", "world"]
    assert rows[0]["speaker"] is None and rows[0]["id"] == 1


def test_caption_payload_is_queryable_without_migration(conn):
    db.insert(conn, [Unit(video_id="vid1", level=0, t_start=0.0, t_end=15.0, strategy="fixed")])
    db.insert(conn, [Caption(unit_id=1, captioner="qwen/v1", payload='{"summary": "a cat"}')])
    got = conn.execute("SELECT json_extract(payload, '$.summary') s FROM captions").fetchone()
    assert got["s"] == "a cat"


def test_stage_done_keys_on_config_hash(conn):
    h1, h2 = db.config_hash({"target_s": 15}), db.config_hash({"target_s": 20})
    assert h1 != h2
    assert not db.stage_done(conn, "vid1", "p2_segment", h1)
    with db.stage_run(conn, "vid1", "p2_segment", h1) as run:
        run.rows_out = db.insert(conn, [
            Unit(video_id="vid1", level=0, t_start=0.0, t_end=15.0, strategy="fixed")
        ])
    assert db.stage_done(conn, "vid1", "p2_segment", h1)
    assert not db.stage_done(conn, "vid1", "p2_segment", h2)  # sweep must not be skipped

    row = conn.execute("SELECT * FROM stage_runs WHERE status='ok'").fetchone()
    assert row["rows_out"] == 1 and row["ended_at"] >= row["started_at"]


def test_failed_pass_rolls_back_its_rows(conn):
    h = db.config_hash({})
    with pytest.raises(ZeroDivisionError):
        with db.stage_run(conn, "vid1", "p1_speech", h) as run:
            run.rows_out = db.insert(conn, [
                Sentence(video_id="vid1", t_start=0.0, t_end=1.0, text="partial")
            ])
            1 / 0

    assert conn.execute("SELECT count(*) c FROM sentences").fetchone()["c"] == 0
    assert not db.stage_done(conn, "vid1", "p1_speech", h)
    row = conn.execute("SELECT * FROM stage_runs").fetchone()
    assert row["status"] == "failed" and "ZeroDivisionError" in row["error"]

    # ...and the re-run appends to a clean slate rather than duplicating.
    with db.stage_run(conn, "vid1", "p1_speech", h) as run:
        run.rows_out = db.insert(conn, [
            Sentence(video_id="vid1", t_start=0.0, t_end=1.0, text="partial")
        ])
    assert conn.execute("SELECT count(*) c FROM sentences").fetchone()["c"] == 1


def test_foreign_keys_enforced(conn):
    with pytest.raises(sqlite3.IntegrityError):
        db.insert(conn, [Sentence(video_id="ghost", t_start=0.0, t_end=1.0, text="x")])
        conn.commit()
