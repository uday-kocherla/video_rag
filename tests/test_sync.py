"""CPU-only, no network — the Hugging Face calls are stubbed out.

The only real logic in sync.py is the WAL guard, so that is what these cover:
a checkpointed database uploads, a dirty one refuses.
"""

from __future__ import annotations

from fnmatch import fnmatch

import pytest

from vrag import db, sync
from vrag.schema import Video


@pytest.fixture()
def artifacts(tmp_path, monkeypatch):
    """An artifact tree with an open database, and a stubbed-out HfApi."""
    uploads = []

    class FakeHfApi:
        def create_repo(self, *args, **kwargs):
            pass

        def upload_folder(self, **kwargs):
            uploads.append(kwargs)

    monkeypatch.setattr(sync, "HfApi", FakeHfApi)

    conn = db.connect(sync.db_path(tmp_path))
    db.insert(conn, [Video(video_id="vid1", url="https://y/1", added_at=0.0)])
    conn.commit()
    yield tmp_path, conn, uploads
    conn.close()


def test_paths_mirror_the_repo_layout(tmp_path):
    assert sync.db_path(tmp_path).as_posix().endswith("db/corpus.sqlite")
    assert sync.derived_dir(tmp_path, "vid1").as_posix().endswith("derived/vid1")
    assert sync.index_dir(tmp_path).as_posix().endswith("index/lance")


def test_checkpoint_empties_the_wal(artifacts):
    local_dir, conn, _ = artifacts
    wal = sync.db_path(local_dir).with_name("corpus.sqlite-wal")
    assert wal.stat().st_size > 0, "committed rows should be sitting in the WAL"

    sync.checkpoint(conn)
    assert wal.stat().st_size == 0


def test_push_checkpoints_when_given_a_connection(artifacts):
    local_dir, conn, uploads = artifacts
    sync.push("me/corpus", local_dir, conn=conn, message="after pass 0")

    assert sync.db_path(local_dir).with_name("corpus.sqlite-wal").stat().st_size == 0
    assert uploads[0]["commit_message"] == "after pass 0"
    assert "*-wal" in uploads[0]["ignore_patterns"]


def uploaded(relpath: str) -> bool:
    """Whether a file at this path survives NEVER_UPLOAD."""
    name = relpath.rsplit("/", 1)[-1]
    return not any(
        fnmatch(name, pattern) or fnmatch(relpath, pattern)
        for pattern in sync.NEVER_UPLOAD
    )


def test_the_proxy_is_uploaded():
    """Pass 3 consumes the proxy, and regenerating it means re-downloading."""
    assert uploaded("derived/abc123/proxy.mp4")
    assert uploaded("derived/abc123/audio.wav")
    assert uploaded("db/corpus.sqlite")


def test_raw_downloads_and_wal_sidecars_are_not_uploaded():
    assert not uploaded("derived/abc123/abc123.webm")
    assert not uploaded("derived/abc123/abc123.mkv")
    assert not uploaded("derived/abc123/abc123.f137.mp4.part")
    assert not uploaded("db/corpus.sqlite-wal")
    assert not uploaded("db/corpus.sqlite-shm")


def test_push_refuses_a_stale_database(artifacts):
    local_dir, conn, uploads = artifacts
    with pytest.raises(RuntimeError, match="un-checkpointed commits"):
        sync.push("me/corpus", local_dir)  # no conn, and the WAL is dirty
    assert uploads == []


def test_push_without_connection_is_fine_once_checkpointed(artifacts):
    local_dir, conn, uploads = artifacts
    sync.checkpoint(conn)
    sync.push("me/corpus", local_dir)
    assert len(uploads) == 1
