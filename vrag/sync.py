"""Sync the artifact tree between local disk and a private Hugging Face dataset repo.

A Kaggle or Colab session starts empty and dies without warning, so every session
is bracketed by `pull(...)` at the top and `push(...)` at the bottom. The repo is
the durable copy; local disk is scratch.

Repo layout (mirrored exactly on local disk):

    db/corpus.sqlite        the source of truth
    derived/{video_id}/     audio.wav, the 360p proxy, keyframes
    index/lance/            LanceDB tables — disposable, rebuildable by pass 4

The one non-obvious rule this module exists to enforce: a WAL-mode database is
not fully contained in its .sqlite file. Uploading without checkpointing first
ships a database missing most of the session's work. See `checkpoint`.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

log = logging.getLogger(__name__)

REPO_TYPE = "dataset"

# Paths inside the artifact repo. Local disk mirrors these exactly, so the same
# relative path works on both sides and nothing has to translate between them.
DB_RELPATH = "db/corpus.sqlite"
DERIVED_RELPATH = "derived"
INDEX_RELPATH = "index/lance"

# Raw video is regenerable from its URL and would dominate the repo size; captions
# are regenerable only by paying for a GPU again. The sidecar -wal and -shm files
# are meaningless outside the machine that wrote them (see `checkpoint`).
NEVER_UPLOAD = [
    "*.mp4",
    "*.mkv",
    "*.webm",
    "*.part",
    "*.ytdl",
    "*-wal",
    "*-shm",
    "**/__pycache__/**",
    "**/.ipynb_checkpoints/**",
]


def db_path(local_dir: str | Path) -> Path:
    """Local path of the corpus database inside an artifact tree."""
    return Path(local_dir) / DB_RELPATH


def derived_dir(local_dir: str | Path, video_id: str) -> Path:
    """Local directory holding one video's derived files (audio, proxy, keyframes)."""
    return Path(local_dir) / DERIVED_RELPATH / video_id


def index_dir(local_dir: str | Path) -> Path:
    """Local directory holding the LanceDB tables."""
    return Path(local_dir) / INDEX_RELPATH


def pull(
    repo_id: str,
    local_dir: str | Path,
    *,
    allow_patterns: list[str] | None = None,
) -> Path:
    """Download the artifact tree, creating the repo if this is the first run.

    Pass `allow_patterns` to fetch only part of it — a segmentation experiment
    needs `["db/**"]` and nothing else, which turns a multi-gigabyte pull into a
    few seconds.
    """
    local_dir = Path(local_dir)
    HfApi().create_repo(repo_id, repo_type=REPO_TYPE, private=True, exist_ok=True)
    snapshot_download(
        repo_id,
        repo_type=REPO_TYPE,
        local_dir=local_dir,
        allow_patterns=allow_patterns,
    )
    log.info("pulled %s into %s", repo_id, local_dir)
    return local_dir


def push(
    repo_id: str,
    local_dir: str | Path,
    *,
    conn: sqlite3.Connection | None = None,
    message: str | None = None,
    allow_patterns: list[str] | None = None,
) -> None:
    """Upload the artifact tree.

    Pass the live `conn` so the database is checkpointed before it is read. If
    you cannot (the connection is already closed), this still refuses to upload
    a database with un-checkpointed commits sitting in its WAL.
    """
    local_dir = Path(local_dir)
    if conn is not None:
        checkpoint(conn)
    else:
        _refuse_if_wal_is_dirty(local_dir)

    HfApi().upload_folder(
        repo_id=repo_id,
        repo_type=REPO_TYPE,
        folder_path=str(local_dir),
        allow_patterns=allow_patterns,
        ignore_patterns=NEVER_UPLOAD,
        commit_message=message or "sync artifacts",
    )
    log.info("pushed %s to %s", local_dir, repo_id)


def checkpoint(conn: sqlite3.Connection) -> None:
    """Fold the write-ahead log back into the .sqlite file.

    In WAL mode, committed transactions live in a separate `corpus.sqlite-wal`
    file until a checkpoint moves them across. `-wal` is in NEVER_UPLOAD because
    it is only valid alongside the machine-local `-shm` index, so without this
    call the uploaded database is silently stale — usually by an entire session.
    """
    conn.commit()
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")


def _refuse_if_wal_is_dirty(local_dir: Path) -> None:
    """Guard for the case where `push` was not given a connection to checkpoint."""
    wal = db_path(local_dir).with_name(db_path(local_dir).name + "-wal")
    if wal.exists() and wal.stat().st_size > 0:
        raise RuntimeError(
            f"{wal} still holds un-checkpointed commits, so uploading "
            f"{DB_RELPATH} would ship a stale database. Pass the open connection "
            "as push(..., conn=conn), or call sync.checkpoint(conn) first."
        )
