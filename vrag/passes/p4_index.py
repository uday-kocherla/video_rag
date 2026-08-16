"""Pass 4 — index.

Consumes: SQLite (`units`, `sentences`).
Produces: the transcript BM25 index at `index/bm25.sqlite`.

BM25 comes from SQLite's own FTS5 extension, which ships with Python. That is
one fewer dependency than a Python BM25 library, and it keeps ranking in the
same process as the rows being ranked.

Only the transcript index exists at M3. The other three from CLAUDE.md —
caption, visual, ocr — arrive at M6 once pass 3 has produced anything to put in
them. The dense half of the transcript index is deliberately deferred too: a
BM25-only baseline turns "does dense retrieval help here?" into a measured
ablation instead of an assumption compiled into the baseline.

The whole index is dropped and rebuilt on every run rather than updated
incrementally. That is not laziness about correctness, it is the invariant:
the index is derived state, so the only supported repair is rebuild. Making
that the normal path means it is exercised constantly instead of being a
recovery routine nobody has run.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

from vrag import sync

log = logging.getLogger(__name__)

PASS_NAME = "p4_index"

TRANSCRIPT_TABLE = "transcript"


def open_index(path: str | Path) -> sqlite3.Connection:
    """Open (creating if needed) the BM25 index database."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def unit_texts(conn, *, level: int) -> list[tuple[int, str, str]]:
    """Every unit's transcript text as (unit_id, video_id, text).

    A unit's text is every sentence that overlaps its span. Overlap rather than
    containment: with fixed windows a sentence routinely straddles a boundary,
    and dropping those would silently lose speech from the index.
    """
    rows = conn.execute(
        """
        SELECT u.id      AS unit_id,
               u.video_id AS video_id,
               COALESCE(GROUP_CONCAT(s.text, ' '), '') AS text
        FROM units u
        LEFT JOIN sentences s
               ON s.video_id = u.video_id
              AND s.t_start  < u.t_end
              AND s.t_end    > u.t_start
        WHERE u.level = ?
        GROUP BY u.id
        ORDER BY u.id
        """,
        (level,),
    ).fetchall()
    return [(row["unit_id"], row["video_id"], row["text"]) for row in rows]


def build_transcript_index(conn, index_conn: sqlite3.Connection, *, level: int) -> int:
    """Rebuild the transcript FTS5 table from scratch. Returns rows indexed.

    `unit_id` and `video_id` are UNINDEXED so they are stored and returned but
    never matched — otherwise a query containing a video ID would score every
    unit of that video as a hit.
    """
    texts = unit_texts(conn, level=level)
    index_conn.execute(f"DROP TABLE IF EXISTS {TRANSCRIPT_TABLE}")
    index_conn.execute(
        f"CREATE VIRTUAL TABLE {TRANSCRIPT_TABLE} USING fts5("
        "unit_id UNINDEXED, video_id UNINDEXED, text)"
    )
    index_conn.executemany(
        f"INSERT INTO {TRANSCRIPT_TABLE} (unit_id, video_id, text) VALUES (?, ?, ?)",
        texts,
    )
    index_conn.commit()

    empty = sum(1 for _, _, text in texts if not text.strip())
    if empty:
        # Expected on the silent videos, and worth seeing in the log so it is
        # not mistaken for an indexing bug when those units never retrieve.
        log.info("%d/%d units have no transcript text", empty, len(texts))
    log.info("transcript index: %d units", len(texts))
    return len(texts)


def escape_query(query: str) -> str:
    """Turn a natural-language query into an FTS5 MATCH expression.

    Every term is quoted, because raw user text containing `-`, `*`, `(` or
    `OR` is either an FTS5 syntax error or, worse, a silently different query.
    Terms are OR-ed: FTS5 defaults to AND, which for a natural-language question
    means one unusual word returns nothing at all.
    """
    terms = [term for term in query.replace('"', " ").split() if term.strip()]
    return " OR ".join(f'"{term}"' for term in terms)


def search_transcript(
    index_conn: sqlite3.Connection,
    query: str,
    *,
    k: int,
    video_id: str | None = None,
) -> list[tuple[int, float]]:
    """Top-k (unit_id, score) for a query, best first.

    FTS5's bm25() returns a negative number where more negative is better, so it
    is negated here — every caller downstream assumes higher is better.
    """
    match = escape_query(query)
    if not match:
        return []

    sql = (
        f"SELECT unit_id, bm25({TRANSCRIPT_TABLE}) AS score "
        f"FROM {TRANSCRIPT_TABLE} WHERE {TRANSCRIPT_TABLE} MATCH ?"
    )
    params: list[Any] = [match]
    if video_id is not None:
        sql += " AND video_id = ?"
        params.append(video_id)
    sql += " ORDER BY score LIMIT ?"
    params.append(k)

    # Positional, not by name: this must work on a plain sqlite3.connect() too,
    # not only on a connection that happens to have row_factory set.
    return [(int(unit_id), -score) for unit_id, score in index_conn.execute(sql, params)]


def run_corpus(conn, config: dict[str, Any], local_dir: str | Path) -> int:
    """Rebuild every index. Corpus-wide, not per video — see the module docstring."""
    index_conn = open_index(sync.bm25_path(local_dir))
    try:
        return build_transcript_index(conn, index_conn, level=config["p4"]["level"])
    finally:
        index_conn.close()
