"""Pass 5 — single-shot retrieval. The baseline, and the agent's substrate.

Query in, ranked time spans out. No agent, no rewriting, no second pass. This
is what M4 scores, what the agent falls back to when it exhausts its budget,
and what the agentic loop has to beat for the headline ablation to mean
anything.

At M3 the only index is the transcript, so `search_units` fuses a single
ranking. The shape is already the four-way one from CLAUDE.md, so M6 adds
indices to the list rather than rewriting the caller.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

from vrag.passes import p4_index
from vrag.retrieve.fusion import Span, merge_spans, reciprocal_rank_fusion

log = logging.getLogger(__name__)


def search_units(
    index_conn: sqlite3.Connection,
    query: str,
    *,
    k: int,
    video_id: str | None = None,
) -> list[tuple[int, float]]:
    """Fused (unit_id, score) across every available index, best first."""
    rankings = [
        [unit_id for unit_id, _ in p4_index.search_transcript(
            index_conn, query, k=k, video_id=video_id
        )],
    ]
    return reciprocal_rank_fusion(rankings)


def spans_for_units(conn, scored_units: list[tuple[int, float]]) -> list[Span]:
    """Turn ranked unit IDs into spans, carrying each unit's fused score."""
    if not scored_units:
        return []

    scores = dict(scored_units)
    placeholders = ", ".join("?" * len(scores))
    rows = conn.execute(
        f"SELECT id, video_id, t_start, t_end FROM units WHERE id IN ({placeholders})",
        list(scores),
    ).fetchall()
    return [
        Span(
            video_id=row["video_id"],
            t_start=row["t_start"],
            t_end=row["t_end"],
            score=scores[row["id"]],
            unit_ids=(row["id"],),
        )
        for row in rows
    ]


def search(
    conn,
    index_conn: sqlite3.Connection,
    query: str,
    config: dict[str, Any],
    *,
    video_id: str | None = None,
) -> list[Span]:
    """Moment retrieval: a query becomes ranked, merged time spans."""
    settings = config["p5"]
    scored_units = search_units(
        index_conn, query, k=settings["top_k"], video_id=video_id
    )
    spans = spans_for_units(conn, scored_units)
    merged = merge_spans(spans, max_gap_s=settings["max_gap_s"])
    log.debug(
        "query %r: %d units -> %d spans", query, len(scored_units), len(merged)
    )
    return merged
