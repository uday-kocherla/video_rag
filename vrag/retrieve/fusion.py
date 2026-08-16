"""Rank fusion and span merging.

Reciprocal rank fusion reads *ranks*, never scores. That is the whole reason it
is here rather than a weighted score blend: SigLIP cosines cluster in a narrow
band while BM25 scores spread across orders of magnitude, so any scheme that
adds raw scores hands every query to whichever index happens to have the widest
dynamic range. RRF cannot see that, so it cannot be fooled by it.

At M3 there is one index and fusion is an identity function. It is written now
anyway — it is eight lines, it is the substrate M6 plugs three more indices
into, and having it tested before it matters means the four-way fusion is not
also the first time this code has ever run.
"""

from __future__ import annotations

from dataclasses import dataclass

# The constant from the original RRF paper. It damps the top ranks so a single
# index cannot dominate the fused order on its own.
DEFAULT_RRF_K = 60


@dataclass(frozen=True, slots=True)
class Span:
    """A retrieved time span — what moment retrieval is scored on."""

    video_id: str
    t_start: float
    t_end: float
    score: float
    unit_ids: tuple[int, ...]


def reciprocal_rank_fusion(
    rankings: list[list[int]], *, rrf_k: int = DEFAULT_RRF_K
) -> list[tuple[int, float]]:
    """Fuse ranked ID lists into one, best first.

    Each list is one index's opinion, best first. An item missing from a list
    simply scores nothing from it — no imputation, no penalty, which is what
    lets an index that knows nothing about a query stay silent instead of
    voting against it.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (rrf_k + rank + 1)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


def merge_spans(
    spans: list[Span], *, max_gap_s: float
) -> list[Span]:
    """Merge touching or nearly-touching spans from the same video.

    Retrieval returns units, but a moment is not a unit — a 40-second answer
    lands as three consecutive clips, and returning them as three results both
    triples the apparent hit count and scores terribly on tIoU against a single
    ground-truth span. The merged span takes the best score of its members,
    because the moment is as good as its strongest evidence.
    """
    if not spans:
        return []

    ordered = sorted(spans, key=lambda span: (span.video_id, span.t_start))
    merged: list[Span] = [ordered[0]]
    for span in ordered[1:]:
        current = merged[-1]
        contiguous = (
            span.video_id == current.video_id
            and span.t_start - current.t_end <= max_gap_s
        )
        if contiguous:
            merged[-1] = Span(
                video_id=current.video_id,
                t_start=current.t_start,
                t_end=max(current.t_end, span.t_end),
                score=max(current.score, span.score),
                unit_ids=current.unit_ids + span.unit_ids,
            )
        else:
            merged.append(span)

    return sorted(merged, key=lambda span: -span.score)
