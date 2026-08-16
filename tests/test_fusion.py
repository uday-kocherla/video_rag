"""Rank fusion and span merging. Pure, CPU, no index needed."""

from __future__ import annotations

from vrag.retrieve.fusion import Span, merge_spans, reciprocal_rank_fusion


def span(video_id, t_start, t_end, score, unit_id=1) -> Span:
    return Span(
        video_id=video_id,
        t_start=t_start,
        t_end=t_end,
        score=score,
        unit_ids=(unit_id,),
    )


def test_single_ranking_is_preserved():
    fused = reciprocal_rank_fusion([[10, 20, 30]])

    assert [item_id for item_id, _ in fused] == [10, 20, 30]


def test_agreement_between_indices_beats_a_single_top_hit():
    """The property RRF exists for: consensus outranks one index's favourite."""
    fused = reciprocal_rank_fusion([[1, 2], [3, 2]])

    assert fused[0][0] == 2, "2 is second in both, 1 and 3 are first in only one"


def test_missing_from_a_ranking_costs_nothing_extra():
    # An index that has never heard of an item stays silent rather than voting
    # against it — otherwise the OCR index would veto every non-slide query.
    fused = dict(reciprocal_rank_fusion([[1], [1, 2]]))

    assert fused[1] > fused[2]
    assert 2 in fused


def test_score_magnitudes_cannot_influence_fusion():
    """RRF reads ranks only. This is why we never blend raw scores."""
    assert reciprocal_rank_fusion([[7, 8, 9]]) == reciprocal_rank_fusion([[7, 8, 9]])


def test_empty_input_is_empty():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[]]) == []


def test_adjacent_units_merge_into_one_moment():
    # Three consecutive clips are one 48-second moment, not three results.
    spans = [
        span("v1", 0.0, 16.0, 0.9, 1),
        span("v1", 16.0, 32.0, 0.8, 2),
        span("v1", 32.0, 48.0, 0.7, 3),
    ]
    merged = merge_spans(spans, max_gap_s=2.0)

    assert len(merged) == 1
    assert (merged[0].t_start, merged[0].t_end) == (0.0, 48.0)
    assert merged[0].score == 0.9, "a moment is as good as its strongest evidence"
    assert set(merged[0].unit_ids) == {1, 2, 3}


def test_distant_spans_stay_separate():
    spans = [span("v1", 0.0, 16.0, 0.9, 1), span("v1", 300.0, 316.0, 0.8, 2)]

    assert len(merge_spans(spans, max_gap_s=2.0)) == 2


def test_spans_from_different_videos_never_merge():
    spans = [span("v1", 0.0, 16.0, 0.9, 1), span("v2", 16.0, 32.0, 0.8, 2)]
    merged = merge_spans(spans, max_gap_s=2.0)

    assert len(merged) == 2
    assert {s.video_id for s in merged} == {"v1", "v2"}


def test_small_gap_is_bridged():
    spans = [span("v1", 0.0, 16.0, 0.5, 1), span("v1", 17.0, 33.0, 0.9, 2)]
    merged = merge_spans(spans, max_gap_s=2.0)

    assert len(merged) == 1
    assert merged[0].score == 0.9


def test_overlapping_spans_merge_without_shrinking():
    # Overlapping windows are what M5's fixed_overlap produces, so this is the
    # normal case there, not an edge case.
    spans = [span("v1", 0.0, 16.0, 0.9, 1), span("v1", 8.0, 24.0, 0.8, 2)]
    merged = merge_spans(spans, max_gap_s=0.0)

    assert (merged[0].t_start, merged[0].t_end) == (0.0, 24.0)


def test_nested_span_does_not_truncate_its_container():
    spans = [span("v1", 0.0, 60.0, 0.9, 1), span("v1", 10.0, 20.0, 0.5, 2)]
    merged = merge_spans(spans, max_gap_s=0.0)

    assert merged[0].t_end == 60.0


def test_results_come_back_best_first():
    spans = [span("v1", 0.0, 16.0, 0.2, 1), span("v2", 0.0, 16.0, 0.9, 2)]
    merged = merge_spans(spans, max_gap_s=2.0)

    assert [s.score for s in merged] == [0.9, 0.2]


def test_no_spans_is_empty():
    assert merge_spans([], max_gap_s=2.0) == []
