"""Pass 2 windowing, including the three degenerate cases CLAUDE.md calls out.

Pure functions on synthetic shot and sentence lists. No models, no ffmpeg.
"""

from __future__ import annotations

import pytest

from vrag.passes.p2_segment import (
    FLAG_SILENT,
    UnitSpan,
    absorb_runt_tail,
    build_l0_spans,
    chapter_windows,
    fixed_windows,
    find_parent_id,
    shot_windows,
    speech_density,
    split_long_span,
)

SETTINGS = {
    "strategy": "fixed",
    "l0": {"target_s": 16.0, "min_s": 8.0, "max_s": 30.0, "stride_ratio": 1.0},
    "l1": {"min_s": 120.0, "max_s": 300.0},
    "silent_words_per_s": 0.3,
}


def talking(seconds: float) -> list[str]:
    """Sentence texts dense enough to count as real speech."""
    return ["word " * 10] * int(seconds / 5)


# --------------------------------------------------------------------------
# fixed windows
# --------------------------------------------------------------------------


def test_windows_tile_the_video_without_holes():
    windows = fixed_windows(100.0, target_s=16.0, stride_s=16.0, min_s=8.0)

    assert windows[0][0] == 0.0
    assert windows[-1][1] == 100.0
    for earlier, later in zip(windows, windows[1:]):
        assert earlier[1] == later[0], "fixed windows must not leave gaps"


def test_short_video_is_one_window():
    assert fixed_windows(10.0, target_s=16.0, stride_s=16.0, min_s=8.0) == [(0.0, 10.0)]


def test_runt_tail_is_absorbed_rather_than_emitted():
    # 34s at 16s windows would leave a 2s tail that no query can usefully hit.
    windows = fixed_windows(34.0, target_s=16.0, stride_s=16.0, min_s=8.0)

    assert windows == [(0.0, 16.0), (16.0, 34.0)]
    assert all(end - start >= 8.0 for start, end in windows)


def test_overlapping_stride_produces_overlapping_windows():
    # The mechanism M5's `fixed_overlap` sweep will use.
    windows = fixed_windows(60.0, target_s=16.0, stride_s=8.0, min_s=8.0)

    assert windows[1][0] < windows[0][1], "windows should overlap"


def test_zero_duration_yields_nothing():
    assert fixed_windows(0.0, target_s=16.0, stride_s=16.0, min_s=8.0) == []


def test_non_positive_stride_is_rejected():
    # Would loop forever rather than fail, which is the worst kind of bug in a
    # nine-hour batch run.
    with pytest.raises(ValueError):
        fixed_windows(60.0, target_s=16.0, stride_s=0.0, min_s=8.0)


# --------------------------------------------------------------------------
# degenerate case: single-shot lecture
# --------------------------------------------------------------------------


def test_single_shot_lecture_still_gets_many_units():
    """PySceneDetect returns one shot for a 50-minute talking head."""
    one_shot = [(0.0, 3000.0)]
    windows = shot_windows(one_shot, 3000.0, target_s=16.0, max_s=30.0, min_s=8.0)

    assert len(windows) > 1, "one unit for 50 minutes is useless for moment retrieval"
    assert all(end - start <= 30.0 for start, end in windows)


def test_split_long_span_respects_the_cap():
    pieces = split_long_span(0.0, 100.0, max_s=30.0)

    assert len(pieces) == 4
    assert all(end - start <= 30.0 for start, end in pieces)
    assert pieces[0][0] == 0.0 and pieces[-1][1] == 100.0


def test_split_leaves_short_spans_alone():
    assert split_long_span(5.0, 20.0, max_s=30.0) == [(5.0, 20.0)]


# --------------------------------------------------------------------------
# degenerate case: rapid-cut montage
# --------------------------------------------------------------------------


def test_rapid_cuts_are_merged_not_emitted_one_per_shot():
    """Hundreds of sub-second shots must not become hundreds of units."""
    shots = [(i * 0.4, (i + 1) * 0.4) for i in range(250)]  # 100s of fast cuts
    windows = shot_windows(shots, 100.0, target_s=16.0, max_s=30.0, min_s=8.0)

    assert len(windows) < 20, f"expected merged windows, got {len(windows)}"
    assert all(end - start >= 8.0 for start, end in windows)


def test_shot_windows_start_and_end_on_shot_boundaries():
    shots = [(0.0, 10.0), (10.0, 22.0), (22.0, 40.0)]
    windows = shot_windows(shots, 40.0, target_s=16.0, max_s=30.0, min_s=8.0)

    boundaries = {0.0, 10.0, 22.0, 40.0}
    assert windows[0][0] in boundaries
    assert windows[-1][1] == 40.0


# --------------------------------------------------------------------------
# degenerate case: silent video
# --------------------------------------------------------------------------


def test_speech_density_separates_silent_from_talking():
    assert speech_density(["one two three"], 100.0) < 0.3
    assert speech_density(talking(100.0), 100.0) > 0.3


def test_silent_video_uses_shots_and_flags_every_unit():
    shots = [(0.0, 20.0), (20.0, 45.0), (45.0, 60.0)]
    spans = build_l0_spans([], shots, 60.0, SETTINGS)

    assert spans, "a silent video must still be segmented"
    assert all(FLAG_SILENT in span.flags for span in spans), (
        "retrieval needs to know the transcript index has nothing here"
    )


def test_talking_video_is_not_flagged_silent():
    spans = build_l0_spans(talking(60.0), [(0.0, 60.0)], 60.0, SETTINGS)

    assert spans
    assert all(span.flags == () for span in spans)


def test_silent_video_with_no_shots_still_gets_units():
    spans = build_l0_spans([], [], 60.0, SETTINGS)

    assert spans
    assert all(FLAG_SILENT in span.flags for span in spans)


# --------------------------------------------------------------------------
# L1 scenes from chapters
# --------------------------------------------------------------------------


def test_chapters_become_scenes():
    chapters = [
        {"start_time": 0, "end_time": 180},
        {"start_time": 180, "end_time": 420},
    ]
    assert chapter_windows(chapters, 420.0, min_s=120.0, max_s=300.0) == [
        (0.0, 180.0), (180.0, 420.0)
    ]


def test_overlong_chapter_is_split():
    chapters = [{"start_time": 0, "end_time": 1200}]
    scenes = chapter_windows(chapters, 1200.0, min_s=120.0, max_s=300.0)

    assert len(scenes) == 4
    assert all(end - start <= 300.0 for start, end in scenes)


def test_short_chapters_merge_forward():
    # A 20-second intro is not a scene.
    chapters = [
        {"start_time": 0, "end_time": 20},
        {"start_time": 20, "end_time": 200},
    ]
    scenes = chapter_windows(chapters, 200.0, min_s=120.0, max_s=300.0)

    assert scenes == [(0.0, 200.0)]


def test_no_chapters_yields_no_scenes():
    # L1 from embedding similarity arrives at M6; until then, no chapters means
    # no scenes rather than a fabricated one.
    assert chapter_windows([], 600.0, min_s=120.0, max_s=300.0) == []


def test_chapters_are_clamped_to_duration():
    chapters = [{"start_time": 0, "end_time": 9999}]
    scenes = chapter_windows(chapters, 200.0, min_s=120.0, max_s=300.0)

    assert scenes[-1][1] == 200.0


# --------------------------------------------------------------------------
# parent linking
# --------------------------------------------------------------------------


class FakeScene(dict):
    """Stands in for a sqlite3.Row, which is also subscript-accessed."""


def test_clip_is_parented_to_the_scene_holding_its_midpoint():
    scenes = [
        FakeScene(id=1, t_start=0.0, t_end=100.0),
        FakeScene(id=2, t_start=100.0, t_end=200.0),
    ]

    assert find_parent_id(scenes, UnitSpan(10.0, 26.0)) == 1
    assert find_parent_id(scenes, UnitSpan(150.0, 166.0)) == 2


def test_clip_straddling_a_boundary_picks_exactly_one_parent():
    scenes = [
        FakeScene(id=1, t_start=0.0, t_end=100.0),
        FakeScene(id=2, t_start=100.0, t_end=200.0),
    ]
    # Midpoint 100.0 falls in scene 2 — the rule is arbitrary but must be single-valued.
    assert find_parent_id(scenes, UnitSpan(92.0, 108.0)) == 2


def test_clip_with_no_scenes_has_no_parent():
    assert find_parent_id([], UnitSpan(0.0, 16.0)) is None


def test_absorb_runt_tail_leaves_a_lone_window_alone():
    assert absorb_runt_tail([(0.0, 3.0)], min_s=8.0) == [(0.0, 3.0)]
