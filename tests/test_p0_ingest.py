"""Shot-span construction (pure) and real PySceneDetect detection.

Nothing here touches the network — download and metadata are thin yt-dlp
wrappers with no logic of our own to test.
"""

from __future__ import annotations

import pytest

from vrag.passes.p0_ingest import (
    build_ydl_options,
    detect_scene_boundaries,
    spans_from_scene_boundaries,
)
from tests.conftest import needs_ffmpeg


def test_cookies_are_wired_in_only_when_configured():
    assert "cookiefile" not in build_ydl_options(None)
    assert build_ydl_options("/tmp/cookies.txt")["cookiefile"] == "/tmp/cookies.txt"
    assert build_ydl_options(None, skip_download=True)["skip_download"] is True


def test_no_cuts_yields_one_shot_covering_the_video():
    # The single-shot lecture. Zero cuts is a real answer, and pass 2's fallback
    # needs to see one shot spanning 50 minutes, not an empty table.
    assert spans_from_scene_boundaries([], 3000.0) == [(0.0, 3000.0)]


def test_boundaries_are_passed_through_unmerged():
    # A rapid-cut montage really does contain sub-second shots. Smoothing them
    # is pass 2's job; pass 0 records what the detector found.
    boundaries = [(0.0, 0.2), (0.2, 0.5), (0.5, 1.0)]
    assert spans_from_scene_boundaries(boundaries, 1.0) == boundaries


def test_edges_are_closed_so_units_have_no_holes():
    spans = spans_from_scene_boundaries([(0.4, 5.0), (5.0, 9.6)], 10.0)
    assert spans[0][0] == 0.0
    assert spans[-1][1] == 10.0


def test_boundaries_past_the_end_are_clamped():
    spans = spans_from_scene_boundaries([(0.0, 5.0), (5.0, 99.0)], 10.0)
    assert spans == [(0.0, 5.0), (5.0, 10.0)]


def test_degenerate_inputs_do_not_crash():
    assert spans_from_scene_boundaries([], 0.0) == []
    assert spans_from_scene_boundaries([(3.0, 3.0)], 8.0) == [(0.0, 8.0)]  # zero-length only


def test_spans_are_contiguous_and_ordered():
    spans = spans_from_scene_boundaries([(0.0, 2.0), (2.0, 4.0), (4.0, 6.0)], 6.0)
    for (_, end), (next_start, _) in zip(spans, spans[1:]):
        assert end == next_start
    assert spans[0][0] == 0.0 and spans[-1][1] == 6.0


@needs_ffmpeg
def test_detects_the_real_cut_in_the_synthetic_clip(clip):
    boundaries = detect_scene_boundaries(
        clip, threshold=27.0, min_len_frames=5, downscale=None
    )
    spans = spans_from_scene_boundaries(boundaries, 4.0)
    assert len(spans) == 2, f"expected one cut, got spans {spans}"
    assert spans[0][1] == pytest.approx(2.0, abs=0.2), "cut should land at the halfway point"
