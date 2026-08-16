"""Pass 2 — segmentation.

Consumes: `shots`, `sentences`.
Produces: `units` rows — L0 clips (the retrieval grain) and L1 scenes (context).

Also reads `videos.duration_s` and `videos.chapters`. That is wider than the
Consumes column in CLAUDE.md, and it is deliberate: duration bounds every
window, and the same file specifies L1 as coming "from YouTube chapters where
present", which only exist on the videos row. Flagging it rather than doing it
quietly, per workflow rule 4.

Only the `fixed` strategy is implemented. It is the honest baseline — literal
time windows that know nothing about sentence or shot boundaries — and making
it clever would rob M5's sweep of anything to prove. `fixed_overlap` and
`snapped` land there.

The three degenerate cases from CLAUDE.md are handled here rather than left to
retrieval:

* single-shot lecture — one shot spanning 50 minutes, so shot boundaries carry
  no information. Time windowing ignores shots entirely, and the silent path
  splits any over-long shot by time.
* rapid-cut montage — hundreds of sub-second shots. The silent path accumulates
  shots until they reach the target length instead of emitting one unit each.
* silent video — no usable speech, so there is nothing to snap to and nothing
  to embed. Windows follow shot boundaries and every unit is flagged `silent`.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any

from vrag import db
from vrag.schema import Unit

log = logging.getLogger(__name__)

PASS_NAME = "p2_segment"

STRATEGY_FIXED = "fixed"
IMPLEMENTED_STRATEGIES = {STRATEGY_FIXED}

LEVEL_CLIP = 0
LEVEL_SCENE = 1

FLAG_SILENT = "silent"


@dataclass(frozen=True, slots=True)
class UnitSpan:
    """A candidate unit before it becomes a row."""

    t_start: float
    t_end: float
    flags: tuple[str, ...] = field(default=())


def config_hash(settings: dict[str, Any]) -> str:
    """Skip key for pass 2. Changing it re-segments without touching ASR."""
    return db.config_hash(settings)


# --------------------------------------------------------------------------
# Windowing — pure, and the part worth testing
# --------------------------------------------------------------------------


def speech_density(sentence_texts: list[str], duration_s: float) -> float:
    """Words per second of video. Below ~0.3 the transcript is not usable."""
    if duration_s <= 0:
        return 0.0
    words = sum(len(text.split()) for text in sentence_texts)
    return words / duration_s


def fixed_windows(
    duration_s: float, *, target_s: float, stride_s: float, min_s: float
) -> list[tuple[float, float]]:
    """Fixed-length time windows across the whole video.

    Knows nothing about sentences or shots — that is the point of the baseline.
    A tail shorter than `min_s` is absorbed into the window before it instead of
    being emitted as a runt nobody can retrieve.
    """
    if duration_s <= 0:
        return []
    if stride_s <= 0:
        raise ValueError(f"stride must be positive, got {stride_s}")
    if duration_s <= target_s:
        return [(0.0, duration_s)]

    windows: list[tuple[float, float]] = []
    start = 0.0
    while start < duration_s:
        end = min(start + target_s, duration_s)
        if 0 < duration_s - end < min_s:
            end = duration_s
        windows.append((start, end))
        if end >= duration_s:
            break
        start += stride_s
    return windows


def shot_windows(
    shots: list[tuple[float, float]],
    duration_s: float,
    *,
    target_s: float,
    max_s: float,
    min_s: float,
) -> list[tuple[float, float]]:
    """Windows cut on shot boundaries, for video with no usable speech.

    Shots are accumulated until they reach `target_s`, which is what stops a
    rapid-cut montage from producing one unit per sub-second shot. Any resulting
    window longer than `max_s` is split by time, which is what stops a
    single-shot lecture from producing one unit for the whole video.
    """
    if not shots:
        return fixed_windows(
            duration_s, target_s=target_s, stride_s=target_s, min_s=min_s
        )

    windows: list[tuple[float, float]] = []
    start: float | None = None
    for shot_start, shot_end in shots:
        if start is None:
            start = shot_start
        if shot_end - start >= target_s:
            windows.extend(split_long_span(start, shot_end, max_s=max_s))
            start = None

    if start is not None and duration_s > start:
        windows.extend(split_long_span(start, duration_s, max_s=max_s))
    return absorb_runt_tail(windows, min_s=min_s)


def split_long_span(
    t_start: float, t_end: float, *, max_s: float
) -> list[tuple[float, float]]:
    """Break a span longer than `max_s` into equal pieces, each under the cap."""
    length = t_end - t_start
    if length <= max_s:
        return [(t_start, t_end)]
    pieces = math.ceil(length / max_s)
    step = length / pieces
    return [
        (t_start + i * step, t_start + (i + 1) * step if i < pieces - 1 else t_end)
        for i in range(pieces)
    ]


def absorb_runt_tail(
    windows: list[tuple[float, float]], *, min_s: float
) -> list[tuple[float, float]]:
    """Fold a too-short final window into the one before it."""
    if len(windows) < 2:
        return windows
    last_start, last_end = windows[-1]
    if last_end - last_start >= min_s:
        return windows
    previous_start, _ = windows[-2]
    return windows[:-2] + [(previous_start, last_end)]


def chapter_windows(
    chapters: list[dict[str, Any]],
    duration_s: float,
    *,
    min_s: float,
    max_s: float,
) -> list[tuple[float, float]]:
    """L1 scenes from YouTube chapters, bounded to the configured range.

    Chapters are author-supplied and wildly uneven — a 20-second intro next to a
    40-minute body. Short ones merge forward, long ones split by time.
    """
    spans: list[tuple[float, float]] = []
    for chapter in chapters:
        start = float(chapter.get("start_time", 0.0))
        end = float(chapter.get("end_time", duration_s))
        if end > start:
            spans.append((start, min(end, duration_s)))
    if not spans:
        return []

    # Accumulate until the run is long enough to be a scene. Merging forward
    # rather than backward matters: the classic short chapter is a 20-second
    # intro at position zero, which has no previous scene to fold into.
    merged: list[tuple[float, float]] = []
    start: float | None = None
    for span_start, span_end in spans:
        if start is None:
            start = span_start
        if span_end - start >= min_s:
            merged.append((start, span_end))
            start = None
    if start is not None:
        merged.append((start, spans[-1][1]))

    bounded: list[tuple[float, float]] = []
    for start, end in merged:
        bounded.extend(split_long_span(start, end, max_s=max_s))
    return absorb_runt_tail(bounded, min_s=min_s)


def build_l0_spans(
    sentence_texts: list[str],
    shots: list[tuple[float, float]],
    duration_s: float,
    settings: dict[str, Any],
) -> list[UnitSpan]:
    """L0 clips for one video, choosing the speech or silent path."""
    l0 = settings["l0"]
    density = speech_density(sentence_texts, duration_s)

    if density < settings["silent_words_per_s"]:
        windows = shot_windows(
            shots,
            duration_s,
            target_s=l0["target_s"],
            max_s=l0["max_s"],
            min_s=l0["min_s"],
        )
        return [UnitSpan(start, end, (FLAG_SILENT,)) for start, end in windows]

    windows = fixed_windows(
        duration_s,
        target_s=l0["target_s"],
        stride_s=l0["target_s"] * l0["stride_ratio"],
        min_s=l0["min_s"],
    )
    return [UnitSpan(start, end) for start, end in windows]


# --------------------------------------------------------------------------
# Pass driver
# --------------------------------------------------------------------------


def segment_video(conn, video_id: str, config: dict[str, Any]) -> None:
    """Build one video's L0 and L1 units."""
    settings = config["p2"]
    strategy = settings["strategy"]
    if strategy not in IMPLEMENTED_STRATEGIES:
        raise NotImplementedError(
            f"segmentation strategy {strategy!r} is not built yet; "
            f"M5 adds it. Available now: {sorted(IMPLEMENTED_STRATEGIES)}"
        )

    with db.stage_run(conn, video_id, PASS_NAME, config_hash(settings)) as run:
        video = conn.execute(
            "SELECT duration_s, chapters FROM videos WHERE video_id = ?", (video_id,)
        ).fetchone()
        duration_s = video["duration_s"] or 0.0
        if duration_s <= 0:
            log.warning("%s: no duration, cannot segment", video_id)
            run.rows_out = 0
            return

        sentence_texts = [
            row["text"]
            for row in conn.execute(
                "SELECT text FROM sentences WHERE video_id = ? ORDER BY t_start",
                (video_id,),
            )
        ]
        shots = [
            (row["t_start"], row["t_end"])
            for row in conn.execute(
                "SELECT t_start, t_end FROM shots WHERE video_id = ? ORDER BY t_start",
                (video_id,),
            )
        ]

        # Re-segmenting replaces; units are derived and a config change is
        # exactly the case that must not leave two generations behind.
        conn.execute("DELETE FROM units WHERE video_id = ?", (video_id,))

        scene_spans = chapter_windows(
            json.loads(video["chapters"]) if video["chapters"] else [],
            duration_s,
            min_s=settings["l1"]["min_s"],
            max_s=settings["l1"]["max_s"],
        )
        db.insert(
            conn,
            [
                Unit(
                    video_id=video_id,
                    level=LEVEL_SCENE,
                    t_start=start,
                    t_end=end,
                    strategy=strategy,
                )
                for start, end in scene_spans
            ],
        )
        scenes = conn.execute(
            "SELECT id, t_start, t_end FROM units "
            "WHERE video_id = ? AND level = ? ORDER BY t_start",
            (video_id, LEVEL_SCENE),
        ).fetchall()

        clip_spans = build_l0_spans(sentence_texts, shots, duration_s, settings)
        db.insert(
            conn,
            [
                Unit(
                    video_id=video_id,
                    level=LEVEL_CLIP,
                    t_start=span.t_start,
                    t_end=span.t_end,
                    strategy=strategy,
                    parent_id=find_parent_id(scenes, span),
                    flags=json.dumps(list(span.flags)) if span.flags else None,
                )
                for span in clip_spans
            ],
        )

        log.info(
            "%s: %d clips, %d scenes (%s)",
            video_id, len(clip_spans), len(scene_spans), strategy,
        )
        run.rows_out = len(clip_spans) + len(scene_spans)


def find_parent_id(scenes: list[Any], span: UnitSpan) -> int | None:
    """The L1 scene containing this clip's midpoint, if any.

    Midpoint rather than overlap, because a clip straddling a scene boundary
    would otherwise match two parents and there is only one column.
    """
    midpoint = (span.t_start + span.t_end) / 2
    for scene in scenes:
        if scene["t_start"] <= midpoint < scene["t_end"]:
            return scene["id"]
    return None


def run_corpus(conn, config: dict[str, Any]) -> list[tuple[str, str]]:
    """Segment every video that needs it. Returns (video_id, error) for failures."""
    cfg_hash = config_hash(config["p2"])
    video_ids = [
        row["video_id"]
        for row in conn.execute("SELECT video_id FROM videos ORDER BY video_id")
    ]
    pending = [
        video_id
        for video_id in video_ids
        if not db.stage_done(conn, video_id, PASS_NAME, cfg_hash)
    ]
    if not pending:
        log.info("pass 2: nothing to segment")
        return []

    failures = []
    for video_id in pending:
        try:
            segment_video(conn, video_id, config)
        except Exception as exc:
            log.exception("segmentation failed for %s", video_id)
            failures.append((video_id, f"{type(exc).__name__}: {exc}"))
    log.info("pass 2 complete, %d failures", len(failures))
    return failures
