"""Pass 0 — ingest.

Consumes: a list of URLs.
Produces: `videos` and `shots` rows, plus `derived/{video_id}/{audio.wav,proxy.mp4}`.

The order of operations here is load-bearing:

1. Metadata first, download second. We need the video ID to answer "already
   ingested?", and downloading 400 MB in order to then skip it is the dumb version.
2. Shot detection runs on the downloaded original at its native frame rate,
   before that file is deleted. The 2 fps proxy is far too coarse — boundaries
   would land within half a second and fast cuts would vanish entirely.
3. The original is deleted at the end. It is regenerable from its URL; the
   derived files are not, and raw video never leaves ephemeral local disk.
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any

import yt_dlp
from scenedetect import ContentDetector, SceneManager, open_video

from vrag import db, media, sync
from vrag.schema import Shot, Video

log = logging.getLogger(__name__)

PASS_NAME = "p0_ingest"

AUDIO_FILENAME = "audio.wav"
PROXY_FILENAME = "proxy.mp4"


def build_ydl_options(cookies_file: str | None, **overrides: Any) -> dict[str, Any]:
    """Common yt-dlp options, shared by the metadata and download calls.

    Both calls hit YouTube and both get rejected without cookies, so the auth
    plumbing lives in one place rather than being duplicated and then drifting.
    """
    options: dict[str, Any] = {"quiet": True, "no_warnings": True}
    if cookies_file:
        options["cookiefile"] = str(cookies_file)
    options.update(overrides)
    return options


def fetch_metadata(url: str, *, cookies_file: str | None = None) -> dict[str, Any]:
    """Read a video's metadata without downloading it."""
    options = build_ydl_options(cookies_file, skip_download=True)
    with yt_dlp.YoutubeDL(options) as ydl:
        return ydl.extract_info(url, download=False)


def download_video(
    url: str,
    dest_dir: Path,
    *,
    max_height: int,
    cookies_file: str | None = None,
) -> Path:
    """Download the video into `dest_dir` and return the resulting file path."""
    options = build_ydl_options(
        cookies_file,
        format=(
            f"bestvideo[height<=?{max_height}]+bestaudio/"
            f"best[height<=?{max_height}]/best"
        ),
        merge_output_format="mp4",
        outtmpl=str(dest_dir / "%(id)s.%(ext)s"),
    )
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
    return Path(info["requested_downloads"][0]["filepath"])


def detect_scene_boundaries(
    video_path: Path,
    *,
    threshold: float,
    min_len_frames: int,
    downscale: int | None,
) -> list[tuple[float, float]]:
    """Run PySceneDetect and return scene boundaries as (start, end) seconds."""
    video = open_video(str(video_path))
    manager = SceneManager()
    manager.add_detector(
        ContentDetector(threshold=threshold, min_scene_len=min_len_frames)
    )
    if downscale is not None:
        manager.auto_downscale = False
        manager.downscale = downscale

    manager.detect_scenes(video, show_progress=False)
    return [(start.seconds, end.seconds) for start, end in manager.get_scene_list()]


def spans_from_scene_boundaries(
    boundaries: list[tuple[float, float]], duration_s: float
) -> list[tuple[float, float]]:
    """Turn detected scene boundaries into contiguous shots covering the video.

    An empty boundary list is a legitimate result, not a failure: a single-shot
    lecture genuinely has no cuts. Returning one span for the whole video keeps
    that fact *in the data*, where pass 2's single-shot fallback can see it,
    rather than writing zero rows and making it look like detection never ran.

    Nothing is merged or filtered here. A rapid-cut montage really does contain
    hundreds of sub-second shots, and that is pass 2's problem to smooth over —
    pass 0 records what is there.
    """
    if duration_s <= 0:
        return []
    if not boundaries:
        return [(0.0, duration_s)]

    spans = []
    for start, end in boundaries:
        start = min(max(start, 0.0), duration_s)
        end = min(max(end, 0.0), duration_s)
        if end > start:
            spans.append((start, end))

    if not spans:
        return [(0.0, duration_s)]

    # Detection can stop a frame or two short at either edge; never leave a gap,
    # or the units built on top of these shots would have holes in them.
    spans[0] = (0.0, spans[0][1])
    spans[-1] = (spans[-1][0], duration_s)
    return spans


def ingest_video(
    conn,
    url: str,
    config: dict[str, Any],
    local_dir: str | Path,
) -> str | None:
    """Ingest one video. Returns its ID, or None if it was already done."""
    settings = config["p0"]
    cookies_file = config.get("cookies_file")
    cfg_hash = db.config_hash(settings)

    info = fetch_metadata(url, cookies_file=cookies_file)
    video_id = info["id"]
    if db.stage_done(conn, video_id, PASS_NAME, cfg_hash):
        log.info("%s: already ingested, skipping", video_id)
        return None

    log.info("%s: ingesting %r", video_id, info.get("title"))
    output_dir = sync.derived_dir(local_dir, video_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    # The original lives only for the duration of this block. Shot detection has
    # to happen inside it, at native frame rate, while the file still exists.
    with tempfile.TemporaryDirectory(prefix=f"vrag-{video_id}-") as workdir:
        source = download_video(
            url,
            Path(workdir),
            max_height=settings["max_height"],
            cookies_file=cookies_file,
        )
        media_info = media.probe(
            source, vfr_tolerance_fps=settings["vfr_tolerance_fps"]
        )

        if media_info.is_variable_frame_rate:
            log.warning(
                "%s: variable frame rate — shot timestamps derived from frame "
                "index may drift. Flagged as is_vfr in the videos table.",
                video_id,
            )
        if not media_info.has_audio:
            log.warning("%s: no audio stream, pass 1 will have nothing to do", video_id)

        boundaries = detect_scene_boundaries(
            source,
            threshold=settings["scene_threshold"],
            min_len_frames=settings["scene_min_len_frames"],
            downscale=settings["scene_downscale"],
        )
        spans = spans_from_scene_boundaries(boundaries, media_info.duration_s)
        log.info("%s: %d shots over %.1fs", video_id, len(spans), media_info.duration_s)

        media.extract_audio_and_proxy(
            source,
            audio_path=output_dir / AUDIO_FILENAME if media_info.has_audio else None,
            proxy_path=output_dir / PROXY_FILENAME,
            sample_rate=settings["audio_sample_rate"],
            proxy_fps=settings["proxy_fps"],
            proxy_height=settings["proxy_height"],
            proxy_crf=settings["proxy_crf"],
        )

    chapters = info.get("chapters")
    video_row = Video(
        video_id=video_id,
        url=info.get("webpage_url") or url,
        added_at=time.time(),
        title=info.get("title"),
        duration_s=media_info.duration_s,
        fps=media_info.fps,
        width=media_info.width,
        height=media_info.height,
        chapters=json.dumps(chapters) if chapters else None,
        is_vfr=int(media_info.is_variable_frame_rate),
    )
    shot_rows = [
        Shot(video_id=video_id, t_start=start, t_end=end) for start, end in spans
    ]

    with db.stage_run(conn, video_id, PASS_NAME, cfg_hash) as run:
        run.rows_out = db.insert(conn, [video_row]) + db.insert(conn, shot_rows)

    return video_id


def run_corpus(
    conn,
    urls: list[str],
    config: dict[str, Any],
    local_dir: str | Path,
) -> list[tuple[str, str]]:
    """Ingest every URL, returning (url, error) for the ones that failed.

    One bad video must never take down a corpus run, so failures are caught per
    item. They are already recorded in stage_runs by then; the return value is
    just so the caller can print a summary without querying.
    """
    failures = []
    for url in urls:
        try:
            ingest_video(conn, url, config, local_dir)
        except Exception as exc:
            log.exception("ingest failed for %s", url)
            failures.append((url, f"{type(exc).__name__}: {exc}"))
    log.info("ingest complete: %d urls, %d failed", len(urls), len(failures))
    return failures
