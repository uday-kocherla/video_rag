"""ffprobe and ffmpeg wrappers.

Kept separate from the passes because pass 3 needs to pull keyframes out of the
same proxy files pass 0 writes, and neither pass should be assembling command
lines inline.
"""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MediaInfo:
    """What we need to know about a media file before deriving anything from it."""

    duration_s: float
    fps: float
    width: int
    height: int
    has_audio: bool
    is_variable_frame_rate: bool


def run_command(command: list[str]) -> str:
    """Run a subprocess, returning stdout and raising with stderr on failure.

    ffmpeg writes its actual error to stderr and exits non-zero, so the default
    CalledProcessError message ("returned non-zero exit status 1") tells you
    nothing. Carrying stderr into the exception is the whole point of this.
    """
    log.debug("running: %s", shlex.join(command))
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        tail = result.stderr.strip()[-2000:]
        raise RuntimeError(f"{command[0]} failed (exit {result.returncode}):\n{tail}")
    return result.stdout


def parse_frame_rate(value: str | None) -> float:
    """Parse an ffprobe frame rate, which is a rational string like '30000/1001'."""
    if not value:
        return 0.0
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        if float(denominator) == 0:
            return 0.0
        return float(numerator) / float(denominator)
    return float(value)


def media_info_from_probe(probe_output: dict, *, vfr_tolerance_fps: float) -> MediaInfo:
    """Build MediaInfo from parsed `ffprobe -show_streams -show_format` JSON.

    Split out from `probe` so the parsing — including the variable-frame-rate
    comparison — is testable without a media file on disk.

    `r_frame_rate` is the rate the stream declares; `avg_frame_rate` is the rate
    it actually achieved. When they disagree the file is variable-frame-rate and
    `frame_index / fps` is no longer a valid clock, which quietly corrupts every
    shot boundary derived from it.
    """
    streams = probe_output.get("streams", [])
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    if not video_streams:
        raise ValueError("file has no video stream")
    video = video_streams[0]

    declared_fps = parse_frame_rate(video.get("r_frame_rate"))
    average_fps = parse_frame_rate(video.get("avg_frame_rate"))
    is_vfr = (
        declared_fps > 0
        and average_fps > 0
        and abs(declared_fps - average_fps) > vfr_tolerance_fps
    )

    duration = video.get("duration") or probe_output.get("format", {}).get("duration")

    return MediaInfo(
        duration_s=float(duration) if duration else 0.0,
        fps=average_fps or declared_fps,
        width=int(video.get("width", 0)),
        height=int(video.get("height", 0)),
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
        is_variable_frame_rate=is_vfr,
    )


def probe(path: str | Path, *, vfr_tolerance_fps: float) -> MediaInfo:
    """Inspect a media file with ffprobe."""
    output = run_command([
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        str(path),
    ])
    return media_info_from_probe(json.loads(output), vfr_tolerance_fps=vfr_tolerance_fps)


def build_derive_command(
    source: Path,
    *,
    audio_path: Path | None,
    proxy_path: Path,
    sample_rate: int,
    proxy_fps: int,
    proxy_height: int,
    proxy_crf: int,
) -> list[str]:
    """Build the single ffmpeg call that writes both derived files.

    One invocation with two outputs, so the source is decoded once instead of
    twice. `audio_path` is None for a silent video, which is a real case in this
    corpus, not an error.

    `-fps_mode cfr` is what makes the proxy constant-frame-rate. Without it a
    dropped frame shifts every later timestamp and pass 3's
    `frame_index / proxy_fps` silently drifts.
    """
    command = ["ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-i", str(source)]

    if audio_path is not None:
        command += [
            "-vn",
            "-ac", "1",
            "-ar", str(sample_rate),
            "-c:a", "pcm_s16le",
            str(audio_path),
        ]

    # Drop frames before scaling — scaling 2 fps is far cheaper than scaling 30.
    command += [
        "-an",
        "-vf", f"fps={proxy_fps},scale=-2:{proxy_height}",
        "-fps_mode", "cfr",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", str(proxy_crf),
        "-pix_fmt", "yuv420p",
        str(proxy_path),
    ]
    return command


def extract_audio_and_proxy(
    source: Path,
    *,
    audio_path: Path | None,
    proxy_path: Path,
    sample_rate: int,
    proxy_fps: int,
    proxy_height: int,
    proxy_crf: int,
) -> None:
    """Write the 16 kHz mono wav and the constant-frame-rate proxy."""
    command = build_derive_command(
        source,
        audio_path=audio_path,
        proxy_path=proxy_path,
        sample_rate=sample_rate,
        proxy_fps=proxy_fps,
        proxy_height=proxy_height,
        proxy_crf=proxy_crf,
    )
    run_command(command)
