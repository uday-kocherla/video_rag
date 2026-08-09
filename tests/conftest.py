"""Shared fixtures. No network anywhere in the test suite.

The synthetic clips are built with ffmpeg rather than committed as binaries, so
the suite stays text-only and still exercises the real ffmpeg/PySceneDetect path
on any machine that has ffmpeg. Machines without it skip those tests.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")

CLIP_SECONDS = 2  # per half, so every clip below is 4 seconds long
CLIP_FPS = 10
CLIP_SIZE = "320x240"


def _build_clip(path, *, with_audio: bool) -> None:
    """Render a 4-second clip that cuts from black to white halfway through.

    The cut is deliberately extreme so ContentDetector finds it at any sane
    threshold — this fixture is testing our plumbing, not tuning detection.
    """
    half = f"s={CLIP_SIZE}:d={CLIP_SECONDS}:r={CLIP_FPS}"
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"color=c=black:{half}",
        "-f", "lavfi", "-i", f"color=c=white:{half}",
    ]
    if with_audio:
        command += [
            "-f", "lavfi",
            "-i", f"sine=frequency=440:duration={CLIP_SECONDS * 2}",
        ]
    command += [
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1[v]",
        "-map", "[v]",
    ]
    if with_audio:
        command += ["-map", "2:a", "-c:a", "aac"]
    command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(CLIP_FPS), str(path)]
    subprocess.run(command, check=True, capture_output=True)


@pytest.fixture(scope="session")
def clip(tmp_path_factory):
    """A 4-second clip with one hard cut and an audio track."""
    path = tmp_path_factory.mktemp("clips") / "clip.mp4"
    _build_clip(path, with_audio=True)
    return path


@pytest.fixture(scope="session")
def silent_clip(tmp_path_factory):
    """The same clip with no audio stream — a real case in this corpus."""
    path = tmp_path_factory.mktemp("clips") / "silent.mp4"
    _build_clip(path, with_audio=False)
    return path
