"""ffprobe parsing (pure) and the ffmpeg derive step (real, when ffmpeg exists)."""

from __future__ import annotations

import json

import pytest

from vrag import media
from tests.conftest import needs_ffmpeg


def probe_output(*, r_frame_rate: str, avg_frame_rate: str, with_audio: bool = True) -> dict:
    streams = [{
        "codec_type": "video",
        "r_frame_rate": r_frame_rate,
        "avg_frame_rate": avg_frame_rate,
        "width": 1280,
        "height": 720,
        "duration": "12.5",
    }]
    if with_audio:
        streams.append({"codec_type": "audio", "sample_rate": "44100"})
    return {"streams": streams, "format": {"duration": "12.5"}}


def test_parse_frame_rate_handles_rationals_and_junk():
    assert media.parse_frame_rate("30/1") == 30.0
    assert media.parse_frame_rate("24") == 24.0
    assert media.parse_frame_rate("30000/1001") == pytest.approx(29.97, abs=0.01)
    assert media.parse_frame_rate("0/0") == 0.0  # ffprobe's answer for "no idea"
    assert media.parse_frame_rate(None) == 0.0


def test_constant_frame_rate_is_not_flagged():
    info = media.media_info_from_probe(
        probe_output(r_frame_rate="30/1", avg_frame_rate="30/1"), vfr_tolerance_fps=0.01
    )
    assert not info.is_variable_frame_rate
    assert info.fps == 30.0
    assert info.duration_s == 12.5
    assert (info.width, info.height) == (1280, 720)
    assert info.has_audio


def test_variable_frame_rate_is_flagged():
    # Declared 30, actually achieved 23.9 — frame_index / fps is now a lie.
    info = media.media_info_from_probe(
        probe_output(r_frame_rate="30/1", avg_frame_rate="23.9"), vfr_tolerance_fps=0.01
    )
    assert info.is_variable_frame_rate


def test_unknown_average_frame_rate_is_not_flagged():
    # "0/0" means ffprobe could not tell, which is not evidence of VFR.
    info = media.media_info_from_probe(
        probe_output(r_frame_rate="25/1", avg_frame_rate="0/0"), vfr_tolerance_fps=0.01
    )
    assert not info.is_variable_frame_rate
    assert info.fps == 25.0


def test_missing_audio_stream_is_detected():
    info = media.media_info_from_probe(
        probe_output(r_frame_rate="30/1", avg_frame_rate="30/1", with_audio=False),
        vfr_tolerance_fps=0.01,
    )
    assert not info.has_audio


def test_probe_requires_a_video_stream():
    with pytest.raises(ValueError, match="no video stream"):
        media.media_info_from_probe({"streams": [{"codec_type": "audio"}]}, vfr_tolerance_fps=0.01)


def test_silent_video_omits_the_audio_output():
    command = media.build_derive_command(
        "in.mp4", audio_path=None, proxy_path="proxy.mp4",
        sample_rate=16000, proxy_fps=2, proxy_height=360, proxy_crf=28,
    )
    assert "pcm_s16le" not in command
    assert "-fps_mode" in command and command[command.index("-fps_mode") + 1] == "cfr"


@needs_ffmpeg
def test_probe_reads_a_real_file(clip):
    info = media.probe(clip, vfr_tolerance_fps=0.01)
    assert info.duration_s == pytest.approx(4.0, abs=0.2)
    assert info.fps == pytest.approx(10.0, abs=0.1)
    assert (info.width, info.height) == (320, 240)
    assert info.has_audio
    assert not info.is_variable_frame_rate


@needs_ffmpeg
def test_derived_proxy_is_constant_2fps_and_audio_is_16k_mono(clip, tmp_path):
    audio_path = tmp_path / "audio.wav"
    proxy_path = tmp_path / "proxy.mp4"
    media.extract_audio_and_proxy(
        clip, audio_path=audio_path, proxy_path=proxy_path,
        sample_rate=16000, proxy_fps=2, proxy_height=360, proxy_crf=28,
    )

    # Exactly 2 fps over 4 seconds, so frame_index / 2.0 is an exact timestamp.
    counted = media.run_command([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=nb_read_frames", "-print_format", "json", str(proxy_path),
    ])
    assert int(json.loads(counted)["streams"][0]["nb_read_frames"]) == 8

    audio = json.loads(media.run_command([
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,channels", "-print_format", "json", str(audio_path),
    ]))["streams"][0]
    assert audio["sample_rate"] == "16000"
    assert audio["channels"] == 1


@needs_ffmpeg
def test_silent_source_still_produces_a_proxy(silent_clip, tmp_path):
    proxy_path = tmp_path / "proxy.mp4"
    media.extract_audio_and_proxy(
        silent_clip, audio_path=None, proxy_path=proxy_path,
        sample_rate=16000, proxy_fps=2, proxy_height=360, proxy_crf=28,
    )
    assert proxy_path.exists()
    assert not (tmp_path / "audio.wav").exists()


@needs_ffmpeg
def test_run_command_surfaces_ffmpeg_stderr():
    with pytest.raises(RuntimeError, match="ffprobe failed"):
        media.run_command(["ffprobe", "-v", "error", "/nonexistent/file.mp4"])
