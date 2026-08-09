"""Forced alignment against a synthetic emission matrix.

torchaudio's forced_align works on any tensor, so the real Viterbi runs here
without downloading MMS_FA weights — which keeps the suite offline while still
testing the part we wrote: turning words into tokens, and turning the returned
frame spans back into seconds.
"""

from __future__ import annotations

import pytest

from vrag.models.aligner import (
    Aligner,
    align_words,
    encode_words,
    load_waveform,
    normalize_word,
)
from tests.conftest import needs_ffmpeg, needs_torch

# Minimal dictionary: blank plus two letters, in the layout torchaudio expects
# (blank at index 0).
DICTIONARY = {"-": 0, "a": 1, "b": 2}

SAMPLE_RATE = 16000
FRAMES = 20
SAMPLES_PER_FRAME = 320  # so the clip is 0.4s and each frame is exactly 0.02s


def test_normalize_word_keeps_only_alignable_characters():
    assert normalize_word("Don't!") == "don't"
    assert normalize_word("  Hello,  ") == "hello"
    assert normalize_word("2024") == ""
    assert normalize_word("—") == ""


def test_encode_words_skips_unalignable_words():
    token_ids, spans = encode_words(["ab", "2024", "ba"], DICTIONARY)
    assert token_ids == [1, 2, 2, 1]
    # "2024" is absent, so the caller keeps its existing timestamp.
    assert spans == [(0, 2), (2, 2)]


def test_encode_words_drops_characters_outside_the_dictionary():
    token_ids, spans = encode_words(["a-b"], DICTIONARY)  # '-' is not alignable text
    assert token_ids == [1, 2]
    assert spans == [(0, 2)]


def test_encode_words_with_nothing_alignable():
    assert encode_words(["123", "!!!"], DICTIONARY) == ([], [])


@pytest.fixture()
def fake_aligner():
    """An Aligner whose acoustic model emits 'a' for 0.0-0.2s and 'b' for 0.2-0.4s."""
    import torch

    logits = torch.full((1, FRAMES, len(DICTIONARY)), -10.0)
    logits[0, : FRAMES // 2, DICTIONARY["a"]] = 10.0
    logits[0, FRAMES // 2 :, DICTIONARY["b"]] = 10.0
    emission = torch.log_softmax(logits, dim=-1)

    class FakeAcousticModel:
        def __call__(self, waveform):
            return emission, None

    return Aligner(
        model=FakeAcousticModel(), dictionary=DICTIONARY, sample_rate=SAMPLE_RATE
    )


@pytest.fixture()
def waveform():
    import torch

    return torch.zeros((1, FRAMES * SAMPLES_PER_FRAME))


@needs_torch
def test_words_land_on_the_frames_that_emitted_them(fake_aligner, waveform):
    spans = align_words(fake_aligner, waveform, ["a", "b"])

    assert spans[0] == pytest.approx((0.0, 0.2), abs=0.02)
    assert spans[1] == pytest.approx((0.2, 0.4), abs=0.02)


@needs_torch
def test_unalignable_word_returns_none_and_neighbours_still_align(fake_aligner, waveform):
    # The middle word keeps whatever timing it already had; the others do not
    # suffer for it.
    spans = align_words(fake_aligner, waveform, ["a", "2024", "b"])

    assert spans[1] is None
    assert spans[0] == pytest.approx((0.0, 0.2), abs=0.02)
    assert spans[2] == pytest.approx((0.2, 0.4), abs=0.02)


@needs_torch
def test_nothing_alignable_falls_back_for_every_word(fake_aligner, waveform):
    assert align_words(fake_aligner, waveform, ["123", "!!"]) == [None, None]


@needs_torch
@needs_ffmpeg
def test_load_waveform_reads_the_wav_pass_0_writes(tmp_path):
    """The exact format pass 0 produces: 16-bit PCM, mono, 16 kHz."""
    import subprocess

    path = tmp_path / "audio.wav"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(path),
    ], check=True)

    loaded = load_waveform(path, target_sample_rate=16000)
    assert loaded.shape == (1, 32000)  # mono, 2 seconds at 16 kHz
    assert loaded.abs().max() <= 1.0, "samples should be normalized to [-1, 1]"


@needs_torch
@needs_ffmpeg
def test_load_waveform_downmixes_and_resamples(tmp_path):
    import subprocess

    path = tmp_path / "stereo.wav"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
        "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", str(path),
    ], check=True)

    loaded = load_waveform(path, target_sample_rate=16000)
    assert loaded.shape[0] == 1, "stereo should be downmixed to mono"
    assert loaded.shape[1] == pytest.approx(32000, abs=100), "should resample to 16 kHz"


@needs_torch
def test_window_too_short_for_the_transcript_falls_back(fake_aligner, waveform):
    # 21 tokens cannot fit in 20 frames. One unalignable window in a long
    # lecture must not cost the rest of it.
    spans = align_words(fake_aligner, waveform, ["ab"] * 11)
    assert spans == [None] * 11
