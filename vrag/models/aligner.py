"""Forced alignment with a wav2vec2 CTC model, for pass 1 stage B.

Whisper's word timestamps come from cross-attention DTW and land within roughly
100-200 ms. That is fine for reading a transcript and much too coarse for
snapping segment boundaries in pass 2, where the error propagates straight into
tIoU. Forced alignment gets to 20-30 ms.

How it works, since this is the part worth understanding:

The acoustic model emits, for every ~20 ms frame, a probability distribution
over characters. Normally you would decode that into text. Here we already know
the text, so instead we search for the most likely *alignment* of the known
character sequence to the frames — a Viterbi path through the emission matrix
that is forced to emit the transcript in order, with CTC blanks allowed between
and within characters. `torchaudio.functional.forced_align` runs that Viterbi.
Everything else in this file is getting text into it and timings back out.

Alignment is done on short windows rather than a whole video: emissions are
frames x vocab floats, and a 50-minute file would not fit in memory.
"""

from __future__ import annotations

import logging
import re
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# The MMS_FA dictionary covers lowercase latin letters plus apostrophe. Anything
# else — digits, punctuation, symbols — has no acoustic token and is dropped.
_KEEPABLE = re.compile(r"[^a-z']")


@dataclass(frozen=True, slots=True)
class Aligner:
    """A loaded acoustic model plus the token dictionary it was trained with."""

    model: Any
    dictionary: dict[str, int]
    sample_rate: int


def load(pipeline_name: str) -> Aligner:
    """Load a torchaudio forced-alignment pipeline by name, e.g. "MMS_FA".

    `with_star=False` / `star=None` drop the wildcard token: it exists to absorb
    out-of-vocabulary audio, and we would rather see a window fail to align and
    fall back to Whisper's timings than silently absorb a mismatch.
    """
    import torchaudio

    bundle = getattr(torchaudio.pipelines, pipeline_name)
    log.info("loading aligner %s (%d Hz)", pipeline_name, bundle.sample_rate)
    return Aligner(
        model=bundle.get_model(with_star=False),
        dictionary=bundle.get_dict(star=None),
        sample_rate=bundle.sample_rate,
    )


def normalize_word(word: str) -> str:
    """Reduce a transcript word to characters the acoustic model knows.

    "Don't!" becomes "don't"; "2024" and "—" become empty, meaning the word
    cannot be aligned and keeps whatever timing it already had.
    """
    return _KEEPABLE.sub("", word.strip().lower())


def load_waveform(path: str | Path, *, target_sample_rate: int):
    """Read a PCM wav file as a mono waveform at the aligner's sample rate.

    Deliberately not `torchaudio.load`: as of torchaudio 2.11 that delegates to
    TorchCodec, a separate install that has to match the torch build. Pass 0
    always writes 16-bit PCM at the aligner's own sample rate, so the stdlib
    `wave` module reads it with no dependency at all and the resample below is
    normally dead code.
    """
    import torch

    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width != 2:
        raise ValueError(f"expected 16-bit PCM audio, got {sample_width * 8}-bit: {path}")

    samples = torch.frombuffer(bytearray(frames), dtype=torch.int16).float() / 32768.0
    waveform = samples.view(-1, channels).t().contiguous()
    if channels > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != target_sample_rate:
        import torchaudio

        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
    return waveform


def encode_words(
    words: list[str], dictionary: dict[str, int]
) -> tuple[list[int], list[tuple[int, int]]]:
    """Turn words into a flat token sequence plus each word's slice of it.

    Returns (token_ids, spans) where spans[k] is the (word_index, token_count)
    of the k-th alignable word. Words that normalize to nothing are absent from
    `spans` entirely, which is how the caller knows to leave their timing alone.
    """
    token_ids: list[int] = []
    spans: list[tuple[int, int]] = []
    for index, word in enumerate(words):
        ids = [dictionary[char] for char in normalize_word(word) if char in dictionary]
        if ids:
            spans.append((index, len(ids)))
            token_ids.extend(ids)
    return token_ids, spans


def align_words(
    aligner: Aligner, waveform: Any, words: list[str]
) -> list[tuple[float, float] | None]:
    """Align `words` against `waveform`, returning one span per input word.

    An entry is None when that word could not be aligned — it normalized to
    nothing, or the whole window failed — and the caller keeps the existing
    timestamp for it. Returning None rather than raising matters: one unalignable
    window in a 50-minute lecture must not cost the other 49 minutes.
    """
    import torch
    import torchaudio.functional as F

    token_ids, word_spans = encode_words(words, aligner.dictionary)
    if not token_ids:
        return [None] * len(words)

    with torch.inference_mode():
        emission, _ = aligner.model(waveform)

    # forced_align needs at least one frame per token; a clipped window can
    # violate that, and there is nothing sensible to do but fall back.
    if emission.size(1) < len(token_ids):
        log.debug(
            "window too short to align: %d frames for %d tokens",
            emission.size(1), len(token_ids),
        )
        return [None] * len(words)

    targets = torch.tensor([token_ids], dtype=torch.int32, device=emission.device)
    aligned_tokens, scores = F.forced_align(emission, targets, blank=0)
    token_spans = F.merge_tokens(aligned_tokens[0], scores[0])

    # One emission frame covers this many seconds of audio.
    seconds_per_frame = waveform.size(1) / emission.size(1) / aligner.sample_rate

    result: list[tuple[float, float] | None] = [None] * len(words)
    cursor = 0
    for word_index, token_count in word_spans:
        spans = token_spans[cursor:cursor + token_count]
        cursor += token_count
        if spans:
            result[word_index] = (
                spans[0].start * seconds_per_frame,
                spans[-1].end * seconds_per_frame,
            )
    return result
