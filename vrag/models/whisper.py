"""faster-whisper (CTranslate2) wrapper for pass 1 stage A.

CTranslate2 rather than transformers because it is several times faster at the
same quality and runs fp16 on a T4 — no bf16 anywhere, which the hardware does
not support.

`faster_whisper` is imported inside `load` rather than at module scope so that
the pure logic in p1_speech can be imported and tested on a machine with no
inference stack installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TranscribedWord:
    """One word with the timing Whisper guessed for it."""

    text: str
    t_start: float
    t_end: float
    conf: float | None = None


@dataclass(frozen=True, slots=True)
class TranscribedSegment:
    """A Whisper segment, kept only so the pass can gate on its quality scores.

    These are plain dataclasses rather than faster-whisper's own types so that
    the hallucination gating and sentence splitting in p1_speech are testable
    without a model, a GPU, or the library installed.
    """

    t_start: float
    t_end: float
    text: str
    no_speech_prob: float
    avg_logprob: float
    words: list[TranscribedWord]


def resolve_runtime(compute_type: str) -> tuple[str, str]:
    """Pick the device and numeric precision actually available here.

    The configured compute type assumes a GPU. On a CPU-only box fp16 is not
    supported at all, so fall back to int8 and say so rather than crashing.
    """
    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None and torch.cuda.is_available():
        return "cuda", compute_type
    log.warning("no CUDA device; running Whisper on CPU with int8 instead of %s", compute_type)
    return "cpu", "int8"


def load(model_name: str, *, compute_type: str) -> Any:
    """Load a Whisper model. Release it with vram.loaded / vram.free."""
    from faster_whisper import WhisperModel

    device, resolved_compute_type = resolve_runtime(compute_type)
    log.info("loading whisper %s on %s (%s)", model_name, device, resolved_compute_type)
    return WhisperModel(model_name, device=device, compute_type=resolved_compute_type)


def transcribe(
    model: Any,
    audio_path: str | Path,
    *,
    language: str | None,
    beam_size: int,
    vad_filter: bool,
    vad_min_silence_ms: int,
) -> list[TranscribedSegment]:
    """Transcribe a wav file into segments carrying per-word timings.

    faster-whisper returns a generator that does the decoding lazily, so the
    list() here is where the GPU time is actually spent.
    """
    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=beam_size,
        word_timestamps=True,
        vad_filter=vad_filter,
        vad_parameters={"min_silence_duration_ms": vad_min_silence_ms},
    )
    log.info("detected language %s (p=%.2f)", info.language, info.language_probability)
    return [_to_segment(segment) for segment in segments]


def _to_segment(segment: Any) -> TranscribedSegment:
    words = [
        TranscribedWord(
            text=word.word,
            t_start=word.start,
            t_end=word.end,
            conf=getattr(word, "probability", None),
        )
        for word in (segment.words or [])
    ]
    return TranscribedSegment(
        t_start=segment.start,
        t_end=segment.end,
        text=segment.text,
        no_speech_prob=segment.no_speech_prob,
        avg_logprob=segment.avg_logprob,
        words=words,
    )
