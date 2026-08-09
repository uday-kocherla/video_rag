"""Pass 1 — speech.

Consumes: `derived/{video_id}/audio.wav`.
Produces: `words` and `sentences` rows.

Run as two corpus-wide stages rather than one loop, because of the VRAM
exclusivity invariant. The naive shape — load Whisper, transcribe a video,
unload, load the aligner, align, unload, next video — reloads each model once
per video, which at ~45 s a load is most of an hour of pure loading across a
30-video corpus.

    stage A (p1_asr)   : load Whisper once  -> transcribe every video -> unload
    stage B (p1_align) : load aligner once  -> realign every video    -> unload

Each stage tracks itself in `stage_runs`, so a crash in stage B costs only stage
B. Stage A writes usable rows on its own — with Whisper's coarser timings — so
the M3 transcript baseline is not blocked on alignment existing.

Stage B reads only `words` rows and the audio file. It deliberately does not
reuse Whisper's segment boundaries, which is what lets it be re-run alone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vrag import db, sync, vram
from vrag.models import aligner as aligner_model
from vrag.models import whisper
from vrag.models.whisper import TranscribedSegment, TranscribedWord
from vrag.passes.p0_ingest import AUDIO_FILENAME
from vrag.schema import Sentence, Word

log = logging.getLogger(__name__)

PASS_ASR = "p1_asr"
PASS_ALIGN = "p1_align"

# A sentence ends here. Trailing quotes and brackets are stripped first, so
# `he said."` counts.
TERMINAL_PUNCTUATION = (".", "!", "?", "…")
CLOSING_CHARACTERS = "\"'”’)]»"


@dataclass(frozen=True, slots=True)
class SentenceSpan:
    """A sentence and the span of audio it covers."""

    text: str
    t_start: float
    t_end: float


def asr_config_hash(settings: dict[str, Any]) -> str:
    """Skip key for stage A. Excludes the aligner block on purpose.

    Tuning the aligner must not invalidate transcription — re-running ASR over
    the corpus costs hours of T4 time that the change did not require.
    """
    return db.config_hash({"asr": settings["asr"], "sentence": settings["sentence"]})


def align_config_hash(settings: dict[str, Any]) -> str:
    """Skip key for stage B."""
    return db.config_hash({"align": settings["align"], "sentence": settings["sentence"]})


def audio_path(local_dir: str | Path, video_id: str) -> Path:
    return sync.derived_dir(local_dir, video_id) / AUDIO_FILENAME


def keep_segment(
    segment: TranscribedSegment, *, no_speech_threshold: float, min_avg_logprob: float
) -> bool:
    """Decide whether a segment is real speech.

    Whisper's failure mode on silence is not silence — it is confident,
    fluent fabrication ("Thank you for watching!") over a soundless screen
    recording. This corpus contains a silent demo deliberately, so ungated
    output would poison the transcript index with text nobody said.

    `no_speech_prob` catches the obvious cases and `avg_logprob` catches the
    rest: hallucinated text scores poorly even when the model is not aware it
    is guessing.
    """
    if segment.no_speech_prob > no_speech_threshold:
        return False
    if segment.avg_logprob < min_avg_logprob:
        return False
    return bool(segment.text.strip())


def split_into_sentences(
    words: list[TranscribedWord], *, max_words: int, max_seconds: float
) -> list[SentenceSpan]:
    """Group words into sentences, using punctuation where it exists.

    Whisper punctuates, but not reliably — on fast or noisy speech it can emit
    long unbroken runs. The length caps are the fallback, so a sentence is
    always bounded even with no punctuation at all. Pass 2 snaps segment
    boundaries to these, so an unbounded "sentence" would mean no snap targets
    across a whole stretch of video.
    """
    sentences: list[SentenceSpan] = []
    current: list[TranscribedWord] = []

    for word in words:
        current.append(word)
        text = word.text.strip().rstrip(CLOSING_CHARACTERS)
        ends_sentence = text.endswith(TERMINAL_PUNCTUATION)
        too_long = (
            len(current) >= max_words
            or current[-1].t_end - current[0].t_start >= max_seconds
        )
        if ends_sentence or too_long:
            sentences.append(_sentence_from(current))
            current = []

    if current:
        sentences.append(_sentence_from(current))
    return sentences


def _sentence_from(words: list[TranscribedWord]) -> SentenceSpan:
    text = " ".join(word.text.strip() for word in words).strip()
    return SentenceSpan(text=text, t_start=words[0].t_start, t_end=words[-1].t_end)


def group_words_into_windows(
    words: list[TranscribedWord], *, max_window_s: float
) -> list[list[int]]:
    """Split word indices into consecutive windows short enough to align at once.

    Emissions are frames x vocab floats, so a whole video does not fit. Windows
    are cut on word boundaries and derived from the words themselves rather than
    from Whisper's segments, which keeps stage B independent of stage A.
    """
    windows: list[list[int]] = []
    current: list[int] = []
    window_start = 0.0

    for index, word in enumerate(words):
        if current and word.t_end - window_start > max_window_s:
            windows.append(current)
            current = []
        if not current:
            window_start = word.t_start
        current.append(index)

    if current:
        windows.append(current)
    return windows


# --------------------------------------------------------------------------
# Stage A — transcription
# --------------------------------------------------------------------------


def transcribe_video(
    conn, model: Any, video_id: str, config: dict[str, Any], local_dir: str | Path
) -> None:
    """Transcribe one video into `words` and `sentences` rows."""
    settings = config["p1"]
    asr = settings["asr"]

    with db.stage_run(conn, video_id, PASS_ASR, asr_config_hash(settings)) as run:
        path = audio_path(local_dir, video_id)
        if not path.exists():
            # Pass 0 writes no wav for a source with no audio stream. That is a
            # completed stage with nothing in it, not a failure.
            log.info("%s: no audio track, nothing to transcribe", video_id)
            run.rows_out = 0
            return

        segments = whisper.transcribe(
            model,
            path,
            language=asr["language"],
            beam_size=asr["beam_size"],
            vad_filter=asr["vad_filter"],
            vad_min_silence_ms=asr["vad_min_silence_ms"],
        )
        kept = [
            segment
            for segment in segments
            if keep_segment(
                segment,
                no_speech_threshold=asr["no_speech_threshold"],
                min_avg_logprob=asr["min_avg_logprob"],
            )
        ]
        if len(kept) < len(segments):
            log.info(
                "%s: dropped %d/%d segments as silence or hallucination",
                video_id, len(segments) - len(kept), len(segments),
            )

        words = [word for segment in kept for word in segment.words]
        sentences = split_into_sentences(
            words,
            max_words=settings["sentence"]["max_words"],
            max_seconds=settings["sentence"]["max_seconds"],
        )
        log.info("%s: %d words, %d sentences", video_id, len(words), len(sentences))

        run.rows_out = db.insert(conn, _word_rows(video_id, words)) + db.insert(
            conn, _sentence_rows(video_id, sentences)
        )


def _word_rows(video_id: str, words: list[TranscribedWord]) -> list[Word]:
    return [
        Word(
            video_id=video_id,
            t_start=word.t_start,
            t_end=word.t_end,
            text=word.text.strip(),
            conf=word.conf,
        )
        for word in words
    ]


def _sentence_rows(video_id: str, sentences: list[SentenceSpan]) -> list[Sentence]:
    return [
        Sentence(
            video_id=video_id,
            t_start=sentence.t_start,
            t_end=sentence.t_end,
            text=sentence.text,
        )
        for sentence in sentences
    ]


def transcribe_corpus(
    conn, config: dict[str, Any], local_dir: str | Path
) -> list[tuple[str, str]]:
    """Stage A over every ingested video. Returns (video_id, error) for failures."""
    settings = config["p1"]
    pending = _pending_videos(conn, PASS_ASR, asr_config_hash(settings))
    if not pending:
        log.info("stage A: nothing to transcribe")
        return []

    log.info("stage A: transcribing %d videos", len(pending))
    failures = []
    with vram.loaded(
        whisper.load(settings["asr"]["model"], compute_type=settings["asr"]["compute_type"])
    ) as model:
        for video_id in pending:
            try:
                transcribe_video(conn, model, video_id, config, local_dir)
            except Exception as exc:
                log.exception("transcription failed for %s", video_id)
                failures.append((video_id, f"{type(exc).__name__}: {exc}"))
    return failures


# --------------------------------------------------------------------------
# Stage B — forced alignment
# --------------------------------------------------------------------------


def align_video(
    conn, aligner: Any, video_id: str, config: dict[str, Any], local_dir: str | Path
) -> None:
    """Replace one video's word timings with forced-aligned ones."""
    settings = config["p1"]
    align = settings["align"]

    with db.stage_run(conn, video_id, PASS_ALIGN, align_config_hash(settings)) as run:
        rows = conn.execute(
            "SELECT id, text, t_start, t_end FROM words WHERE video_id = ? ORDER BY id",
            (video_id,),
        ).fetchall()
        path = audio_path(local_dir, video_id)
        if not rows or not path.exists():
            log.info("%s: no words to align", video_id)
            run.rows_out = 0
            return

        words = [
            TranscribedWord(text=row["text"], t_start=row["t_start"], t_end=row["t_end"])
            for row in rows
        ]
        waveform = aligner_model.load_waveform(path, target_sample_rate=aligner.sample_rate)

        updates = []
        for window in group_words_into_windows(words, max_window_s=align["window_s"]):
            updates.extend(
                _align_window(aligner, waveform, words, rows, window, pad_s=align["pad_s"])
            )

        conn.executemany("UPDATE words SET t_start = ?, t_end = ? WHERE id = ?", updates)
        log.info("%s: realigned %d/%d words", video_id, len(updates), len(words))

        # Sentence boundaries are derived from word times, so they are stale now.
        conn.execute("DELETE FROM sentences WHERE video_id = ?", (video_id,))
        realigned = [
            TranscribedWord(text=row["text"], t_start=row["t_start"], t_end=row["t_end"])
            for row in conn.execute(
                "SELECT text, t_start, t_end FROM words WHERE video_id = ? ORDER BY id",
                (video_id,),
            )
        ]
        sentences = split_into_sentences(
            realigned,
            max_words=settings["sentence"]["max_words"],
            max_seconds=settings["sentence"]["max_seconds"],
        )
        db.insert(conn, _sentence_rows(video_id, sentences))
        run.rows_out = len(updates)


def _align_window(
    aligner: Any,
    waveform: Any,
    words: list[TranscribedWord],
    rows: list[Any],
    window: list[int],
    *,
    pad_s: float,
) -> list[tuple[float, float, int]]:
    """Align one window, returning (t_start, t_end, word_id) updates.

    The slice is padded because the window bounds come from Whisper's timings,
    which are the very thing being corrected — a word can start slightly before
    Whisper thinks it does, and clipping it makes the window unalignable.
    """
    sample_rate = aligner.sample_rate
    total_samples = waveform.size(1)
    start_sample = max(0, int((words[window[0]].t_start - pad_s) * sample_rate))
    end_sample = min(total_samples, int((words[window[-1]].t_end + pad_s) * sample_rate))
    if end_sample <= start_sample:
        return []

    spans = aligner_model.align_words(
        aligner,
        waveform[:, start_sample:end_sample],
        [words[index].text for index in window],
    )

    offset = start_sample / sample_rate
    updates = []
    for index, span in zip(window, spans):
        if span is not None:
            updates.append((offset + span[0], offset + span[1], rows[index]["id"]))
    return updates


def align_corpus(
    conn, config: dict[str, Any], local_dir: str | Path
) -> list[tuple[str, str]]:
    """Stage B over every video that has words. Returns (video_id, error) for failures."""
    settings = config["p1"]
    pending = _pending_videos(conn, PASS_ALIGN, align_config_hash(settings))
    if not pending:
        log.info("stage B: nothing to align")
        return []

    log.info("stage B: aligning %d videos", len(pending))
    failures = []
    with vram.loaded(aligner_model.load(settings["align"]["model"])) as aligner:
        for video_id in pending:
            try:
                align_video(conn, aligner, video_id, config, local_dir)
            except Exception as exc:
                log.exception("alignment failed for %s", video_id)
                failures.append((video_id, f"{type(exc).__name__}: {exc}"))
    return failures


def _pending_videos(conn, pass_name: str, cfg_hash: str) -> list[str]:
    """Videos still needing this stage.

    Checked before the model is loaded, so a fully-completed stage costs nothing
    rather than 45 seconds of loading a model it will not use.
    """
    video_ids = [
        row["video_id"]
        for row in conn.execute("SELECT video_id FROM videos ORDER BY video_id")
    ]
    return [
        video_id
        for video_id in video_ids
        if not db.stage_done(conn, video_id, pass_name, cfg_hash)
    ]


def run_corpus(
    conn, config: dict[str, Any], local_dir: str | Path
) -> list[tuple[str, str]]:
    """Both stages, in order. Never holds two models at once."""
    failures = transcribe_corpus(conn, config, local_dir)
    failures += align_corpus(conn, config, local_dir)
    log.info("pass 1 complete, %d failures", len(failures))
    return failures
