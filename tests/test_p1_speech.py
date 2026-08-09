"""Pass 1 logic: hallucination gating, sentence splitting, alignment windows.

No models are downloaded and nothing hits the network. The forced-alignment
test in test_aligner.py feeds a synthetic emission matrix instead.
"""

from __future__ import annotations

from vrag.models.whisper import TranscribedSegment, TranscribedWord
from vrag.passes.p1_speech import (
    align_config_hash,
    asr_config_hash,
    group_words_into_windows,
    keep_segment,
    split_into_sentences,
)


def word(text: str, t_start: float, t_end: float) -> TranscribedWord:
    return TranscribedWord(text=text, t_start=t_start, t_end=t_end)


def segment(text: str, *, no_speech_prob: float = 0.0, avg_logprob: float = -0.2):
    return TranscribedSegment(
        t_start=0.0, t_end=1.0, text=text,
        no_speech_prob=no_speech_prob, avg_logprob=avg_logprob, words=[],
    )


GATES = {"no_speech_threshold": 0.6, "min_avg_logprob": -1.0}


def test_real_speech_is_kept():
    assert keep_segment(segment("the cat sat on the mat"), **GATES)


def test_hallucinated_silence_is_dropped():
    # The classic Whisper-on-silence output, with the tell-tale no_speech_prob.
    assert not keep_segment(
        segment("Thank you for watching!", no_speech_prob=0.95), **GATES
    )


def test_low_confidence_text_is_dropped():
    # Hallucination that does not trip no_speech_prob still scores badly.
    assert not keep_segment(segment("mumble mumble", avg_logprob=-2.5), **GATES)


def test_empty_segment_is_dropped():
    assert not keep_segment(segment("   "), **GATES)


def test_sentences_split_on_punctuation():
    words = [
        word("Hello", 0.0, 0.5), word("there.", 0.5, 1.0),
        word("How", 1.2, 1.5), word("are", 1.5, 1.8), word("you?", 1.8, 2.2),
    ]
    sentences = split_into_sentences(words, max_words=60, max_seconds=20.0)
    assert [s.text for s in sentences] == ["Hello there.", "How are you?"]
    assert sentences[0].t_start == 0.0 and sentences[0].t_end == 1.0
    assert sentences[1].t_start == 1.2 and sentences[1].t_end == 2.2


def test_closing_quote_after_punctuation_still_ends_a_sentence():
    words = [word('"Stop!"', 0.0, 0.5), word("Then", 0.6, 0.9), word("silence.", 0.9, 1.4)]
    sentences = split_into_sentences(words, max_words=60, max_seconds=20.0)
    assert len(sentences) == 2


def test_unpunctuated_run_is_broken_by_the_word_cap():
    # Whisper on fast speech sometimes emits no punctuation at all. Without the
    # cap this would be one "sentence" and pass 2 would have no snap targets.
    words = [word(f"w{i}", float(i), i + 1.0) for i in range(7)]
    sentences = split_into_sentences(words, max_words=3, max_seconds=999.0)
    assert [len(s.text.split()) for s in sentences] == [3, 3, 1]


def test_unpunctuated_run_is_broken_by_the_duration_cap():
    words = [word(f"w{i}", float(i), i + 1.0) for i in range(12)]
    sentences = split_into_sentences(words, max_words=999, max_seconds=5.0)
    assert all(s.t_end - s.t_start <= 5.0 for s in sentences)
    assert len(sentences) > 1


def test_no_words_yields_no_sentences():
    assert split_into_sentences([], max_words=60, max_seconds=20.0) == []


def test_windows_respect_the_duration_cap():
    words = [word(f"w{i}", float(i), i + 1.0) for i in range(25)]
    windows = group_words_into_windows(words, max_window_s=10.0)

    assert [i for window in windows for i in window] == list(range(25)), "no word lost"
    for window in windows:
        span = words[window[-1]].t_end - words[window[0]].t_start
        assert span <= 10.0


def test_a_single_overlong_word_gets_its_own_window():
    words = [word("um", 0.0, 30.0), word("hello", 30.0, 30.5)]
    windows = group_words_into_windows(words, max_window_s=10.0)
    assert windows == [[0], [1]]


def test_no_words_yields_no_windows():
    assert group_words_into_windows([], max_window_s=10.0) == []


def test_stage_hashes_are_independent():
    """Tuning the aligner must not force a corpus-wide re-transcription."""
    settings = {
        "asr": {"model": "large-v3", "beam_size": 5},
        "align": {"model": "MMS_FA", "window_s": 20.0},
        "sentence": {"max_words": 60, "max_seconds": 20.0},
    }
    tuned = {**settings, "align": {"model": "MMS_FA", "window_s": 30.0}}

    assert asr_config_hash(settings) == asr_config_hash(tuned)
    assert align_config_hash(settings) != align_config_hash(tuned)


def test_sentence_settings_invalidate_both_stages():
    """Both stages derive sentences, so both must re-run when those rules change."""
    settings = {
        "asr": {"model": "large-v3"},
        "align": {"model": "MMS_FA"},
        "sentence": {"max_words": 60, "max_seconds": 20.0},
    }
    tuned = {**settings, "sentence": {"max_words": 40, "max_seconds": 20.0}}

    assert asr_config_hash(settings) != asr_config_hash(tuned)
    assert align_config_hash(settings) != align_config_hash(tuned)
