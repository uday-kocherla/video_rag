"""Transcript BM25 index and end-to-end single-shot retrieval.

Uses real SQLite and real FTS5 against a hand-built two-video corpus, so the
whole M3 path — segment, index, search, merge — runs here with no models and no
network.
"""

from __future__ import annotations

import time

import pytest

from vrag import db
from vrag.passes import p2_segment, p4_index
from vrag.retrieve import search as retrieve
from vrag.schema import Sentence, Shot, Video

CONFIG = {
    "p2": {
        "strategy": "fixed",
        "l0": {"target_s": 16.0, "min_s": 8.0, "max_s": 30.0, "stride_ratio": 1.0},
        "l1": {"min_s": 120.0, "max_s": 300.0},
        "silent_words_per_s": 0.3,
    },
    "p4": {"level": 0},
    "p5": {"top_k": 20, "max_gap_s": 2.0},
}


@pytest.fixture()
def corpus(tmp_path):
    """Two videos: one about espresso, one silent."""
    conn = db.connect(tmp_path / "corpus.sqlite")
    db.insert(conn, [
        Video(video_id="talk", url="u1", added_at=time.time(), duration_s=60.0),
        Video(video_id="quiet", url="u2", added_at=time.time(), duration_s=60.0),
    ])
    db.insert(conn, [
        Shot(video_id="talk", t_start=0.0, t_end=60.0),
        Shot(video_id="quiet", t_start=0.0, t_end=30.0),
        Shot(video_id="quiet", t_start=30.0, t_end=60.0),
    ])
    # Dense enough to clear the silent threshold, and each sentence sits in a
    # different 16-second window so retrieval has somewhere specific to land.
    db.insert(conn, [
        Sentence(video_id="talk", t_start=1.0, t_end=6.0,
                 text="Today we are grinding coffee beans for espresso."),
        Sentence(video_id="talk", t_start=7.0, t_end=14.0,
                 text="The grinder burr size changes extraction a great deal."),
        Sentence(video_id="talk", t_start=20.0, t_end=28.0,
                 text="Now we tamp the puck and pull the shot."),
        Sentence(video_id="talk", t_start=40.0, t_end=50.0,
                 text="Finally we steam the milk into microfoam for a flat white."),
    ])
    conn.commit()

    p2_segment.run_corpus(conn, CONFIG)
    index_conn = p4_index.open_index(tmp_path / "bm25.sqlite")
    p4_index.build_transcript_index(conn, index_conn, level=0)
    yield conn, index_conn
    index_conn.close()
    conn.close()


def test_units_were_built_for_both_videos(corpus):
    conn, _ = corpus
    counts = dict(conn.execute(
        "SELECT video_id, count(*) FROM units WHERE level = 0 GROUP BY video_id"
    ).fetchall())

    assert counts["talk"] > 1
    assert counts["quiet"] > 1, "a silent video still gets units"


def test_silent_video_units_are_flagged(corpus):
    conn, _ = corpus
    flags = [row["flags"] for row in conn.execute(
        "SELECT flags FROM units WHERE video_id = 'quiet' AND level = 0"
    )]

    assert all(f and "silent" in f for f in flags)


def test_query_finds_the_right_moment(corpus):
    conn, index_conn = corpus
    spans = retrieve.search(conn, index_conn, "steaming milk foam", CONFIG)

    assert spans, "expected a hit"
    best = spans[0]
    assert best.video_id == "talk"
    assert best.t_start <= 45.0 < best.t_end, (
        f"the milk sentence is at 40-50s, got {best.t_start}-{best.t_end}"
    )


def test_a_different_query_finds_a_different_moment(corpus):
    conn, index_conn = corpus
    spans = retrieve.search(conn, index_conn, "grinder burr extraction", CONFIG)

    assert spans[0].t_start < 20.0, "the grinder sentence is early in the video"


def test_silent_video_never_wins_a_transcript_query(corpus):
    conn, index_conn = corpus
    spans = retrieve.search(conn, index_conn, "espresso", CONFIG)

    assert all(s.video_id == "talk" for s in spans), (
        "units with no transcript text cannot match a transcript query"
    )


def test_unmatched_query_returns_nothing(corpus):
    conn, index_conn = corpus

    assert retrieve.search(conn, index_conn, "quantum chromodynamics", CONFIG) == []


def test_video_filter_restricts_results(corpus):
    conn, index_conn = corpus
    spans = retrieve.search(conn, index_conn, "coffee", CONFIG, video_id="quiet")

    assert spans == []


def test_sentences_straddling_a_boundary_are_not_lost(corpus):
    conn, _ = corpus
    # The sentence at 7-14s and the window boundary at 16s: the 20-28s sentence
    # crosses into the second window, and both windows must carry its text.
    texts = dict(
        (unit_id, text) for unit_id, _, text in p4_index.unit_texts(conn, level=0)
    )
    with_tamp = [t for t in texts.values() if "tamp" in t]

    assert with_tamp, "the tamp sentence must appear in at least one unit"


def test_index_rebuild_is_idempotent(corpus):
    conn, index_conn = corpus
    first = p4_index.build_transcript_index(conn, index_conn, level=0)
    second = p4_index.build_transcript_index(conn, index_conn, level=0)

    assert first == second
    assert retrieve.search(conn, index_conn, "espresso", CONFIG), "still searchable"


def test_index_is_rebuildable_from_the_database_alone(corpus, tmp_path):
    """The disposability invariant: delete the index, pass 4 restores it."""
    conn, _ = corpus
    fresh = p4_index.open_index(tmp_path / "rebuilt.sqlite")
    p4_index.build_transcript_index(conn, fresh, level=0)

    assert retrieve.search(conn, fresh, "microfoam flat white", CONFIG)
    fresh.close()


# --------------------------------------------------------------------------
# query escaping — the class of bug that only shows up on real user input
# --------------------------------------------------------------------------


@pytest.mark.parametrize("query", [
    "what's the point?",
    "C++ vs Rust",
    'he said "hello"',
    "cost-benefit analysis",
    "NOT AND OR",
    "*",
    "(unbalanced",
])
def test_punctuation_in_a_query_does_not_raise(corpus, query):
    conn, index_conn = corpus

    retrieve.search(conn, index_conn, query, CONFIG)  # must not raise


def test_empty_query_returns_nothing(corpus):
    conn, index_conn = corpus

    assert retrieve.search(conn, index_conn, "   ", CONFIG) == []


def test_terms_are_or_ed_not_and_ed(corpus):
    """AND would return nothing the moment one query word is unusual."""
    conn, index_conn = corpus
    spans = retrieve.search(conn, index_conn, "espresso zzzzzz", CONFIG)

    assert spans, "one unknown term must not wipe out the whole query"


def test_re_segmenting_replaces_units_rather_than_duplicating(corpus):
    conn, _ = corpus
    before = conn.execute("SELECT count(*) FROM units").fetchone()[0]

    changed = {**CONFIG, "p2": {**CONFIG["p2"], "l0": {
        **CONFIG["p2"]["l0"], "target_s": 20.0}}}
    p2_segment.run_corpus(conn, changed)
    after = conn.execute("SELECT count(*) FROM units").fetchone()[0]

    assert after < before + 10, "old generation of units must be gone"
    strategies = conn.execute("SELECT DISTINCT strategy FROM units").fetchall()
    assert len(strategies) == 1


def test_unimplemented_strategy_fails_loudly(corpus):
    conn, _ = corpus
    changed = {**CONFIG, "p2": {**CONFIG["p2"], "strategy": "snapped"}}
    failures = p2_segment.run_corpus(conn, changed)

    assert failures, "selecting an unbuilt strategy must not silently use `fixed`"
    assert "NotImplementedError" in failures[0][1]
