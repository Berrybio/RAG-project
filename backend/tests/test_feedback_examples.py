"""Tests for the up-voted few-shot example store."""
from __future__ import annotations

from app.core.feedback import FeedbackPaths, append_feedback
from app.core.feedback_examples import (
    FeedbackExampleStore,
    format_examples_block,
)


_LONG_ANSWER = (
    "Recruiting Phase II / III breast cancer trials of trastuzumab deruxtecan "
    "include DESTINY-Breast06 (NCT04494425) and DESTINY-Breast09 (NCT04784715). "
    "Both target HER2-low and HER2-positive populations and report progression-"
    "free survival as the primary endpoint."
)


def _add_up(paths: FeedbackPaths, query: str, answer: str = _LONG_ANSWER):
    append_feedback(paths, {
        "type": "rating_up",
        "context": {
            "query": query,
            "assistant_message": answer,
            "source_nct_ids": ["NCT04494425"],
        },
    })


def test_load_empty(tmp_path):
    paths = FeedbackPaths.from_data_dir(tmp_path)
    store = FeedbackExampleStore.load(paths)
    assert store.examples == []
    assert store.pick("anything") == []


def test_load_keeps_only_substantive_up_votes(tmp_path):
    paths = FeedbackPaths.from_data_dir(tmp_path)
    _add_up(paths, "trastuzumab deruxtecan trials")
    # Down-vote with same long answer must NOT become a few-shot example.
    append_feedback(paths, {
        "type": "rating_down",
        "context": {
            "query": "different query",
            "assistant_message": _LONG_ANSWER,
            "source_nct_ids": ["NCT001"],
        },
    })
    # Up-vote with too-short answer is dropped.
    append_feedback(paths, {
        "type": "rating_up",
        "context": {
            "query": "short",
            "assistant_message": "Short reply.",
            "source_nct_ids": [],
        },
    })
    store = FeedbackExampleStore.load(paths)
    assert len(store.examples) == 1
    assert store.examples[0]["query"] == "trastuzumab deruxtecan trials"


def test_pick_finds_similar_query(tmp_path):
    paths = FeedbackPaths.from_data_dir(tmp_path)
    _add_up(paths, "trastuzumab deruxtecan trials in HER2-low breast cancer")
    _add_up(paths, "PARP inhibitors in BRCA-mutated ovarian cancer")
    store = FeedbackExampleStore.load(paths)
    picked = store.pick("trastuzumab deruxtecan in HER2-positive breast cancer", top_n=1)
    assert len(picked) == 1
    assert "trastuzumab" in picked[0]["query"]


def test_pick_skips_unrelated(tmp_path):
    paths = FeedbackPaths.from_data_dir(tmp_path)
    _add_up(paths, "PARP inhibitors in BRCA-mutated ovarian cancer")
    store = FeedbackExampleStore.load(paths)
    # Wholly unrelated query — TF-IDF cosine should fall below the threshold
    # and pick should return nothing rather than confuse the planner.
    picked = store.pick("xyz unrelated random tokens that do not overlap")
    assert picked == []


def test_format_examples_block_empty():
    assert format_examples_block([]) == ""


def test_format_examples_block_renders_q_and_a():
    block = format_examples_block([
        {"query": "Q1?", "answer": "A1 detailed reply."},
    ])
    assert "Q1?" in block
    assert "A1 detailed reply." in block
    assert "rated as helpful" in block.lower()


def test_format_examples_block_truncates_long_answer():
    long_answer = "x" * 5000
    block = format_examples_block([{"query": "Q", "answer": long_answer}])
    assert "…" in block
    # The whole prompt block must be smaller than the answer alone proves
    # truncation actually happened.
    assert len(block) < len(long_answer) + 200
