"""Tests for the feedback-driven reranker."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.core.feedback import FeedbackPaths, append_feedback
from app.core.feedback_reranker import (
    _MAX_ADJUSTMENT,
    adjust_scores,
    compute_source_scores,
)


def _doc(nct: str, score: float) -> dict:
    return {"doc_id": nct, "metadata": {"nctId": nct}, "score": score}


def test_compute_scores_empty_log(tmp_path):
    paths = FeedbackPaths.from_data_dir(tmp_path)
    assert compute_source_scores(paths) == {}


def test_compute_scores_aggregates_up_and_down(tmp_path):
    paths = FeedbackPaths.from_data_dir(tmp_path)
    now = datetime.now(timezone.utc)
    # Two ups and one down for NCT001 — should still be positive (up=1.0, down=1.5).
    append_feedback(paths, {
        "type": "rating_up",
        "context": {"source_nct_ids": ["NCT001", "NCT002"]},
        "created_at": now.isoformat(),
    })
    append_feedback(paths, {
        "type": "rating_up",
        "context": {"source_nct_ids": ["NCT001"]},
        "created_at": now.isoformat(),
    })
    append_feedback(paths, {
        "type": "rating_down",
        "context": {"source_nct_ids": ["NCT001"]},
        "created_at": now.isoformat(),
    })
    scores = compute_source_scores(paths, now=now)
    # 2*1.0 - 1*1.5 = 0.5 for NCT001; 1.0 for NCT002.
    assert scores["NCT001"] == 0.5
    assert scores["NCT002"] == 1.0


def test_compute_scores_decays_old_feedback(tmp_path):
    paths = FeedbackPaths.from_data_dir(tmp_path)
    now = datetime.now(timezone.utc)
    # 30 days old → decay factor = 0.5.
    append_feedback(paths, {
        "type": "rating_up",
        "context": {"source_nct_ids": ["NCT_OLD"]},
        "created_at": (now - timedelta(days=30)).isoformat(),
    })
    scores = compute_source_scores(paths, now=now)
    assert abs(scores["NCT_OLD"] - 0.5) < 1e-6


def test_compute_scores_skips_rows_without_nct_ids(tmp_path):
    paths = FeedbackPaths.from_data_dir(tmp_path)
    # Older feedback rows from before the reranker shipped have no
    # source_nct_ids — they should be ignored, not crash.
    append_feedback(paths, {
        "type": "rating_up",
        "context": {"query": "anything", "assistant_message": "..."},
    })
    assert compute_source_scores(paths) == {}


def test_compute_scores_ignores_corrections(tmp_path):
    paths = FeedbackPaths.from_data_dir(tmp_path)
    append_feedback(paths, {
        "type": "correction",
        "correction_kind": "missing_alias",
        "alias": "x",
        "canonical": "y",
        "context": {"source_nct_ids": ["NCT001"]},
    })
    assert compute_source_scores(paths) == {}


def test_adjust_scores_no_op_with_empty_scores():
    docs = [_doc("NCT001", 0.9), _doc("NCT002", 0.5)]
    out = adjust_scores(docs, {})
    # Same order, same scores, but a fresh list.
    assert [d["doc_id"] for d in out] == ["NCT001", "NCT002"]
    assert out is not docs


def test_adjust_scores_reorders_when_feedback_flips_ranking():
    docs = [_doc("NCT001", 0.50), _doc("NCT002", 0.49)]
    # Big down-vote on the leader; small up on the runner-up. The cap applies
    # so the swap only happens when the original gap is small.
    scores = {"NCT001": -5.0, "NCT002": 1.0}
    out = adjust_scores(docs, scores)
    assert [d["doc_id"] for d in out] == ["NCT002", "NCT001"]


def test_adjust_scores_does_not_overrule_strong_similarity_gap():
    # Gap of 0.5 between docs is far larger than the ±0.05 cap, so a single
    # down-vote should not change the order.
    docs = [_doc("NCT001", 0.95), _doc("NCT002", 0.45)]
    out = adjust_scores(docs, {"NCT001": -10.0})
    assert [d["doc_id"] for d in out] == ["NCT001", "NCT002"]


def test_adjust_scores_caps_at_max_adjustment():
    docs = [_doc("NCT001", 0.5)]
    # Even a huge accumulated up-vote can only nudge by _MAX_ADJUSTMENT.
    out = adjust_scores(docs, {"NCT001": 1000.0})
    assert abs(out[0]["score"] - (0.5 + _MAX_ADJUSTMENT)) < 1e-6


def test_adjust_scores_does_not_mutate_input():
    docs = [_doc("NCT001", 0.5)]
    adjust_scores(docs, {"NCT001": 1.0})
    assert docs[0]["score"] == 0.5
