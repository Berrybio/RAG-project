"""Reranker that nudges retrieval scores based on past 👍/👎 feedback.

Two-step loop:

1. ``compute_source_scores`` walks the feedback log and returns a
   ``{nct_id: net_score}`` map, weighting recent votes more heavily via an
   exponential time decay.
2. ``adjust_scores`` adds a *capped* nudge to each retrieved doc's score and
   resorts. The cap matters — a single down-vote should never bury an
   objectively-best similarity match, only break ties / re-order the middle of
   the result list.

The map lives on ``app.state.source_scores`` and is rebuilt whenever new
feedback arrives, so the next request after a thumbs-down already feels the
adjustment without a server restart.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

from .feedback import FeedbackPaths, read_feedback

# How long a single vote takes to lose half its weight. 30 days is short
# enough that a fixed regression gets quickly washed out, long enough that
# clear wrong answers don't get re-promoted the moment activity slows.
_HALF_LIFE_DAYS = 30.0

# Hard ceiling on the per-source score nudge. Embedding similarities sit in
# roughly the [0, 1] range; ±0.05 is enough to break ties without overruling
# a strong signal.
_MAX_ADJUSTMENT = 0.05

# Per-vote weights before decay. Down-votes get a stronger pull because
# they're the explicit "this was wrong" signal — up-votes can come from a
# user just being polite about a mediocre answer.
_UP_WEIGHT = 1.0
_DOWN_WEIGHT = 1.5


def compute_source_scores(
    paths: FeedbackPaths,
    now: datetime | None = None,
) -> dict[str, float]:
    """Aggregate net feedback per NCT id with exponential time decay.

    Returns ``{}`` when no feedback has been submitted or none of the rows
    carry source NCT ids — older rows from before the reranker shipped don't.
    """
    rows = read_feedback(paths)
    if not rows:
        return {}
    now = now or datetime.now(timezone.utc)
    scores: dict[str, float] = {}
    for row in rows:
        kind = row.get("type")
        if kind not in ("rating_up", "rating_down"):
            continue
        ctx = row.get("context") or {}
        nct_ids = ctx.get("source_nct_ids") or []
        if not nct_ids:
            continue
        decay = _decay_factor(_parse_ts(row.get("created_at")), now)
        delta = (_UP_WEIGHT if kind == "rating_up" else -_DOWN_WEIGHT) * decay
        for nct in nct_ids:
            nct = (nct or "").strip()
            if not nct:
                continue
            scores[nct] = scores.get(nct, 0.0) + delta
    return scores


def _parse_ts(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return datetime.now(timezone.utc)


def _decay_factor(then: datetime, now: datetime) -> float:
    age_days = max(0.0, (now - then).total_seconds() / 86400.0)
    return math.exp(-math.log(2.0) * age_days / _HALF_LIFE_DAYS)


def adjust_scores(
    docs: list[dict],
    source_scores: dict[str, float] | None,
) -> list[dict]:
    """Apply capped feedback nudges to retrieved docs and resort by score.

    Returns a new list — never mutates the input. ``tanh`` keeps the nudge
    bounded regardless of feedback volume, so a pile-on can't dominate the
    similarity signal.
    """
    if not source_scores or not docs:
        return list(docs)
    adjusted: list[dict] = []
    for d in docs:
        nct = (d.get("metadata", {}) or {}).get("nctId") or d.get("doc_id") or ""
        net = source_scores.get(nct, 0.0)
        nudge = math.tanh(net) * _MAX_ADJUSTMENT
        new_doc = dict(d)
        new_doc["score"] = float(d.get("score", 0.0)) + nudge
        adjusted.append(new_doc)
    adjusted.sort(key=lambda d: d.get("score", 0.0), reverse=True)
    return adjusted
