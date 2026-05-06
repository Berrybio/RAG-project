"""Feedback collection + admin review endpoints.

Public:
- POST /api/feedback              -> append rating or correction
Admin (gated by ?admin=1 in the UI; no server-side auth in this build):
- GET  /api/feedback              -> list all entries + stats
- POST /api/feedback/{id}/status  -> mark resolved/dismissed
- POST /api/feedback/promote      -> add an alias to the dictionary
                                     (and mark the source feedback resolved)
- GET  /api/aliases               -> current alias dictionary
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from ..core.analytics import log_event
from ..dependencies import get_identity
from ..core.feedback import (
    FeedbackPaths,
    add_alias,
    append_feedback,
    feedback_stats,
    load_aliases,
    read_feedback,
    update_feedback_status,
)
from ..core.feedback_examples import FeedbackExampleStore
from ..core.feedback_reranker import compute_source_scores
from ..models.schemas import (
    AliasesResponse,
    CorrectionFeedback,
    FeedbackListResponse,
    FeedbackResponse,
    FeedbackStatusUpdate,
    PromoteAliasRequest,
    RatingFeedback,
)

router = APIRouter()


def get_feedback_paths(request: Request) -> FeedbackPaths:
    return request.app.state.feedback_paths


# Truncate any context fields submitted from the client to keep the log
# compact and avoid storing hundreds of KB per row when the user clicks
# thumbs-down on a long protocol-related answer.
_MAX_CONTEXT_CHARS = 2000


def _truncate_context(ctx: dict) -> dict:
    # Cap the source list too — a long landscape diversification can return
    # 12-15 trials, but past 25 is almost certainly junk or a bug.
    raw_sources = ctx.get("source_nct_ids") or []
    sources = [str(s).strip() for s in raw_sources if s][:25]
    return {
        "query": (ctx.get("query") or "")[:_MAX_CONTEXT_CHARS],
        "assistant_message": (ctx.get("assistant_message") or "")[:_MAX_CONTEXT_CHARS],
        "source_nct_ids": sources,
    }


@router.post("/feedback", response_model=FeedbackResponse)
async def post_feedback(
    body: RatingFeedback | CorrectionFeedback,
    request: Request,
    paths: FeedbackPaths = Depends(get_feedback_paths),
    identity: dict = Depends(get_identity),
):
    payload = body.model_dump()
    if "context" in payload:
        payload["context"] = _truncate_context(payload["context"])
    stored = append_feedback(paths, payload)
    # Refresh derived caches so the next retrieval / next chat reply already
    # feels this feedback. Both are cheap to recompute (the feedback log is
    # human-scale) and only ratings actually move them — corrections are a
    # no-op for both.
    if payload.get("type") in ("rating_up", "rating_down"):
        request.app.state.source_scores = compute_source_scores(paths)
        request.app.state.examples = FeedbackExampleStore.load(paths)
    # Two distinct signals matter for analytics: (a) which kind of feedback it
    # was — thumbs up vs thumbs down vs free-text correction; (b) the query
    # that produced the answer being rated. Pull both out of the truncated
    # context the client sent, falling back gracefully when fields are absent.
    ctx = payload.get("context") or {}
    log_event(
        "feedback_submitted",
        feedback_id=stored["id"],
        kind=payload.get("type"),  # rating_up | rating_down | correction
        correction_kind=payload.get("correction_kind"),
        query_text=ctx.get("query"),
        correction_text=payload.get("notes") or payload.get("reason"),
        **identity,
    )
    return FeedbackResponse(id=stored["id"], status=stored.get("status", "open"))


@router.get("/feedback", response_model=FeedbackListResponse)
async def list_feedback(paths: FeedbackPaths = Depends(get_feedback_paths)):
    return FeedbackListResponse(
        entries=read_feedback(paths),
        stats=feedback_stats(paths),
    )


@router.post("/feedback/{feedback_id}/status", response_model=FeedbackResponse)
async def update_status(
    feedback_id: str,
    body: FeedbackStatusUpdate,
    paths: FeedbackPaths = Depends(get_feedback_paths),
):
    updated = update_feedback_status(paths, feedback_id, body.status)
    if updated is None:
        raise HTTPException(status_code=404, detail="feedback id not found")
    return FeedbackResponse(id=updated["id"], status=updated.get("status", body.status))


@router.post("/feedback/promote", response_model=AliasesResponse)
async def promote_to_alias(
    body: PromoteAliasRequest,
    request: Request,
    paths: FeedbackPaths = Depends(get_feedback_paths),
):
    """Add an alias to the dictionary and mark the source feedback resolved.

    Also refreshes the in-memory alias cache on app.state so the next query
    benefits immediately without a server restart.
    """
    if not body.alias.strip() or not body.canonical.strip():
        raise HTTPException(status_code=400, detail="alias and canonical are required")
    new_aliases = add_alias(paths, body.alias, body.canonical)
    update_feedback_status(paths, body.feedback_id, "resolved")
    request.app.state.aliases = new_aliases
    log_event(
        "feedback_promoted",
        feedback_id=body.feedback_id,
        alias=body.alias,
        canonical=body.canonical,
        # No identity from the admin endpoint — admins typically have URL-flag
        # access, not the cookie. The action itself is what matters.
    )
    return AliasesResponse(aliases=new_aliases)


@router.get("/aliases", response_model=AliasesResponse)
async def get_aliases(paths: FeedbackPaths = Depends(get_feedback_paths)):
    return AliasesResponse(aliases=load_aliases(paths))
