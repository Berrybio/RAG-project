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

from ..core.feedback import (
    FeedbackPaths,
    add_alias,
    append_feedback,
    feedback_stats,
    load_aliases,
    read_feedback,
    update_feedback_status,
)
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
    return {
        "query": (ctx.get("query") or "")[:_MAX_CONTEXT_CHARS],
        "assistant_message": (ctx.get("assistant_message") or "")[:_MAX_CONTEXT_CHARS],
    }


@router.post("/feedback", response_model=FeedbackResponse)
async def post_feedback(
    body: RatingFeedback | CorrectionFeedback,
    paths: FeedbackPaths = Depends(get_feedback_paths),
):
    payload = body.model_dump()
    if "context" in payload:
        payload["context"] = _truncate_context(payload["context"])
    stored = append_feedback(paths, payload)
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
    return AliasesResponse(aliases=new_aliases)


@router.get("/aliases", response_model=AliasesResponse)
async def get_aliases(paths: FeedbackPaths = Depends(get_feedback_paths)):
    return AliasesResponse(aliases=load_aliases(paths))
