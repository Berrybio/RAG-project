"""Multi-turn chat endpoint for clinician-facing trial planning."""
import json
import logging
import time
from typing import Literal

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..core.analytics import log_event
from ..core.llm import BaseLLMProvider
from ..dependencies import (
    get_aliases,
    get_examples,
    get_identity,
    get_llm,
    get_pipeline,
    get_protocol_history_paths,
    get_source_scores,
)
from ..core.pipeline import ClinicalTrialRAG
from ..core.chat import generate_chat_stream, summarize_conversation
from ..core.feedback import expand_query
from ..core.feedback_examples import FeedbackExampleStore
from ..core.protocol_history import (
    load_protocol,
    match_query_to_entry,
    query_mentions_previous_protocol,
)
from ..core.landscape import (
    compute_landscape,
    detect_disease,
    diversify_by_drug_class,
    merge_filters_from_history,
)

# When a population is established, we want the LLM context to span drug
# classes rather than the top-K nearest neighbors (which often cluster on one
# modality). Overfetch a wide candidate pool, then diversify to this many
# trials regardless of the slider's top_k value.
_DIVERSIFIED_BUDGET = 12
_OVERFETCH_POOL = 80
from ..api.search import _to_source_doc

logger = logging.getLogger(__name__)

router = APIRouter()


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    # When false, skip the deterministic landscape brief (and the drug-class
    # diversification that goes with it). Useful for clinicians who already
    # know exactly what they're looking for and want a focused top-K result.
    include_landscape: bool = True
    # Optional id of the protocol the user currently has open in the planner.
    # When set, the backend injects a small <active_protocol> hint into the
    # LLM context so questions like "is our sample size reasonable?" don't
    # require pasting the protocol back into chat.
    active_protocol_id: str = ""


class SummarizeRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1)


class SummarizeResponse(BaseModel):
    summary: str


@router.post("/chat/stream")
async def chat_stream(
    body: ChatRequest,
    pipeline: ClinicalTrialRAG = Depends(get_pipeline),
    llm: BaseLLMProvider = Depends(get_llm),
    aliases: dict = Depends(get_aliases),
    source_scores: dict = Depends(get_source_scores),
    examples: FeedbackExampleStore | None = Depends(get_examples),
    history_paths=Depends(get_protocol_history_paths),
    identity: dict = Depends(get_identity),
):
    started = time.monotonic()
    if body.messages[-1].role != "user":
        return {"error": "Last message must be from user"}

    latest_query = body.messages[-1].content
    history = [m.model_dump() for m in body.messages]
    turn_number = sum(1 for m in body.messages if m.role == "user")
    log_event(
        "chat_message",
        turn_number=turn_number,
        query_text=latest_query,
        include_landscape=body.include_landscape,
        **identity,
    )

    # Carry filter context across turns. If the population (e.g. TNBC) was
    # established earlier but the current turn says only "phase II", the
    # retriever would otherwise pick up unrelated populations (HER2+, etc.).
    merged_filters = merge_filters_from_history(history)
    prior_filters = merge_filters_from_history(history[:-1])

    retrieval_query = latest_query
    if merged_filters and not detect_disease(latest_query):
        retrieval_query = f"{merged_filters['disease_name']} — {latest_query}"
    # Expand any user-curated drug aliases (e.g. "Dato-DXd" -> append
    # "Datopotamab deruxtecan") so retrieval matches the canonical name in
    # the trial CSV. Original wording stays in `latest_query` for the LLM.
    retrieval_query = expand_query(retrieval_query, aliases)

    # When a population is established AND the clinician opted into the
    # landscape view, fetch a wide pool and diversify across drug classes so
    # the LLM sees representative coverage rather than top-K near-duplicates.
    # If they opted out, they want focused similarity-ranked results.
    if merged_filters and body.include_landscape:
        pool = pipeline.retrieve(
            retrieval_query, top_k=_OVERFETCH_POOL, source_scores=source_scores,
        )

        # Dense retrieval ranks by similarity but doesn't enforce phase/status,
        # so a query like "Phase III PD-(L)1 in TNBC" can pull Phase I/II
        # neighbors. Apply the structural filters from merged_filters here so
        # the source cards (and LLM trial list) match the landscape population.
        target_phase = merged_filters.get("phase")
        target_status = merged_filters.get("status")
        if target_phase or target_status:
            filtered = [
                d for d in pool
                if (not target_phase or d.get("metadata", {}).get("phases", "").strip() == target_phase)
                and (not target_status or d.get("metadata", {}).get("status", "").strip() == target_status)
            ]
            # If filtering wipes the pool (over-strict), fall back to the
            # unfiltered candidates rather than returning nothing.
            pool = filtered or pool

        retrieved = diversify_by_drug_class(pool, _DIVERSIFIED_BUDGET) or pool[: body.top_k]
    else:
        retrieved = pipeline.retrieve(
            retrieval_query, top_k=body.top_k, source_scores=source_scores,
        )

    # Stabilize ordering with two priority tiers (Python's sort is stable, so
    # retrieval rank is preserved within each bucket). Sponsor is the primary
    # key — clinicians want to see the pharma-driven competitive landscape
    # together — and recruiting status is secondary within each sponsor group:
    #   1. INDUSTRY + RECRUITING
    #   2. INDUSTRY + other statuses
    #   3. Non-INDUSTRY + RECRUITING
    #   4. Non-INDUSTRY + other statuses
    def _sort_key(d: dict) -> tuple[int, int]:
        meta = d.get("metadata", {})
        sponsor_tier = 0 if meta.get("sponsorClass", "").strip() == "INDUSTRY" else 1
        status_tier = 0 if meta.get("status", "").strip() == "RECRUITING" else 1
        return (sponsor_tier, status_tier)

    retrieved = sorted(retrieved, key=_sort_key)

    landscape = (
        compute_landscape(pipeline.documents, merged_filters)
        if merged_filters and body.include_landscape
        else None
    )

    user_id = (identity.get("user_id") or "anonymous").strip() or "anonymous"

    # Detect "based on the previous protocol" hints in the user's message and
    # surface ALL their stored protocols as clickable load chips in the UI.
    # The list (capped, ranked by relevance) is also injected into the LLM's
    # user message as a <previous_protocols> hint so the model acknowledges
    # them rather than gaslighting the user with "no protocol exists".
    protocol_matches: list[dict] = []
    if query_mentions_previous_protocol(latest_query):
        try:
            protocol_matches = match_query_to_entry(
                history_paths, user_id, latest_query, top_n=10,
            )
        except Exception:  # pragma: no cover - history is best-effort
            logger.exception("Failed to match previous protocols for chat hint")

    # Resolve the active-protocol meta + latest JSON. When the frontend passes
    # active_protocol_id, the planner has one open in the preview pane; we
    # surface that to the LLM so it can answer questions about the protocol
    # without the user having to paste it back into chat.
    active_protocol_meta: dict | None = None
    active_protocol_json: dict | None = None
    if body.active_protocol_id:
        try:
            active_protocol_json, active_protocol_meta = load_protocol(
                history_paths, user_id, body.active_protocol_id,
            )
        except FileNotFoundError:
            logger.info(
                "chat_stream: active_protocol_id %s not found for user %s",
                body.active_protocol_id, user_id,
            )
        except Exception:  # pragma: no cover - non-critical
            logger.exception("Failed to load active protocol for chat context")

    log_event(
        "query_received",
        endpoint="chat_stream",
        query_text=latest_query,
        num_sources=len(retrieved),
        landscape_emitted=bool(landscape and merged_filters != prior_filters),
        protocol_matches=len(protocol_matches) or None,
        latency_ms=int((time.monotonic() - started) * 1000),
        **identity,
    )

    async def event_generator():
        # Deterministic landscape brief over the full corpus. Emit whenever the
        # current turn introduces a new filter (disease / phase / status); also
        # fires on the first turn since prior_filters is empty there.
        if landscape and merged_filters != prior_filters:
            yield f"event: landscape\ndata: {json.dumps(landscape)}\n\n"

        # If the user said "based on the previous protocol" (or similar) and we
        # found one or more candidates, emit them so the frontend can render
        # clickable load chips. The reply itself proceeds normally — the chip
        # is an additional offer, not a redirect.
        if protocol_matches:
            yield (
                "event: protocol_match\n"
                f"data: {json.dumps(protocol_matches)}\n\n"
            )

        source_docs = [_to_source_doc(doc).model_dump() for doc in retrieved]
        yield f"event: sources\ndata: {json.dumps(source_docs)}\n\n"

        try:
            async for token in generate_chat_stream(
                llm, history, retrieved,
                landscape=landscape, aliases=aliases, examples=examples,
                previous_protocols=protocol_matches,
                active_protocol_meta=active_protocol_meta,
                active_protocol_json=active_protocol_json,
            ):
                yield f"event: token\ndata: {json.dumps(token)}\n\n"
            yield "event: done\ndata: {}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/chat/summarize", response_model=SummarizeResponse)
async def chat_summarize(
    body: SummarizeRequest,
    llm: BaseLLMProvider = Depends(get_llm),
):
    history = [m.model_dump() for m in body.messages]
    summary = await summarize_conversation(llm, history)
    return SummarizeResponse(summary=summary)
