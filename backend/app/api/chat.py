"""Multi-turn chat endpoint for clinician-facing trial planning."""
import json
from typing import Literal

import anthropic
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..dependencies import get_aliases, get_pipeline, get_client
from ..core.pipeline import ClinicalTrialRAG
from ..core.chat import generate_chat_stream, summarize_conversation
from ..core.feedback import expand_query
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


class SummarizeRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1)


class SummarizeResponse(BaseModel):
    summary: str


@router.post("/chat/stream")
async def chat_stream(
    body: ChatRequest,
    pipeline: ClinicalTrialRAG = Depends(get_pipeline),
    client: anthropic.AsyncAnthropic = Depends(get_client),
    aliases: dict = Depends(get_aliases),
):
    if body.messages[-1].role != "user":
        return {"error": "Last message must be from user"}

    latest_query = body.messages[-1].content
    history = [m.model_dump() for m in body.messages]

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
        pool = pipeline.retrieve(retrieval_query, top_k=_OVERFETCH_POOL)

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
        retrieved = pipeline.retrieve(retrieval_query, top_k=body.top_k)

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

    async def event_generator():
        # Deterministic landscape brief over the full corpus. Emit whenever the
        # current turn introduces a new filter (disease / phase / status); also
        # fires on the first turn since prior_filters is empty there.
        if landscape and merged_filters != prior_filters:
            yield f"event: landscape\ndata: {json.dumps(landscape)}\n\n"

        source_docs = [_to_source_doc(doc).model_dump() for doc in retrieved]
        yield f"event: sources\ndata: {json.dumps(source_docs)}\n\n"

        try:
            async for token in generate_chat_stream(
                client, history, retrieved, pipeline.model,
                landscape=landscape, aliases=aliases,
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
    pipeline: ClinicalTrialRAG = Depends(get_pipeline),
    client: anthropic.AsyncAnthropic = Depends(get_client),
):
    history = [m.model_dump() for m in body.messages]
    summary = await summarize_conversation(client, history, pipeline.model)
    return SummarizeResponse(summary=summary)
