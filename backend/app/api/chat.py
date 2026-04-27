"""Multi-turn chat endpoint for clinician-facing trial planning."""
import json
from typing import Literal

import anthropic
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..dependencies import get_pipeline, get_client
from ..core.pipeline import ClinicalTrialRAG
from ..core.chat import generate_chat_stream, summarize_conversation
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


class SummarizeRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1)


class SummarizeResponse(BaseModel):
    summary: str


@router.post("/chat/stream")
async def chat_stream(
    body: ChatRequest,
    pipeline: ClinicalTrialRAG = Depends(get_pipeline),
    client: anthropic.AsyncAnthropic = Depends(get_client),
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

    # When a population is established, fetch a wide pool and diversify across
    # drug classes so the LLM sees representative coverage rather than top-K
    # near-duplicates. Otherwise stick with the user's top_k.
    if merged_filters:
        pool = pipeline.retrieve(retrieval_query, top_k=_OVERFETCH_POOL)
        retrieved = diversify_by_drug_class(pool, _DIVERSIFIED_BUDGET) or pool[: body.top_k]
    else:
        retrieved = pipeline.retrieve(retrieval_query, top_k=body.top_k)

    landscape = compute_landscape(pipeline.documents, merged_filters) if merged_filters else None

    async def event_generator():
        # Deterministic landscape brief over the full corpus. Emit whenever the
        # current turn introduces a new filter (disease / phase / status); also
        # fires on the first turn since prior_filters is empty there.
        if landscape and merged_filters != prior_filters:
            yield f"event: landscape\ndata: {json.dumps(landscape)}\n\n"

        source_docs = [_to_source_doc(doc).model_dump() for doc in retrieved]
        yield f"event: sources\ndata: {json.dumps(source_docs)}\n\n"

        try:
            async for token in generate_chat_stream(client, history, retrieved, pipeline.model, landscape=landscape):
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
