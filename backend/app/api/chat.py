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
    retrieved = pipeline.retrieve(latest_query, top_k=body.top_k)
    history = [m.model_dump() for m in body.messages]

    async def event_generator():
        source_docs = [_to_source_doc(doc).model_dump() for doc in retrieved]
        yield f"event: sources\ndata: {json.dumps(source_docs)}\n\n"

        try:
            async for token in generate_chat_stream(client, history, retrieved, pipeline.model):
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
