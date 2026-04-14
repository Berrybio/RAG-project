import json

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from ..dependencies import get_pipeline
from ..models.schemas import SearchRequest, SearchResponse, SourceDoc
from ..core.pipeline import ClinicalTrialRAG

router = APIRouter()


def _to_source_doc(doc: dict) -> SourceDoc:
    meta = doc["metadata"]
    return SourceDoc(
        nct_id=meta["nctId"],
        title=meta["title"],
        score=doc["score"],
        phases=meta["phases"],
        status=meta["status"],
        conditions=meta["conditions"],
        intervention=meta["interventionName"],
        enrollment=str(meta["enrollmentCont"]),
        sex=meta["sex"],
        minimum_age=meta["minimumAge"],
        location=meta["locationInfo"],
        contact=meta["contactInfo"],
    )


@router.post("/search", response_model=SearchResponse)
async def search_trials(
    body: SearchRequest,
    pipeline: ClinicalTrialRAG = Depends(get_pipeline),
):
    answer, sources = await pipeline.ask(body.query, top_k=body.top_k)
    return SearchResponse(
        answer=answer,
        sources=[_to_source_doc(doc) for doc in sources],
    )


@router.get("/search/stream")
async def search_trials_stream(
    query: str = Query(..., min_length=1),
    top_k: int = Query(default=5, ge=1, le=20),
    pipeline: ClinicalTrialRAG = Depends(get_pipeline),
):
    stream, sources = await pipeline.ask_stream(query, top_k=top_k)

    async def event_generator():
        source_docs = [_to_source_doc(doc).model_dump() for doc in sources]
        yield f"event: sources\ndata: {json.dumps(source_docs)}\n\n"

        if stream is None:
            yield f"event: token\ndata: No relevant clinical trials found for your query.\n\n"
            yield "event: done\ndata: {}\n\n"
            return

        try:
            async for token in stream:
                yield f"event: token\ndata: {json.dumps(token)}\n\n"
            yield "event: done\ndata: {}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
