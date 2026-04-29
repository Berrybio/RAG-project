import json
import time

from fastapi import APIRouter, Depends, Query

from fastapi.responses import StreamingResponse

from ..core.analytics import log_event
from ..dependencies import get_identity, get_pipeline
from ..models.schemas import SearchRequest, SearchResponse, SourceDoc
from ..core.pipeline import ClinicalTrialRAG

router = APIRouter()


def _to_source_doc(doc: dict) -> SourceDoc:
    meta = doc["metadata"]
    return SourceDoc(
        nct_id=meta.get("nctId", ""),
        title=meta.get("title", ""),
        score=doc["score"],
        # Status & design
        phases=meta.get("phases", ""),
        status=meta.get("status", ""),
        study_type=meta.get("studyType", ""),
        allocation=meta.get("allocation", ""),
        masking=meta.get("masking", ""),
        primary_purpose=meta.get("primaryPurpose", ""),
        # Conditions
        conditions=meta.get("conditions", ""),
        keywords=meta.get("keywords", ""),
        # Interventions
        intervention=meta.get("interventionName", ""),
        intervention_type=meta.get("interventionType", ""),
        drug_aliases=meta.get("interventionOtherNames", ""),
        # Enrollment & eligibility
        enrollment=str(meta.get("enrollmentCont", "")),
        enrollment_type=meta.get("enrollmentType", ""),
        sex=meta.get("sex", ""),
        minimum_age=meta.get("minimumAge", ""),
        maximum_age=meta.get("maximumAge", ""),
        # Outcomes
        primary_outcomes=meta.get("primaryOutcomes", ""),
        # Sponsor & PI
        sponsor=meta.get("sponsorName", ""),
        sponsor_class=meta.get("sponsorClass", ""),
        pi_name=meta.get("piName", ""),
        pi_affiliation=meta.get("piAffiliation", ""),
        # Dates
        start_date=meta.get("startDate", ""),
        completion_date=meta.get("completionETA", ""),
        last_updated=meta.get("lastUpdateDate", ""),
        # Location & contact
        location=meta.get("locationInfo", ""),
        location_countries=meta.get("locationCountries", ""),
        location_count=meta.get("locationCount", 0),
        contact=meta.get("contactInfo", ""),
    )


@router.post("/search", response_model=SearchResponse)
async def search_trials(
    body: SearchRequest,
    pipeline: ClinicalTrialRAG = Depends(get_pipeline),
    identity: dict = Depends(get_identity),
):
    started = time.monotonic()
    answer, sources = await pipeline.ask(body.query, top_k=body.top_k)
    log_event(
        "query_received",
        endpoint="search",
        query_text=body.query,
        num_sources=len(sources),
        latency_ms=int((time.monotonic() - started) * 1000),
        **identity,
    )
    return SearchResponse(
        answer=answer,
        sources=[_to_source_doc(doc) for doc in sources],
    )


@router.get("/search/stream")
async def search_trials_stream(
    query: str = Query(..., min_length=1),
    top_k: int = Query(default=5, ge=1, le=20),
    pipeline: ClinicalTrialRAG = Depends(get_pipeline),
    identity: dict = Depends(get_identity),
):
    started = time.monotonic()
    stream, sources = await pipeline.ask_stream(query, top_k=top_k)
    # Log retrieval as soon as sources are known. We don't wait for the full
    # streaming generation to finish — for analytics, "the user asked and got
    # N candidates back" is the event worth recording.
    log_event(
        "query_received",
        endpoint="search_stream",
        query_text=query,
        num_sources=len(sources),
        latency_ms=int((time.monotonic() - started) * 1000),
        **identity,
    )

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
