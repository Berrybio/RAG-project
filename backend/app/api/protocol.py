import time

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from ..core.analytics import log_event
from ..core.llm import BaseLLMProvider
from ..dependencies import get_identity, get_llm, get_pipeline, get_source_scores
from ..models.schemas import (
    ProtocolRequest,
    ProtocolResponse,
    DocxRequest,
    SourceDoc,
    RefineProtocolRequest,
    RefineProtocolResponse,
)
from ..core.pipeline import ClinicalTrialRAG
from ..core.protocol import (
    generate_protocol_json,
    refine_protocol_json,
    build_protocol_docx,
    build_protocol_pdf,
)
from ..api.search import _to_source_doc

router = APIRouter()


@router.post("/protocol/json", response_model=ProtocolResponse)
async def create_protocol_json(
    body: ProtocolRequest,
    pipeline: ClinicalTrialRAG = Depends(get_pipeline),
    llm: BaseLLMProvider = Depends(get_llm),
    source_scores: dict = Depends(get_source_scores),
    identity: dict = Depends(get_identity),
):
    started = time.monotonic()
    retrieved = pipeline.retrieve(
        body.query, top_k=body.top_k, source_scores=source_scores,
    )
    protocol = await generate_protocol_json(llm, body.query, retrieved)
    log_event(
        "protocol_generated",
        format="json",
        phase=protocol.get("phase"),
        study_type=protocol.get("study_type"),
        num_reference_trials=len(retrieved),
        latency_ms=int((time.monotonic() - started) * 1000),
        **identity,
    )
    return ProtocolResponse(
        protocol=protocol,
        reference_trials=[_to_source_doc(doc) for doc in retrieved],
    )


def _protocol_filename(protocol: dict, extension: str) -> str:
    """Build a safe filename, falling back to 'protocol' when protocol_id is empty."""
    stem = (protocol.get("protocol_id") or "").strip() or "protocol"
    return f"{stem}.{extension}"


@router.post("/protocol/refine", response_model=RefineProtocolResponse)
async def refine_protocol(
    body: RefineProtocolRequest,
    llm: BaseLLMProvider = Depends(get_llm),
    identity: dict = Depends(get_identity),
):
    """Apply a clinician's correction request to an existing protocol JSON."""
    started = time.monotonic()
    updated, note, changed = await refine_protocol_json(
        llm,
        current_protocol=body.protocol,
        refinement_messages=[m.model_dump() for m in body.refinement_messages],
        original_summary=body.original_summary,
    )
    # turn_number = how many refinement rounds the user has done so far
    turn_number = sum(1 for m in body.refinement_messages if m.role == "user")
    log_event(
        "protocol_refined",
        turn_number=turn_number,
        sections_changed=changed,
        num_sections_changed=len(changed) if changed else 0,
        latency_ms=int((time.monotonic() - started) * 1000),
        **identity,
    )
    return RefineProtocolResponse(
        protocol=updated,
        assistant_message=note,
        changed_fields=changed,
    )


@router.post("/protocol/docx")
async def create_protocol_docx(
    body: DocxRequest,
    identity: dict = Depends(get_identity),
):
    buffer = build_protocol_docx(body.protocol)
    filename = _protocol_filename(body.protocol, "docx")
    log_event(
        "protocol_downloaded",
        format="docx",
        phase=body.protocol.get("phase"),
        study_type=body.protocol.get("study_type"),
        **identity,
    )
    return StreamingResponse(
        buffer,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/protocol/pdf")
async def create_protocol_pdf(
    body: DocxRequest,
    identity: dict = Depends(get_identity),
):
    """Build and return a PDF rendering of the protocol.

    Reuses DocxRequest since the payload shape ({"protocol": {...}}) is the same.
    """
    buffer = build_protocol_pdf(body.protocol)
    filename = _protocol_filename(body.protocol, "pdf")
    log_event(
        "protocol_downloaded",
        format="pdf",
        phase=body.protocol.get("phase"),
        study_type=body.protocol.get("study_type"),
        **identity,
    )
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
