from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from ..dependencies import get_pipeline, get_client
from ..models.schemas import ProtocolRequest, ProtocolResponse, DocxRequest, SourceDoc
from ..core.pipeline import ClinicalTrialRAG
from ..core.protocol import generate_protocol_json, build_protocol_docx, build_protocol_pdf
from ..api.search import _to_source_doc

import anthropic

router = APIRouter()


@router.post("/protocol/json", response_model=ProtocolResponse)
async def create_protocol_json(
    body: ProtocolRequest,
    pipeline: ClinicalTrialRAG = Depends(get_pipeline),
    client: anthropic.AsyncAnthropic = Depends(get_client),
):
    retrieved = pipeline.retrieve(body.query, top_k=body.top_k)
    protocol = await generate_protocol_json(
        client, body.query, retrieved, model=pipeline.model,
    )
    return ProtocolResponse(
        protocol=protocol,
        reference_trials=[_to_source_doc(doc) for doc in retrieved],
    )


def _protocol_filename(protocol: dict, extension: str) -> str:
    """Build a safe filename, falling back to 'protocol' when protocol_id is empty."""
    stem = (protocol.get("protocol_id") or "").strip() or "protocol"
    return f"{stem}.{extension}"


@router.post("/protocol/docx")
async def create_protocol_docx(body: DocxRequest):
    buffer = build_protocol_docx(body.protocol)
    filename = _protocol_filename(body.protocol, "docx")
    return StreamingResponse(
        buffer,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/protocol/pdf")
async def create_protocol_pdf(body: DocxRequest):
    """Build and return a PDF rendering of the protocol.

    Reuses DocxRequest since the payload shape ({"protocol": {...}}) is the same.
    """
    buffer = build_protocol_pdf(body.protocol)
    filename = _protocol_filename(body.protocol, "pdf")
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
