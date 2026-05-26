import json
import logging
import time

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from ..cancer_registry import CANCER_TYPES
from ..core.analytics import log_event
from ..core.llm import BaseLLMProvider
from ..core.protocol_history import (
    ProtocolHistoryPaths,
    list_protocols,
    load_all_versions,
    load_protocol,
    save_initial,
    save_revision,
)
from ..dependencies import (
    get_identity,
    get_llm,
    get_pipeline_manager,
    get_protocol_history_paths,
    get_source_scores,
)
from ..models.schemas import (
    DocxRequest,
    ProtocolEntry,
    ProtocolListResponse,
    ProtocolLoadResponse,
    ProtocolRequest,
    ProtocolResponse,
    ProtocolVersionEntry,
    RefineProtocolRequest,
    RefineProtocolResponse,
    SourceDoc,
)
from ..core.pipeline import ClinicalTrialRAG
from ..core.pipeline_manager import PipelineManager
from ..core.protocol import (
    generate_protocol_json,
    generate_protocol_json_stream,
    refine_protocol_json,
    refine_protocol_json_stream,
    build_protocol_docx,
    build_protocol_pdf,
    _parse_protocol_raw,
    parse_refine_output,
)
from ..api.search import _to_source_doc

logger = logging.getLogger(__name__)

router = APIRouter()


def _user_id(identity: dict) -> str:
    """Resolve the per-user history bucket. ``anonymous`` is the deliberate
    fallback for direct API hits without the frontend's X-User-Id header."""
    return (identity.get("user_id") or "anonymous").strip() or "anonymous"


@router.post("/protocol/json", response_model=ProtocolResponse)
async def create_protocol_json(
    body: ProtocolRequest,
    manager: PipelineManager = Depends(get_pipeline_manager),
    llm: BaseLLMProvider = Depends(get_llm),
    source_scores: dict = Depends(get_source_scores),
    history_paths: ProtocolHistoryPaths = Depends(get_protocol_history_paths),
    identity: dict = Depends(get_identity),
):
    started = time.monotonic()
    pipeline = await manager.get_pipeline(body.cancer_type)
    cancer_display = CANCER_TYPES.get(body.cancer_type, {}).get(
        "display_name", body.cancer_type,
    )
    retrieved = pipeline.retrieve(
        body.query, top_k=body.top_k, source_scores=source_scores,
    )
    protocol = await generate_protocol_json(
        llm, body.query, retrieved, cancer_type_display=cancer_display,
    )
    meta = {"id": "", "title": "", "version_count": 0}
    try:
        meta = save_initial(
            history_paths,
            user_id=_user_id(identity),
            protocol=protocol,
            summary_brief=body.query,
            source_nct_ids=[d.get("metadata", {}).get("nctId", "") for d in retrieved],
        )
    except Exception:
        logger.exception("Failed to auto-save protocol; returning ungated response")
    log_event(
        "protocol_generated",
        format="json",
        cancer_type=body.cancer_type,
        phase=protocol.get("phase"),
        study_type=protocol.get("study_type"),
        num_reference_trials=len(retrieved),
        protocol_id=meta.get("id") or None,
        latency_ms=int((time.monotonic() - started) * 1000),
        **identity,
    )
    return ProtocolResponse(
        protocol=protocol,
        reference_trials=[_to_source_doc(doc) for doc in retrieved],
        protocol_id=meta.get("id", ""),
        version=int(meta.get("version_count", 0) or 0),
        title=meta.get("title", ""),
    )


@router.post("/protocol/json-stream")
async def create_protocol_json_stream(
    body: ProtocolRequest,
    manager: PipelineManager = Depends(get_pipeline_manager),
    llm: BaseLLMProvider = Depends(get_llm),
    source_scores: dict = Depends(get_source_scores),
    history_paths: ProtocolHistoryPaths = Depends(get_protocol_history_paths),
    identity: dict = Depends(get_identity),
):
    """SSE streaming variant of /protocol/json.

    Events:
      - ``sources``: JSON array of reference trials (sent immediately)
      - ``token``:   raw text token from the LLM (many events)
      - ``done``:    final JSON with protocol, protocol_id, version, title
      - ``error``:   error message if something fails mid-stream
    """
    started = time.monotonic()
    pipeline = await manager.get_pipeline(body.cancer_type)
    cancer_display = CANCER_TYPES.get(body.cancer_type, {}).get(
        "display_name", body.cancer_type,
    )
    retrieved = pipeline.retrieve(
        body.query, top_k=body.top_k, source_scores=source_scores,
    )

    async def event_generator():
        source_docs = [_to_source_doc(doc).model_dump() for doc in retrieved]
        yield f"event: sources\ndata: {json.dumps(source_docs)}\n\n"

        raw_chunks: list[str] = []
        try:
            async for token in generate_protocol_json_stream(
                llm, body.query, retrieved, cancer_type_display=cancer_display,
            ):
                raw_chunks.append(token)
                yield f"event: token\ndata: {json.dumps(token)}\n\n"
        except Exception as e:
            logger.exception("Protocol stream failed")
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
            return

        raw = "".join(raw_chunks)
        try:
            protocol = _parse_protocol_raw(raw)
        except Exception as e:
            logger.exception("Protocol JSON parse failed")
            yield f"event: error\ndata: {json.dumps({'message': 'Failed to parse protocol JSON'})}\n\n"
            return

        meta = {"id": "", "title": "", "version_count": 0}
        try:
            meta = save_initial(
                history_paths,
                user_id=_user_id(identity),
                protocol=protocol,
                summary_brief=body.query,
                source_nct_ids=[d.get("metadata", {}).get("nctId", "") for d in retrieved],
            )
        except Exception:
            logger.exception("Failed to auto-save protocol")

        log_event(
            "protocol_generated",
            format="json_stream",
            cancer_type=body.cancer_type,
            phase=protocol.get("phase"),
            study_type=protocol.get("study_type"),
            num_reference_trials=len(retrieved),
            protocol_id=meta.get("id") or None,
            latency_ms=int((time.monotonic() - started) * 1000),
            **identity,
        )

        done_payload = {
            "protocol": protocol,
            "protocol_id": meta.get("id", ""),
            "version": int(meta.get("version_count", 0) or 0),
            "title": meta.get("title", ""),
        }
        yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/protocol/refine-stream")
async def refine_protocol_stream(
    body: RefineProtocolRequest,
    llm: BaseLLMProvider = Depends(get_llm),
    history_paths: ProtocolHistoryPaths = Depends(get_protocol_history_paths),
    identity: dict = Depends(get_identity),
):
    """SSE streaming variant of /protocol/refine."""
    started = time.monotonic()

    async def event_generator():
        raw_chunks: list[str] = []
        try:
            async for token in refine_protocol_json_stream(
                llm,
                current_protocol=body.protocol,
                refinement_messages=[m.model_dump() for m in body.refinement_messages],
                original_summary=body.original_summary,
            ):
                raw_chunks.append(token)
                yield f"event: token\ndata: {json.dumps(token)}\n\n"
        except Exception as e:
            logger.exception("Refine stream failed")
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
            return

        raw = "".join(raw_chunks)
        try:
            updated, note, changed = parse_refine_output(raw)
        except Exception as e:
            logger.exception("Refine output parse failed")
            yield f"event: error\ndata: {json.dumps({'message': 'Failed to parse refined protocol'})}\n\n"
            return

        turn_number = sum(1 for m in body.refinement_messages if m.role == "user")
        latest_user_request = ""
        for m in reversed(body.refinement_messages):
            if m.role == "user" and (m.content or "").strip():
                latest_user_request = m.content
                break

        new_meta: dict = {"id": body.protocol_id, "version_count": 0}
        if body.protocol_id:
            try:
                new_meta = save_revision(
                    history_paths,
                    user_id=_user_id(identity),
                    protocol_id=body.protocol_id,
                    protocol=updated,
                    user_request=latest_user_request,
                    assistant_note=note,
                    changed_fields=changed,
                )
            except FileNotFoundError:
                logger.warning("Refine-stream: protocol_id %s not found; saving as new", body.protocol_id)
                try:
                    new_meta = save_initial(
                        history_paths,
                        user_id=_user_id(identity),
                        protocol=updated,
                        summary_brief=body.original_summary,
                    )
                except Exception:
                    logger.exception("Refine-stream: fallback save_initial failed")
            except Exception:
                logger.exception("Refine-stream: save_revision failed")

        log_event(
            "protocol_refined",
            turn_number=turn_number,
            sections_changed=changed,
            num_sections_changed=len(changed) if changed else 0,
            protocol_id=new_meta.get("id") or None,
            version=int(new_meta.get("version_count", 0) or 0) or None,
            latency_ms=int((time.monotonic() - started) * 1000),
            **identity,
        )

        done_payload = {
            "protocol": updated,
            "assistant_message": note,
            "changed_fields": changed,
            "protocol_id": new_meta.get("id", ""),
            "version": int(new_meta.get("version_count", 0) or 0),
        }
        yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _protocol_filename(protocol: dict, extension: str) -> str:
    stem = (protocol.get("protocol_id") or "").strip() or "protocol"
    return f"{stem}.{extension}"


@router.post("/protocol/refine", response_model=RefineProtocolResponse)
async def refine_protocol(
    body: RefineProtocolRequest,
    llm: BaseLLMProvider = Depends(get_llm),
    history_paths: ProtocolHistoryPaths = Depends(get_protocol_history_paths),
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
    turn_number = sum(1 for m in body.refinement_messages if m.role == "user")
    latest_user_request = ""
    for m in reversed(body.refinement_messages):
        if m.role == "user" and (m.content or "").strip():
            latest_user_request = m.content
            break
    new_meta: dict = {"id": body.protocol_id, "version_count": 0}
    if body.protocol_id:
        try:
            new_meta = save_revision(
                history_paths,
                user_id=_user_id(identity),
                protocol_id=body.protocol_id,
                protocol=updated,
                user_request=latest_user_request,
                assistant_note=note,
                changed_fields=changed,
            )
        except FileNotFoundError:
            logger.warning(
                "Refine: protocol_id %s not found for user; saving as new entry",
                body.protocol_id,
            )
            try:
                new_meta = save_initial(
                    history_paths,
                    user_id=_user_id(identity),
                    protocol=updated,
                    summary_brief=body.original_summary,
                )
            except Exception:
                logger.exception("Refine: fallback save_initial failed too")
        except Exception:
            logger.exception("Refine: save_revision failed")
    log_event(
        "protocol_refined",
        turn_number=turn_number,
        sections_changed=changed,
        num_sections_changed=len(changed) if changed else 0,
        protocol_id=new_meta.get("id") or None,
        version=int(new_meta.get("version_count", 0) or 0) or None,
        latency_ms=int((time.monotonic() - started) * 1000),
        **identity,
    )
    return RefineProtocolResponse(
        protocol=updated,
        assistant_message=note,
        changed_fields=changed,
        protocol_id=new_meta.get("id", ""),
        version=int(new_meta.get("version_count", 0) or 0),
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


# ---------------------------------------------------------------------------
# Protocol history (per-user list / load for the "Recent protocols" UI)
# ---------------------------------------------------------------------------

@router.get("/protocols", response_model=ProtocolListResponse)
async def list_user_protocols(
    history_paths: ProtocolHistoryPaths = Depends(get_protocol_history_paths),
    identity: dict = Depends(get_identity),
):
    rows = list_protocols(history_paths, _user_id(identity))
    return ProtocolListResponse(protocols=[ProtocolEntry(**r) for r in rows])


@router.get("/protocols/{protocol_id}", response_model=ProtocolLoadResponse)
async def load_user_protocol(
    protocol_id: str,
    version: int | None = None,
    include_versions: bool = False,
    history_paths: ProtocolHistoryPaths = Depends(get_protocol_history_paths),
    identity: dict = Depends(get_identity),
):
    user_id = _user_id(identity)
    try:
        protocol, meta = load_protocol(
            history_paths,
            user_id=user_id,
            protocol_id=protocol_id,
            version=version,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    all_versions: list[ProtocolVersionEntry] = []
    if include_versions:
        try:
            for entry in load_all_versions(history_paths, user_id, protocol_id):
                all_versions.append(ProtocolVersionEntry(**entry))
        except FileNotFoundError:
            logger.warning("load_all_versions failed for %s/%s", user_id, protocol_id)

    return ProtocolLoadResponse(
        protocol=protocol,
        meta=ProtocolEntry(**meta),
        version=version if version is not None else int(meta.get("version_count", 1)),
        all_versions=all_versions,
    )


@router.post("/protocol/pdf")
async def create_protocol_pdf(
    body: DocxRequest,
    identity: dict = Depends(get_identity),
):
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
