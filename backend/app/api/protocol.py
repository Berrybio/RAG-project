import logging
import time

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

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
    get_pipeline,
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
from ..core.protocol import (
    generate_protocol_json,
    refine_protocol_json,
    build_protocol_docx,
    build_protocol_pdf,
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
    pipeline: ClinicalTrialRAG = Depends(get_pipeline),
    llm: BaseLLMProvider = Depends(get_llm),
    source_scores: dict = Depends(get_source_scores),
    history_paths: ProtocolHistoryPaths = Depends(get_protocol_history_paths),
    identity: dict = Depends(get_identity),
):
    started = time.monotonic()
    retrieved = pipeline.retrieve(
        body.query, top_k=body.top_k, source_scores=source_scores,
    )
    protocol = await generate_protocol_json(llm, body.query, retrieved)
    # Auto-save as v1 in the per-user history. Best-effort: a failure here
    # must never break the user's request — we'd rather the protocol came
    # back without an id than fail the call.
    meta = {"id": "", "title": "", "version_count": 0}
    try:
        meta = save_initial(
            history_paths,
            user_id=_user_id(identity),
            protocol=protocol,
            summary_brief=body.query,
            source_nct_ids=[d.get("metadata", {}).get("nctId", "") for d in retrieved],
        )
    except Exception:  # pragma: no cover - persistence is non-critical
        logger.exception("Failed to auto-save protocol; returning ungated response")
    log_event(
        "protocol_generated",
        format="json",
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


def _protocol_filename(protocol: dict, extension: str) -> str:
    """Build a safe filename, falling back to 'protocol' when protocol_id is empty."""
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
    # turn_number = how many refinement rounds the user has done so far
    turn_number = sum(1 for m in body.refinement_messages if m.role == "user")
    # Save as the next version under the same history entry. Best-effort:
    # if the protocol_id is missing or the meta file vanished (manual cleanup),
    # we fall back to saving as a new initial entry so the work isn't lost.
    # Capture the user's latest request + assistant note + changed fields so
    # the saved change_log is rich enough to rebuild version chips on a
    # future session.
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
            except Exception:  # pragma: no cover
                logger.exception("Refine: fallback save_initial failed too")
        except Exception:  # pragma: no cover
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
    """Return this user's saved protocols, newest-first.

    The frontend renders these as a strip above the planner so the clinician
    can pick up where they left off across sessions.
    """
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
    """Load a stored protocol for editing.

    With ``include_versions=true`` (used by the planner UI on history load),
    the response also carries every version of the protocol with its
    change-log metadata so the version chips + compare-any-two flow work
    across sessions, not just within the current session's refinements.
    """
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
            # Meta existed for the latest-version load above; the per-version
            # load shouldn't normally fail. If it does we just return an empty
            # all_versions list rather than 500'ing the request.
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
