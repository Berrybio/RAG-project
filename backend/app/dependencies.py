import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request

from .config import settings
from .core import analytics
from .core.feedback import FeedbackPaths, load_aliases
from .core.feedback_examples import FeedbackExampleStore
from .core.feedback_reranker import compute_source_scores
from .core.llm import BaseLLMProvider, get_llm_provider
from .core.pipeline import ClinicalTrialRAG
from .core.pipeline_manager import PipelineManager
from .core.protocol_history import ProtocolHistoryPaths

logger = logging.getLogger(__name__)


def _init_supabase():
    if not settings.supabase_url or not settings.supabase_service_role_key:
        return None
    try:
        from supabase import create_client
        client = create_client(settings.supabase_url, settings.supabase_service_role_key)
        logger.info("Supabase client initialized for persistent feedback storage")
        return client
    except Exception:
        logger.exception("Failed to initialize Supabase client; falling back to file storage")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Construct the LLM client + PipelineManager + feedback paths once at startup.

    Everything lives on app.state so request handlers can pull what they need
    via FastAPI dependencies; there's no per-request construction cost.
    """
    llm = get_llm_provider()

    # Build the multi-cancer PipelineManager (GCS-only).
    preload_list = [
        ct.strip()
        for ct in settings.preload_cancer_types.split(",")
        if ct.strip()
    ]
    manager = PipelineManager(
        bucket=settings.gcs_bucket,
        llm=llm,
        retriever_type=settings.retriever_type,
        voyage_api_key=settings.voyage_api_key,
        max_loaded=settings.max_loaded_pipelines,
    )
    manager.preload(preload_list)

    supabase_client = _init_supabase()
    analytics.configure(supabase_client)

    # Feedback / protocol history use a data directory.  With GCS-backed
    # pipelines the CSV lives in a temp dir; use the first preloaded
    # pipeline's csv_path parent as the canonical data dir.
    default_ct = preload_list[0] if preload_list else settings.default_cancer_type
    try:
        default_pipeline = await manager.get_pipeline(default_ct)
        data_dir = Path(default_pipeline.csv_path).resolve().parent
    except Exception:
        logger.warning("Could not resolve data_dir from default pipeline; using cwd")
        data_dir = Path.cwd() / "data"
        data_dir.mkdir(exist_ok=True)

    feedback_paths = FeedbackPaths.from_data_dir(data_dir, supabase_client=supabase_client)

    app.state.pipeline_manager = manager
    app.state.llm = llm
    app.state.feedback_paths = feedback_paths
    app.state.aliases = load_aliases(feedback_paths)
    app.state.source_scores = compute_source_scores(feedback_paths)
    app.state.examples = FeedbackExampleStore.load(feedback_paths)
    app.state.protocol_history_paths = ProtocolHistoryPaths.from_data_dir(data_dir)
    yield


# ---------------------------------------------------------------------------
# FastAPI dependency helpers
# ---------------------------------------------------------------------------

def get_pipeline_manager(request: Request) -> PipelineManager:
    return request.app.state.pipeline_manager


def get_llm(request: Request) -> BaseLLMProvider:
    """Provider-agnostic LLM dependency. Replaces the old `get_client`."""
    return request.app.state.llm


def get_aliases(request: Request) -> dict[str, str]:
    return getattr(request.app.state, "aliases", {})


def get_source_scores(request: Request) -> dict[str, float]:
    return getattr(request.app.state, "source_scores", {})


def get_examples(request: Request) -> "FeedbackExampleStore | None":
    return getattr(request.app.state, "examples", None)


def get_protocol_history_paths(request: Request) -> ProtocolHistoryPaths:
    return request.app.state.protocol_history_paths


def get_identity(request: Request) -> dict[str, str | None]:
    """Pull the anonymous user_id + session_id the frontend sets in headers.

    Returns a dict so endpoints can `**identity` into log_event(). Both keys
    may be None for direct API hits (curl, smoke tests) — that's fine; the
    analytics layer drops None fields.
    """
    return {
        "user_id": request.headers.get("x-user-id"),
        "session_id": request.headers.get("x-session-id"),
    }
