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
    """Construct the LLM client + RAG pipeline + feedback paths once at startup.

    Everything lives on app.state so request handlers can pull what they need
    via FastAPI dependencies; there's no per-request construction cost.
    """
    llm = get_llm_provider()
    pipeline = ClinicalTrialRAG(
        csv_path=settings.csv_path,
        llm=llm,
        retriever_type=settings.retriever_type,
        voyage_api_key=settings.voyage_api_key,
    )
    supabase_client = _init_supabase()
    analytics.configure(supabase_client)
    data_dir = Path(settings.csv_path).resolve().parent
    feedback_paths = FeedbackPaths.from_data_dir(data_dir, supabase_client=supabase_client)
    app.state.pipeline = pipeline
    app.state.llm = llm
    app.state.feedback_paths = feedback_paths
    app.state.aliases = load_aliases(feedback_paths)
    app.state.source_scores = compute_source_scores(feedback_paths)
    app.state.examples = FeedbackExampleStore.load(feedback_paths)
    app.state.protocol_history_paths = ProtocolHistoryPaths.from_data_dir(data_dir)
    yield


def get_pipeline(request: Request) -> ClinicalTrialRAG:
    return request.app.state.pipeline


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
