from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request

from .config import settings
from .core.feedback import FeedbackPaths, load_aliases
from .core.llm import BaseLLMProvider, get_llm_provider
from .core.pipeline import ClinicalTrialRAG


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
    # Feedback log + alias dictionary live next to the trial CSV.
    data_dir = Path(settings.csv_path).resolve().parent
    feedback_paths = FeedbackPaths.from_data_dir(data_dir)
    app.state.pipeline = pipeline
    app.state.llm = llm
    app.state.feedback_paths = feedback_paths
    app.state.aliases = load_aliases(feedback_paths)
    yield


def get_pipeline(request: Request) -> ClinicalTrialRAG:
    return request.app.state.pipeline


def get_llm(request: Request) -> BaseLLMProvider:
    """Provider-agnostic LLM dependency. Replaces the old `get_client`."""
    return request.app.state.llm


def get_aliases(request: Request) -> dict[str, str]:
    return getattr(request.app.state, "aliases", {})


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
