from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request

import anthropic

from .config import settings
from .core.feedback import FeedbackPaths, load_aliases
from .core.pipeline import ClinicalTrialRAG


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load data and build TF-IDF index at startup."""
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    pipeline = ClinicalTrialRAG(
        csv_path=settings.csv_path,
        client=client,
        model=settings.claude_model,
        retriever_type=settings.retriever_type,
        voyage_api_key=settings.voyage_api_key,
    )
    # Feedback log + alias dictionary live next to the trial CSV.
    data_dir = Path(settings.csv_path).resolve().parent
    feedback_paths = FeedbackPaths.from_data_dir(data_dir)
    app.state.pipeline = pipeline
    app.state.client = client
    app.state.feedback_paths = feedback_paths
    app.state.aliases = load_aliases(feedback_paths)
    yield


def get_pipeline(request: Request) -> ClinicalTrialRAG:
    return request.app.state.pipeline


def get_client(request: Request) -> anthropic.AsyncAnthropic:
    return request.app.state.client


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
