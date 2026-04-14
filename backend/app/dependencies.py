from contextlib import asynccontextmanager
from fastapi import FastAPI, Request

import anthropic

from .config import settings
from .core.pipeline import ClinicalTrialRAG


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load data and build TF-IDF index at startup."""
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    pipeline = ClinicalTrialRAG(
        csv_path=settings.csv_path,
        client=client,
        model=settings.claude_model,
    )
    app.state.pipeline = pipeline
    app.state.client = client
    yield


def get_pipeline(request: Request) -> ClinicalTrialRAG:
    return request.app.state.pipeline


def get_client(request: Request) -> anthropic.AsyncAnthropic:
    return request.app.state.client
