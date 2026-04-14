from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.core.data import build_documents
from app.core.retriever import TFIDFRetriever
from tests.conftest import SAMPLE_DATA
import pandas as pd


@pytest.fixture
def client():
    """Create a test client with mocked pipeline."""
    from app.main import app
    from app.core.pipeline import ClinicalTrialRAG

    # Build real retriever from sample data
    df = pd.DataFrame(SAMPLE_DATA)
    documents = build_documents(df)
    retriever = TFIDFRetriever(documents)

    # Mock the pipeline
    mock_pipeline = MagicMock(spec=ClinicalTrialRAG)
    mock_pipeline.documents = documents
    mock_pipeline.model = "test-model"
    mock_pipeline.retriever = retriever
    mock_pipeline.retrieve = retriever.retrieve

    mock_client = AsyncMock()
    app.state.pipeline = mock_pipeline
    app.state.client = mock_client

    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["trials_loaded"] == 3


def test_search_endpoint(client):
    mock_pipeline = client.app.state.pipeline
    mock_pipeline.ask = AsyncMock(
        return_value=(
            "Found CDK4/6 inhibitor trials.",
            [
                {
                    "doc_id": "NCT00000001",
                    "text": "test",
                    "score": 0.95,
                    "metadata": {
                        "nctId": "NCT00000001",
                        "title": "CDK4/6 Trial",
                        "phases": "PHASE2",
                        "status": "RECRUITING",
                        "conditions": "Breast Cancer",
                        "interventionName": "Palbociclib",
                        "enrollmentCont": 100,
                        "sex": "FEMALE",
                        "minimumAge": "18 Years",
                        "locationInfo": "New York",
                        "contactInfo": "Dr. Smith",
                    },
                }
            ],
        )
    )

    response = client.post("/api/search", json={"query": "CDK4/6", "top_k": 5})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["sources"]) == 1
    assert data["sources"][0]["nct_id"] == "NCT00000001"


def test_search_stream_endpoint(client):
    mock_pipeline = client.app.state.pipeline

    async def mock_stream():
        yield "test token"

    mock_pipeline.ask_stream = AsyncMock(
        return_value=(
            mock_stream(),
            [
                {
                    "doc_id": "NCT00000001",
                    "text": "test",
                    "score": 0.95,
                    "metadata": {
                        "nctId": "NCT00000001",
                        "title": "CDK4/6 Trial",
                        "phases": "PHASE2",
                        "status": "RECRUITING",
                        "conditions": "Breast Cancer",
                        "interventionName": "Palbociclib",
                        "enrollmentCont": 100,
                        "sex": "FEMALE",
                        "minimumAge": "18 Years",
                        "locationInfo": "New York",
                        "contactInfo": "Dr. Smith",
                    },
                }
            ],
        )
    )

    response = client.get("/api/search/stream?query=CDK4/6&top_k=5")
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]


def test_protocol_docx_endpoint(client):
    sample_protocol = {
        "title": "Test Protocol",
        "protocol_id": "TEST-001",
        "phase": "Phase II",
    }
    response = client.post("/api/protocol/docx", json={"protocol": sample_protocol})
    assert response.status_code == 200
    assert "application/vnd.openxmlformats" in response.headers["content-type"]
