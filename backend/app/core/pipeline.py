import logging

import anthropic

from .data import build_documents, load_clinical_trials
from .generation import generate_answer, generate_answer_stream
from .retriever import TFIDFRetriever

logger = logging.getLogger(__name__)


class ClinicalTrialRAG:
    """End-to-end RAG pipeline for clinical trial Q&A."""

    def __init__(
        self,
        csv_path: str,
        client: anthropic.AsyncAnthropic,
        model: str = "claude-sonnet-4-20250514",
    ):
        logger.info("Loading clinical trial data from %s", csv_path)
        self.df = load_clinical_trials(csv_path)
        self.documents = build_documents(self.df)
        self.retriever = TFIDFRetriever(self.documents)
        self.client = client
        self.model = model
        logger.info("RAG pipeline ready (%d trials indexed)", len(self.documents))

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieve relevant trials without generating an answer."""
        return self.retriever.retrieve(query, top_k=top_k)

    async def ask(self, query: str, top_k: int = 5) -> tuple[str, list[dict]]:
        """Retrieve trials and generate a full answer. Returns (answer, sources)."""
        retrieved = self.retriever.retrieve(query, top_k=top_k)
        if not retrieved:
            return "No relevant clinical trials found for your query.", []
        answer = await generate_answer(self.client, query, retrieved, self.model)
        return answer, retrieved

    async def ask_stream(self, query: str, top_k: int = 5):
        """Retrieve trials and stream answer tokens. Returns (stream_generator, sources)."""
        retrieved = self.retriever.retrieve(query, top_k=top_k)
        if not retrieved:
            return None, []
        stream = generate_answer_stream(self.client, query, retrieved, self.model)
        return stream, retrieved
