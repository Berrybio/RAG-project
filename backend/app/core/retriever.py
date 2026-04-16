import logging
from abc import ABC, abstractmethod

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    """Abstract base class - all retrievers expose the same .retrieve() API."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Return top_k docs as dicts with 'doc_id', 'text', 'metadata', 'score'."""
        ...


class TFIDFRetriever(BaseRetriever):
    """
    Sparse retriever using TF-IDF vectors + cosine similarity.
    Strengths : exact keyword matching, fast, zero downloads.
    Weaknesses: misses synonyms and paraphrases.
    """

    def __init__(self, documents: list[dict]):
        self.documents = documents
        self.texts = [doc["text"] for doc in documents]

        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)
        logger.info(
            "TF-IDF indexed %d documents (%d features)",
            len(self.texts),
            self.tfidf_matrix.shape[1],
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = scores.argsort()[::-1][:top_k]

        return [
            {**self.documents[i], "score": float(scores[i])}
            for i in top_indices
            if scores[i] > 0
        ]


class VoyageRetriever(BaseRetriever):
    """
    Dense retriever using Voyage AI embeddings + cosine similarity.
    Strengths : semantic understanding, synonyms, paraphrases, domain knowledge.
    Weaknesses: requires API calls for embedding, slower indexing.
    """

    EMBED_MODEL = "voyage-3"
    BATCH_SIZE = 128  # Voyage API limit per request
    MAX_DOC_CHARS = 16000  # truncate long docs to stay within token limits

    def __init__(self, documents: list[dict], api_key: str):
        import voyageai

        self.documents = documents
        self.client = voyageai.Client(api_key=api_key)

        texts = [doc["text"][:self.MAX_DOC_CHARS] for doc in documents]

        # Embed documents in batches
        logger.info(
            "Embedding %d documents with Voyage AI (%s) in batches of %d...",
            len(texts), self.EMBED_MODEL, self.BATCH_SIZE,
        )
        all_embeddings = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i : i + self.BATCH_SIZE]
            result = self.client.embed(batch, model=self.EMBED_MODEL, input_type="document")
            all_embeddings.extend(result.embeddings)
            if (i // self.BATCH_SIZE) % 10 == 0:
                logger.info("  Embedded %d / %d documents", min(i + self.BATCH_SIZE, len(texts)), len(texts))

        self.embeddings = np.array(all_embeddings, dtype=np.float32)
        # Pre-normalize for fast cosine similarity via dot product
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = self.embeddings / norms

        logger.info(
            "Voyage AI indexed %d documents (dim=%d)",
            len(self.documents), self.embeddings.shape[1],
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        result = self.client.embed([query], model=self.EMBED_MODEL, input_type="query")
        query_vec = np.array(result.embeddings[0], dtype=np.float32)
        query_vec = query_vec / np.linalg.norm(query_vec)

        # Cosine similarity via dot product (vectors are pre-normalized)
        scores = self.embeddings @ query_vec
        top_indices = scores.argsort()[::-1][:top_k]

        return [
            {**self.documents[i], "score": float(scores[i])}
            for i in top_indices
            if scores[i] > 0
        ]
