import hashlib
import logging
from abc import ABC, abstractmethod
from pathlib import Path

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
    DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "embeddings_cache"

    def __init__(
        self,
        documents: list[dict],
        api_key: str,
        use_cache: bool = True,
        cache_dir: Path | None = None,
    ):
        import voyageai

        self.documents = documents
        self.cache_dir = cache_dir or self.DEFAULT_CACHE_DIR
        self.client = voyageai.Client(api_key=api_key)

        texts = [doc["text"][:self.MAX_DOC_CHARS] for doc in documents]

        cache_path = self._cache_path(texts) if use_cache else None
        cached = self._load_cache(cache_path) if cache_path else None

        if cached is not None:
            self.embeddings = cached
            logger.info(
                "Loaded cached Voyage embeddings for %d documents (dim=%d) from %s",
                len(self.documents), self.embeddings.shape[1], cache_path.name,
            )
            return

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

        if cache_path is not None:
            self._save_cache(cache_path, self.embeddings)

    def _cache_path(self, texts: list[str]) -> Path:
        """Derive a deterministic cache file path from corpus content + model."""
        hasher = hashlib.sha256()
        hasher.update(self.EMBED_MODEL.encode("utf-8"))
        hasher.update(str(len(texts)).encode("utf-8"))
        for t in texts:
            hasher.update(t.encode("utf-8", errors="replace"))
            hasher.update(b"\x00")
        digest = hasher.hexdigest()[:16]
        return self.cache_dir / f"voyage_{self.EMBED_MODEL}_{len(texts)}_{digest}.npz"

    @staticmethod
    def _load_cache(path: Path) -> np.ndarray | None:
        if not path.exists():
            return None
        try:
            with np.load(path) as data:
                return data["embeddings"].astype(np.float32)
        except Exception as e:
            logger.warning("Failed to load embedding cache %s: %s", path.name, e)
            return None

    @staticmethod
    def _save_cache(path: Path, embeddings: np.ndarray) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, embeddings=embeddings)
            logger.info("Saved Voyage embeddings cache to %s", path)
        except Exception as e:
            logger.warning("Failed to save embedding cache %s: %s", path.name, e)

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
