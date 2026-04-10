from abc import ABC, abstractmethod
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# 2. RETRIEVER BASE CLASS
# ---------------------------------------------------------------------------

class BaseRetriever(ABC):
    """Abstract base class — all retrievers expose the same .retrieve() API."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Return top_k docs as dicts with 'doc_id', 'text', 'metadata', 'score'."""
        ...


# ---------------------------------------------------------------------------
# 3a. RETRIEVER — TF-IDF  (sparse, keyword-based)
# ---------------------------------------------------------------------------

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
        print(f"  [TF-IDF] Indexed {len(self.texts)} documents "
              f"({self.tfidf_matrix.shape[1]} features)")

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = scores.argsort()[::-1][:top_k]

        return [
            {**self.documents[i], "score": float(scores[i])}
            for i in top_indices
            if scores[i] > 0
        ]

