"""Few-shot examples mined from up-voted (query, answer) pairs.

When the planner answers a new question, we look up the most semantically
similar past question that the user gave a 👍 to and inject that prior
(question, answer) pair into the system prompt. The model then has a concrete
example of the structure / tone the user has already endorsed.

The store is built at app startup and rebuilt on each ``/api/feedback`` POST,
so a fresh up-vote is usable on the very next reply. Below ``_MIN_SIMILARITY``
we'd rather show nothing than confuse the planner with off-topic prose.

Similarity is TF-IDF + cosine — no extra Voyage call per request, and the
up-voted pool is small enough that fitting a vectorizer on every refresh is
near-free.
"""
from __future__ import annotations

import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .feedback import FeedbackPaths, read_feedback

logger = logging.getLogger(__name__)

# Hard caps. Keep the few-shot block compact: too many examples bloat context
# and start dictating answer structure even when the user wants something
# different.
_MAX_EXAMPLES = 2
# Skip very short answers — a one-line acknowledgement isn't useful as a
# few-shot exemplar. The threshold also conveniently filters the common
# "[CHOICES] only" replies which are pure clarification.
_MIN_ANSWER_CHARS = 200
# Trim each rendered example so a single long past answer doesn't dominate
# the system prompt.
_MAX_ANSWER_CHARS_IN_PROMPT = 1200
# Below this cosine similarity, the past question is too unrelated to be
# useful as an example. Picked empirically — TF-IDF cosine of 0.15 roughly
# corresponds to "shares one or two non-stopword tokens".
_MIN_SIMILARITY = 0.15


class FeedbackExampleStore:
    """In-memory index of up-voted (query, answer) pairs.

    Cheap to (re)build because the feedback log is human-scale. Empty when no
    up-votes have been collected — ``pick`` is then a no-op.
    """

    def __init__(self, examples: list[dict[str, str]]):
        self.examples = examples
        self._vec: TfidfVectorizer | None = None
        self._matrix = None
        if examples:
            queries = [ex["query"] for ex in examples]
            try:
                self._vec = TfidfVectorizer(max_features=4096).fit(queries)
                self._matrix = self._vec.transform(queries)
            except ValueError:
                # All-empty / stop-word-only corpus — fall back to no matching.
                logger.warning("FeedbackExampleStore: vectorizer fit failed; pick() will return []")
                self._vec = None
                self._matrix = None

    @classmethod
    def load(cls, paths: FeedbackPaths) -> "FeedbackExampleStore":
        """Read the feedback log and keep only up-voted, substantive replies."""
        rows = read_feedback(paths)
        examples: list[dict[str, str]] = []
        for row in rows:
            if row.get("type") != "rating_up":
                continue
            ctx = row.get("context") or {}
            q = (ctx.get("query") or "").strip()
            a = (ctx.get("assistant_message") or "").strip()
            if not q or not a or len(a) < _MIN_ANSWER_CHARS:
                continue
            examples.append({"query": q, "answer": a})
        return cls(examples)

    def pick(self, query: str, top_n: int = _MAX_EXAMPLES) -> list[dict[str, str]]:
        """Return the most semantically similar up-voted pairs to ``query``."""
        if not self.examples or self._matrix is None or not query.strip():
            return []
        try:
            qv = self._vec.transform([query])
        except ValueError:
            return []
        sims = cosine_similarity(qv, self._matrix)[0]
        order = sims.argsort()[::-1]
        picked: list[dict[str, str]] = []
        for idx in order[: max(top_n, 0)]:
            if sims[idx] < _MIN_SIMILARITY:
                break
            picked.append(self.examples[int(idx)])
        return picked


def format_examples_block(examples: list[dict[str, str]]) -> str:
    """Render up-voted examples as a system-prompt addition. Empty if none."""
    if not examples:
        return ""
    lines = [
        "Examples of past replies the user rated as helpful — match this "
        "structure and depth when the new question is similar:",
    ]
    for i, ex in enumerate(examples, 1):
        answer = ex["answer"]
        if len(answer) > _MAX_ANSWER_CHARS_IN_PROMPT:
            answer = answer[:_MAX_ANSWER_CHARS_IN_PROMPT].rstrip() + "…"
        lines.append(f"\nExample {i}")
        lines.append(f"Q: {ex['query']}")
        lines.append(f"A: {answer}")
    return "\n".join(lines)
