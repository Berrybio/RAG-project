import logging
import re

import anthropic

from .data import build_documents, load_clinical_trials
from .generation import generate_answer, generate_answer_stream
from .retriever import TFIDFRetriever, VoyageRetriever

logger = logging.getLogger(__name__)

# Country aliases → canonical name used in locationCountries field.
# Keys are matched case-insensitively against the query.
_COUNTRY_ALIASES: dict[str, str] = {
    "united states": "United States",
    "usa": "United States",
    "u.s.a.": "United States",
    "u.s.": "United States",
    "us": "United States",
    "america": "United States",
    "american": "United States",
    "united kingdom": "United Kingdom",
    "uk": "United Kingdom",
    "britain": "United Kingdom",
    "england": "United Kingdom",
    "china": "China",
    "chinese": "China",
    "canada": "Canada",
    "canadian": "Canada",
    "france": "France",
    "french": "France",
    "germany": "Germany",
    "german": "Germany",
    "japan": "Japan",
    "japanese": "Japan",
    "south korea": "South Korea",
    "korea": "South Korea",
    "korean": "South Korea",
    "australia": "Australia",
    "australian": "Australia",
    "italy": "Italy",
    "italian": "Italy",
    "spain": "Spain",
    "spanish": "Spain",
    "netherlands": "Netherlands",
    "dutch": "Netherlands",
    "belgium": "Belgium",
    "switzerland": "Switzerland",
    "swiss": "Switzerland",
    "india": "India",
    "indian": "India",
    "brazil": "Brazil",
    "brazilian": "Brazil",
    "mexico": "Mexico",
    "mexican": "Mexico",
    "turkey": "Turkey (Türkiye)",
    "turkish": "Turkey (Türkiye)",
}


def _extract_countries(query: str) -> list[str]:
    """Return canonical country names mentioned in the query, longest-alias-first."""
    q = query.lower()
    found: list[str] = []
    # Sort aliases by length desc so "united states" matches before "us".
    for alias in sorted(_COUNTRY_ALIASES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", q):
            canonical = _COUNTRY_ALIASES[alias]
            if canonical not in found:
                found.append(canonical)
            # Remove matched span so shorter aliases ("us") don't double-match inside "united states".
            q = re.sub(rf"\b{re.escape(alias)}\b", " ", q)
    return found


def _doc_matches_countries(doc: dict, countries: list[str]) -> bool:
    """True if any of the target countries appears in the doc's locationCountries."""
    haystack = (doc.get("metadata", {}).get("locationCountries") or "").lower()
    return any(c.lower() in haystack for c in countries)


class ClinicalTrialRAG:
    """End-to-end RAG pipeline for clinical trial Q&A."""

    def __init__(
        self,
        csv_path: str,
        client: anthropic.AsyncAnthropic,
        model: str = "claude-sonnet-4-20250514",
        retriever_type: str = "voyage",
        voyage_api_key: str = "",
    ):
        logger.info("Loading clinical trial data from %s", csv_path)
        self.df = load_clinical_trials(csv_path)
        self.documents = build_documents(self.df)

        if retriever_type == "voyage" and voyage_api_key:
            logger.info("Using Voyage AI dense retriever")
            self.retriever = VoyageRetriever(self.documents, api_key=voyage_api_key)
        else:
            if retriever_type == "voyage" and not voyage_api_key:
                logger.warning("Voyage API key not set — falling back to TF-IDF retriever")
            logger.info("Using TF-IDF sparse retriever")
            self.retriever = TFIDFRetriever(self.documents)

        self.client = client
        self.model = model
        logger.info("RAG pipeline ready (%d trials indexed)", len(self.documents))

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieve relevant trials without generating an answer."""
        return self._retrieve_with_country_filter(query, top_k)

    def _retrieve_with_country_filter(self, query: str, top_k: int) -> list[dict]:
        """Retrieve top_k docs, hard-filtering by country if the query names one."""
        countries = _extract_countries(query)
        if not countries:
            return self.retriever.retrieve(query, top_k=top_k)

        # Overfetch, then keep only docs whose locations include a requested country.
        pool = self.retriever.retrieve(query, top_k=max(top_k * 10, 50))
        filtered = [d for d in pool if _doc_matches_countries(d, countries)]
        if not filtered:
            logger.info(
                "Country filter %s matched 0 trials in top pool — falling back to unfiltered results",
                countries,
            )
            return pool[:top_k]
        logger.info(
            "Country filter %s kept %d / %d retrieved trials",
            countries, len(filtered), len(pool),
        )
        return filtered[:top_k]

    async def ask(self, query: str, top_k: int = 5) -> tuple[str, list[dict]]:
        """Retrieve trials and generate a full answer. Returns (answer, sources)."""
        retrieved = self._retrieve_with_country_filter(query, top_k)
        if not retrieved:
            return "No relevant clinical trials found for your query.", []
        answer = await generate_answer(self.client, query, retrieved, self.model)
        return answer, retrieved

    async def ask_stream(self, query: str, top_k: int = 5):
        """Retrieve trials and stream answer tokens. Returns (stream_generator, sources)."""
        retrieved = self._retrieve_with_country_filter(query, top_k)
        if not retrieved:
            return None, []
        stream = generate_answer_stream(self.client, query, retrieved, self.model)
        return stream, retrieved
