"""Manages per-cancer-type ClinicalTrialRAG instances with LRU eviction.

The ``PipelineManager`` is the single source of truth for which cancer
types are loaded and how to obtain a ``ClinicalTrialRAG`` for any
supported type.  It eagerly pre-loads a configurable set of types at
startup and lazy-loads the rest on first request, evicting the
least-recently-used pipeline when the memory cap is reached.
"""
from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from pathlib import Path

from ..cancer_registry import CANCER_TYPES, gcs_csv_blob, gcs_embeddings_prefix
from .gcs_storage import download_cancer_type_data
from .llm import BaseLLMProvider
from .pipeline import ClinicalTrialRAG

logger = logging.getLogger(__name__)


class PipelineManager:
    """Registry of per-cancer-type RAG pipelines with LRU caching."""

    def __init__(
        self,
        bucket: str,
        llm: BaseLLMProvider,
        retriever_type: str = "voyage",
        voyage_api_key: str = "",
        max_loaded: int = 5,
        enabled_types: set[str] | None = None,
    ):
        self._bucket = bucket
        self._llm = llm
        self._retriever_type = retriever_type
        self._voyage_api_key = voyage_api_key
        self._max_loaded = max_loaded
        # Which cancer types are available for queries.  ``None`` = all in
        # CANCER_TYPES; otherwise a subset (e.g. from registry.json).
        self._enabled_types: set[str] = enabled_types or set(CANCER_TYPES.keys())
        self._pipelines: OrderedDict[str, ClinicalTrialRAG] = OrderedDict()
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Startup: eagerly pre-load the configured types
    # ------------------------------------------------------------------

    def preload(self, cancer_types: list[str]) -> None:
        """Eagerly load pipelines for the listed cancer types.

        Called synchronously during the FastAPI lifespan startup.
        """
        for ct in cancer_types:
            if ct not in self._enabled_types:
                logger.warning(
                    "Skipping preload of %s — not in enabled types", ct,
                )
                continue
            try:
                self._load_pipeline(ct)
            except Exception:
                logger.exception("Failed to preload pipeline for %s", ct)

    # ------------------------------------------------------------------
    # Core accessor: get (or lazy-load) a pipeline
    # ------------------------------------------------------------------

    async def get_pipeline(self, cancer_type: str) -> ClinicalTrialRAG:
        """Return the pipeline for *cancer_type*, loading on demand if needed."""
        if cancer_type not in self._enabled_types:
            raise ValueError(
                f"Cancer type {cancer_type!r} is not enabled. "
                f"Available: {sorted(self._enabled_types)}"
            )

        # Fast path: already loaded — just bump LRU position.
        if cancer_type in self._pipelines:
            self._pipelines.move_to_end(cancer_type)
            return self._pipelines[cancer_type]

        # Slow path: download from GCS + build pipeline (serialized).
        async with self._lock:
            # Double-check after acquiring lock.
            if cancer_type in self._pipelines:
                self._pipelines.move_to_end(cancer_type)
                return self._pipelines[cancer_type]
            logger.info("Lazy-loading pipeline for %s ...", cancer_type)
            self._load_pipeline(cancer_type)
            return self._pipelines[cancer_type]

    # ------------------------------------------------------------------
    # Internal: download data + instantiate a ClinicalTrialRAG
    # ------------------------------------------------------------------

    def _load_pipeline(self, cancer_type: str) -> None:
        csv_blob = gcs_csv_blob(cancer_type)
        emb_prefix = gcs_embeddings_prefix(cancer_type)
        csv_path, cache_dir = download_cancer_type_data(
            self._bucket, cancer_type, csv_blob, emb_prefix,
        )
        pipeline = ClinicalTrialRAG(
            csv_path=str(csv_path),
            llm=self._llm,
            retriever_type=self._retriever_type,
            voyage_api_key=self._voyage_api_key,
            embeddings_cache_dir=cache_dir,
        )
        self._pipelines[cancer_type] = pipeline
        self._pipelines.move_to_end(cancer_type)

        # Evict LRU entries if we're over the cap.
        while len(self._pipelines) > self._max_loaded:
            evicted_key, _ = self._pipelines.popitem(last=False)
            logger.info("Evicted LRU pipeline: %s", evicted_key)

        logger.info(
            "Pipeline for %s loaded (%d trials, %d/%d slots used)",
            cancer_type,
            len(pipeline.documents),
            len(self._pipelines),
            self._max_loaded,
        )

    # ------------------------------------------------------------------
    # Metadata accessors (for /api/cancer-types, /health, etc.)
    # ------------------------------------------------------------------

    @property
    def available_cancer_types(self) -> list[dict]:
        """Return display metadata for all enabled cancer types."""
        result = []
        for key in CANCER_TYPES:
            if key not in self._enabled_types:
                continue
            info = CANCER_TYPES[key]
            loaded = key in self._pipelines
            trial_count = (
                len(self._pipelines[key].documents) if loaded else None
            )
            result.append({
                "key": key,
                "display_name": info["display_name"],
                "trial_count": trial_count,
                "loaded": loaded,
            })
        return result

    @property
    def loaded_cancer_types(self) -> list[str]:
        return list(self._pipelines.keys())

    @property
    def default_cancer_type(self) -> str:
        """Return the first enabled type as a sensible default."""
        for key in CANCER_TYPES:
            if key in self._enabled_types:
                return key
        return "breast_cancer"
