"""GCS helpers for downloading trial data and embeddings at startup.

All trial data lives on GCS — there is no local-file fallback.  The
container starts empty; this module pulls per-cancer-type CSV and
embeddings from a GCS bucket into a local scratch directory so the rest
of the pipeline can read them as ordinary files.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

from google.cloud import storage as gcs

logger = logging.getLogger(__name__)

_LOCAL_DATA_DIR: Path | None = None


def _ensure_local_dir() -> Path:
    """Return (and lazily create) a persistent scratch directory for data files."""
    global _LOCAL_DATA_DIR
    if _LOCAL_DATA_DIR is None:
        _LOCAL_DATA_DIR = Path(tempfile.mkdtemp(prefix="rag_data_"))
        logger.info("GCS scratch directory: %s", _LOCAL_DATA_DIR)
    return _LOCAL_DATA_DIR


def _ensure_cancer_dir(cancer_type: str) -> Path:
    """Return a per-cancer-type subdirectory inside the scratch dir."""
    d = _ensure_local_dir() / cancer_type
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Per-cancer-type convenience: download CSV + embeddings in one call
# ---------------------------------------------------------------------------

def download_cancer_type_data(
    bucket_name: str,
    cancer_type: str,
    csv_blob: str,
    embeddings_prefix: str,
) -> tuple[Path, Path]:
    """Download CSV + embeddings for a single cancer type.

    Returns ``(csv_path, embeddings_cache_dir)`` ready for the pipeline.
    """
    cancer_dir = _ensure_cancer_dir(cancer_type)
    csv_path = _download_csv(bucket_name, csv_blob, cancer_dir)
    cache_dir = _download_embeddings(bucket_name, embeddings_prefix, cancer_dir)
    return csv_path, cache_dir


# ---------------------------------------------------------------------------
# Low-level download helpers
# ---------------------------------------------------------------------------

def _download_csv(bucket_name: str, blob_name: str, local_dir: Path) -> Path:
    """Download the trial CSV from GCS and return its local path."""
    local_path = local_dir / "trials.csv"
    if local_path.exists():
        logger.info("CSV already present at %s — skipping download", local_path)
        return local_path

    client = gcs.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    if not blob.exists():
        logger.warning("CSV blob %s not found in gs://%s", blob_name, bucket_name)
        # Return the path anyway — the pipeline will fail gracefully later.
        return local_path
    blob.download_to_filename(str(local_path))
    size_mb = local_path.stat().st_size / 1_048_576
    logger.info("Downloaded CSV %s (%.1f MB) -> %s", blob_name, size_mb, local_path)
    return local_path


def _download_embeddings(
    bucket_name: str, prefix: str, local_dir: Path,
) -> Path:
    """Download all .npz / .json files under *prefix* and return the local cache dir."""
    cache_dir = local_dir / "embeddings_cache"
    cache_dir.mkdir(exist_ok=True)

    client = gcs.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    relevant = [b for b in blobs if b.name.endswith((".npz", ".json"))]

    if not relevant:
        logger.warning("No embedding files found under gs://%s/%s", bucket_name, prefix)
        return cache_dir

    for blob in relevant:
        filename = Path(blob.name).name
        dest = cache_dir / filename
        if dest.exists():
            logger.info("Embeddings file %s already present — skipping", filename)
            continue
        blob.download_to_filename(str(dest))
        size_mb = dest.stat().st_size / 1_048_576
        logger.info("Downloaded embeddings %s (%.1f MB)", filename, size_mb)

    logger.info(
        "Embeddings cache ready at %s (%d files)", cache_dir, len(relevant),
    )
    return cache_dir


# ---------------------------------------------------------------------------
# Upload helpers (used by incremental_update and migration scripts)
# ---------------------------------------------------------------------------

def upload_csv(bucket_name: str, blob_name: str, local_path: str | Path) -> str:
    """Upload a CSV to GCS. Returns the gs:// URI."""
    client = gcs.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(local_path))
    uri = f"gs://{bucket_name}/{blob_name}"
    logger.info("Uploaded %s -> %s", local_path, uri)
    return uri


def upload_embeddings(bucket_name: str, prefix: str, local_dir: str | Path) -> int:
    """Upload all .npz files from *local_dir* to GCS. Returns count uploaded."""
    local_dir = Path(local_dir)
    client = gcs.Client()
    bucket = client.bucket(bucket_name)
    count = 0
    for npz_file in sorted(local_dir.glob("*.npz")):
        blob_name = f"{prefix.rstrip('/')}/{npz_file.name}"
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(npz_file))
        count += 1
        logger.info("Uploaded %s -> gs://%s/%s", npz_file.name, bucket_name, blob_name)
    return count


def get_csv_metadata(bucket_name: str, blob_name: str) -> dict:
    """Return size and last-modified timestamp for the CSV blob (for health checks)."""
    try:
        client = gcs.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.reload()
        return {
            "size_bytes": blob.size,
            "updated": blob.updated.isoformat() if blob.updated else None,
        }
    except Exception as e:
        logger.warning("Failed to read GCS metadata for %s: %s", blob_name, e)
        return {}


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def download_registry(bucket_name: str, blob_name: str = "registry.json") -> dict:
    """Download and parse registry.json from GCS. Returns {} if not found."""
    try:
        client = gcs.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        if not blob.exists():
            logger.info("No registry.json found in gs://%s — using defaults", bucket_name)
            return {}
        content = blob.download_as_text()
        return json.loads(content)
    except Exception as e:
        logger.warning("Failed to download registry from GCS: %s", e)
        return {}


def upload_registry(bucket_name: str, registry: dict, blob_name: str = "registry.json") -> None:
    """Upload registry.json to GCS."""
    client = gcs.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(json.dumps(registry, indent=2), content_type="application/json")
    logger.info("Uploaded registry.json to gs://%s/%s", bucket_name, blob_name)
