"""GCS helpers for downloading trial data and embeddings at startup.

When ``data_source=gcs`` the container starts with no baked-in CSV or
embeddings cache.  This module pulls them from a GCS bucket into a
local scratch directory so the rest of the pipeline can read them as
ordinary files.
"""
from __future__ import annotations

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


def download_csv(bucket_name: str, blob_name: str) -> Path:
    """Download the trial CSV from GCS and return its local path."""
    local_dir = _ensure_local_dir()
    local_path = local_dir / Path(blob_name).name
    if local_path.exists():
        logger.info("CSV already present at %s — skipping download", local_path)
        return local_path

    client = gcs.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(str(local_path))
    size_mb = local_path.stat().st_size / 1_048_576
    logger.info("Downloaded CSV %s (%.1f MB) → %s", blob_name, size_mb, local_path)
    return local_path


def download_embeddings(bucket_name: str, prefix: str) -> Path:
    """Download all .npz files under *prefix* and return the local cache dir."""
    local_dir = _ensure_local_dir() / "embeddings_cache"
    local_dir.mkdir(exist_ok=True)

    client = gcs.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    npz_blobs = [b for b in blobs if b.name.endswith(".npz")]

    if not npz_blobs:
        logger.warning("No .npz files found under gs://%s/%s", bucket_name, prefix)
        return local_dir

    for blob in npz_blobs:
        filename = Path(blob.name).name
        dest = local_dir / filename
        if dest.exists():
            logger.info("Embeddings file %s already present — skipping", filename)
            continue
        blob.download_to_filename(str(dest))
        size_mb = dest.stat().st_size / 1_048_576
        logger.info("Downloaded embeddings %s (%.1f MB)", filename, size_mb)

    logger.info("Embeddings cache ready at %s (%d files)", local_dir, len(npz_blobs))
    return local_dir


def upload_csv(bucket_name: str, blob_name: str, local_path: str | Path) -> str:
    """Upload a CSV to GCS. Returns the gs:// URI."""
    client = gcs.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(local_path))
    uri = f"gs://{bucket_name}/{blob_name}"
    logger.info("Uploaded %s → %s", local_path, uri)
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
        logger.info("Uploaded %s → gs://%s/%s", npz_file.name, bucket_name, blob_name)
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
