"""Bi-weekly incremental data refresh for the clinical-trial RAG pipeline.

Supports refreshing a single cancer type or all 13 registered types.
Pulls recently updated trials from the ClinicalTrials.gov API, merges
them into the existing CSV, incrementally re-embeds only changed
documents, and uploads the updated artefacts to GCS.

Designed to run as a Cloud Run Job on a bi-weekly Cloud Scheduler trigger.

Usage (local testing):
    python -m scripts.incremental_update --bucket berrybio-rag-data --dry-run
    python -m scripts.incremental_update --bucket berrybio-rag-data --cancer-type lung_cancer
    python -m scripts.incremental_update --bucket berrybio-rag-data --all

Environment variables (Cloud Run Job):
    GCS_BUCKET              — required
    VOYAGE_API_KEY          — required for re-embedding
    CLOUD_RUN_SERVICE       — if set, triggers a rolling restart after upload
    GCP_REGION              — default: us-central1
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, fields
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

# Re-use the existing puller's data model + parsing logic.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.pull_clinical_trials import (
    CTData,
    fetch_page,
    parse_study,
    DEFAULT_PAGE_SIZE,
    REQUEST_DELAY,
)
from app.core.data import load_clinical_trials, build_documents
from app.cancer_registry import CANCER_TYPES, gcs_csv_blob, gcs_embeddings_prefix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("incremental_update")

EMBED_MODEL = "voyage-3"
EMBED_BATCH_SIZE = 128
MAX_DOC_CHARS = 16_000


# ---------------------------------------------------------------------------
# 1. Fetch recently updated trials
# ---------------------------------------------------------------------------

def fetch_recent_trials(
    since: datetime,
    condition: str = "Breast Cancer",
) -> list[CTData]:
    """Pull trials whose lastUpdatePostDate >= *since*."""
    date_str = since.strftime("%Y-%m-%d")
    logger.info("Fetching trials updated since %s for condition=%r", date_str, condition)

    params = {
        "query.cond": condition,
        "filter.lastUpdatePostDate": f"{date_str},",
        "pageSize": DEFAULT_PAGE_SIZE,
        "sort": "LastUpdatePostDate",
    }
    all_trials: list[CTData] = []
    page = 1

    while True:
        logger.info("  Page %d (total so far: %d)", page, len(all_trials))
        data = fetch_page(params)
        if not data or "studies" not in data:
            break
        studies = data["studies"]
        for study in studies:
            all_trials.append(parse_study(study))
        next_token = data.get("nextPageToken")
        if not next_token:
            break
        params["pageToken"] = next_token
        page += 1
        time.sleep(REQUEST_DELAY)

    logger.info("Fetched %d recently updated trials", len(all_trials))
    return all_trials


# ---------------------------------------------------------------------------
# 2. Merge into existing CSV
# ---------------------------------------------------------------------------

def load_existing_csv(path: Path) -> dict[str, dict]:
    """Load the current CSV into a dict keyed by nctId."""
    if not path.exists():
        return {}
    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nct = row.get("nctId", "")
            if nct:
                rows[nct] = row
    logger.info("Loaded %d existing trials from %s", len(rows), path)
    return rows


def merge_trials(
    existing: dict[str, dict],
    incoming: list[CTData],
) -> tuple[dict[str, dict], int, int]:
    """Merge incoming trials into existing. Returns (merged, n_new, n_updated)."""
    n_new = 0
    n_updated = 0
    for trial in incoming:
        d = asdict(trial)
        nct = d["nctId"]
        if nct in existing:
            if d != existing[nct]:
                existing[nct] = d
                n_updated += 1
        else:
            existing[nct] = d
            n_new += 1
    logger.info("Merge: %d new, %d updated, %d total", n_new, n_updated, len(existing))
    return existing, n_new, n_updated


def save_csv(rows: dict[str, dict], path: Path) -> None:
    fieldnames = [f.name for f in fields(CTData)]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows.values():
            writer.writerow(row)
    size_mb = path.stat().st_size / 1_048_576
    logger.info("Saved %d trials to %s (%.1f MB)", len(rows), path, size_mb)


# ---------------------------------------------------------------------------
# 3. Incremental embedding
# ---------------------------------------------------------------------------

def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def _build_doc_texts(csv_path: Path) -> tuple[list[str], list[str]]:
    """Use the real pipeline's build_documents to produce (nct_ids, texts).

    This guarantees the embeddings hash matches what the retriever computes
    at startup, so the cache file is usable without re-embedding.
    """
    df = load_clinical_trials(str(csv_path))
    docs = build_documents(df)
    nct_ids = [d["doc_id"] for d in docs]
    texts = [d["text"][:MAX_DOC_CHARS] for d in docs]
    return nct_ids, texts


def incremental_embed(
    csv_path: Path,
    cache_dir: Path,
    voyage_api_key: str,
    dry_run: bool = False,
) -> Path:
    """Build or update the embeddings cache."""
    import voyageai

    sidecar_path = cache_dir / "text_hashes.json"
    old_hashes: dict[str, str] = {}
    if sidecar_path.exists():
        old_hashes = json.loads(sidecar_path.read_text())

    nct_ids, texts = _build_doc_texts(csv_path)
    new_hashes = {nct: _text_hash(t) for nct, t in zip(nct_ids, texts)}

    changed_indices = [
        i for i, nct in enumerate(nct_ids)
        if new_hashes[nct] != old_hashes.get(nct)
    ]
    logger.info(
        "Embedding delta: %d changed out of %d total",
        len(changed_indices), len(nct_ids),
    )

    # Load existing per-nct embedding store if present
    per_nct_path = cache_dir / "per_nct_embeddings.npz"
    existing_embeddings: dict[str, np.ndarray] = {}
    if per_nct_path.exists():
        with np.load(per_nct_path) as data:
            for key in data.files:
                existing_embeddings[key] = data[key]

    if changed_indices and not dry_run:
        client = voyageai.Client(api_key=voyage_api_key)
        changed_texts = [texts[i] for i in changed_indices]
        changed_ncts = [nct_ids[i] for i in changed_indices]

        logger.info("Re-embedding %d documents...", len(changed_texts))
        for batch_start in range(0, len(changed_texts), EMBED_BATCH_SIZE):
            batch_end = min(batch_start + EMBED_BATCH_SIZE, len(changed_texts))
            batch_texts = changed_texts[batch_start:batch_end]
            batch_ncts = changed_ncts[batch_start:batch_end]
            result = client.embed(batch_texts, model=EMBED_MODEL, input_type="document")
            for nct, emb in zip(batch_ncts, result.embeddings):
                existing_embeddings[nct] = np.array(emb, dtype=np.float32)
            logger.info("  Embedded %d / %d", batch_end, len(changed_texts))

    # Rebuild the ordered matrix the retriever expects
    all_embeddings = []
    dim = None
    for nct in nct_ids:
        if nct in existing_embeddings:
            vec = existing_embeddings[nct]
            dim = vec.shape[0]
            all_embeddings.append(vec)
        else:
            if dim is None:
                for v in existing_embeddings.values():
                    dim = v.shape[0]
                    break
            if dim:
                all_embeddings.append(np.zeros(dim, dtype=np.float32))

    if all_embeddings:
        matrix = np.stack(all_embeddings)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms

        cache_dir.mkdir(parents=True, exist_ok=True)
        if not dry_run:
            np.savez_compressed(per_nct_path, **existing_embeddings)

        # Save the monolithic cache the retriever loads
        corpus_hasher = hashlib.sha256()
        corpus_hasher.update(EMBED_MODEL.encode("utf-8"))
        corpus_hasher.update(str(len(texts)).encode("utf-8"))
        for t in texts:
            corpus_hasher.update(t.encode("utf-8", errors="replace"))
            corpus_hasher.update(b"\x00")
        digest = corpus_hasher.hexdigest()[:16]
        cache_filename = f"voyage_{EMBED_MODEL}_{len(texts)}_{digest}.npz"
        cache_path = cache_dir / cache_filename

        if not dry_run:
            np.savez_compressed(cache_path, embeddings=matrix)
            logger.info("Saved retriever cache: %s (shape=%s)", cache_path.name, matrix.shape)

        # Clean up old cache files
        for old_file in cache_dir.glob("voyage_*.npz"):
            if old_file.name != cache_filename and old_file.name != "per_nct_embeddings.npz":
                old_file.unlink()
                logger.info("Removed stale cache: %s", old_file.name)

    # Save sidecar
    if not dry_run:
        sidecar_path.write_text(json.dumps(new_hashes))

    return cache_dir


# ---------------------------------------------------------------------------
# 4. GCS upload (per cancer type)
# ---------------------------------------------------------------------------

def upload_to_gcs(
    csv_path: Path,
    cache_dir: Path,
    bucket_name: str,
    csv_blob: str,
    embeddings_prefix: str,
) -> None:
    from google.cloud import storage as gcs

    client = gcs.Client()
    bucket = client.bucket(bucket_name)

    # Upload CSV
    blob = bucket.blob(csv_blob)
    blob.upload_from_filename(str(csv_path))
    logger.info("Uploaded CSV -> gs://%s/%s", bucket_name, csv_blob)

    # Upload embeddings
    for f in sorted(cache_dir.glob("*.npz")):
        blob_name = f"{embeddings_prefix.rstrip('/')}/{f.name}"
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(f))
        logger.info("Uploaded %s -> gs://%s/%s", f.name, bucket_name, blob_name)

    # Upload sidecar
    sidecar = cache_dir / "text_hashes.json"
    if sidecar.exists():
        blob_name = f"{embeddings_prefix.rstrip('/')}/text_hashes.json"
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(sidecar))
        logger.info("Uploaded text_hashes.json")


# ---------------------------------------------------------------------------
# 5. Rolling restart trigger
# ---------------------------------------------------------------------------

def trigger_restart(service: str, region: str) -> None:
    """Force a new Cloud Run revision so the service picks up fresh data."""
    logger.info("Triggering rolling restart for %s in %s", service, region)
    try:
        subprocess.run(
            [
                "gcloud", "run", "services", "update", service,
                "--region", region,
                "--update-env-vars", f"DATA_REFRESHED_AT={datetime.now(timezone.utc).isoformat()}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("Rolling restart triggered successfully")
    except subprocess.CalledProcessError as e:
        logger.error("Rolling restart failed: %s\n%s", e.returncode, e.stderr)
        raise


# ---------------------------------------------------------------------------
# 6. Process a single cancer type end-to-end
# ---------------------------------------------------------------------------

def process_cancer_type(
    cancer_type: str,
    bucket: str,
    voyage_api_key: str,
    lookback_days: int,
    dry_run: bool,
) -> dict:
    """Run the full incremental update for one cancer type.

    Returns a summary dict: {cancer_type, n_new, n_updated, total, skipped}.
    """
    ct_info = CANCER_TYPES[cancer_type]
    condition = ct_info["condition_query"]
    csv_blob = gcs_csv_blob(cancer_type)
    emb_prefix = gcs_embeddings_prefix(cancer_type)

    logger.info("=" * 60)
    logger.info("Processing %s (condition=%r)", cancer_type, condition)
    logger.info("=" * 60)

    work_dir = Path(tempfile.mkdtemp(prefix=f"rag_update_{cancer_type}_"))
    csv_path = work_dir / "trials.csv"

    # Step 1: Download existing CSV from GCS
    if not dry_run:
        try:
            from google.cloud import storage as gcs_lib
            client = gcs_lib.Client()
            bkt = client.bucket(bucket)
            blob = bkt.blob(csv_blob)
            if blob.exists():
                blob.download_to_filename(str(csv_path))
                logger.info("Downloaded existing CSV from GCS for %s", cancer_type)
            else:
                logger.info("No existing CSV in GCS for %s — starting fresh", cancer_type)
        except Exception:
            logger.exception("Failed to download CSV from GCS for %s", cancer_type)

    # Step 2: Fetch recently updated trials
    since = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    incoming = fetch_recent_trials(since, condition=condition)

    if not incoming:
        logger.info("No updated trials found for %s — skipping", cancer_type)
        return {"cancer_type": cancer_type, "n_new": 0, "n_updated": 0, "total": 0, "skipped": True}

    # Step 3: Merge
    existing = load_existing_csv(csv_path)
    merged, n_new, n_updated = merge_trials(existing, incoming)

    if n_new == 0 and n_updated == 0:
        logger.info("No changes after merge for %s — skipping upload", cancer_type)
        return {"cancer_type": cancer_type, "n_new": 0, "n_updated": 0, "total": len(merged), "skipped": True}

    save_csv(merged, csv_path)

    # Step 4: Incremental embedding
    cache_dir = work_dir / "embeddings_cache"
    cache_dir.mkdir(exist_ok=True)

    if not dry_run:
        try:
            from google.cloud import storage as gcs_lib
            client = gcs_lib.Client()
            bkt = client.bucket(bucket)
            for name in ["per_nct_embeddings.npz", "text_hashes.json"]:
                blob = bkt.blob(f"{emb_prefix.rstrip('/')}/{name}")
                if blob.exists():
                    blob.download_to_filename(str(cache_dir / name))
                    logger.info("Downloaded %s from GCS for %s", name, cancer_type)
        except Exception:
            logger.exception("Failed to download embeddings sidecars from GCS for %s", cancer_type)

    if voyage_api_key:
        incremental_embed(csv_path, cache_dir, voyage_api_key, dry_run=dry_run)
    else:
        logger.warning("No VOYAGE_API_KEY — skipping embedding step for %s", cancer_type)

    # Step 5: Upload to GCS
    if not dry_run:
        upload_to_gcs(csv_path, cache_dir, bucket, csv_blob, emb_prefix)
    else:
        logger.info("[DRY RUN] Would upload CSV + embeddings for %s", cancer_type)

    return {
        "cancer_type": cancer_type,
        "n_new": n_new,
        "n_updated": n_updated,
        "total": len(merged),
        "skipped": False,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Incremental clinical-trial data refresh")
    parser.add_argument("--bucket", default=os.environ.get("GCS_BUCKET", ""))
    parser.add_argument(
        "--cancer-type", default=None,
        help="Single cancer type to refresh (e.g. lung_cancer)"
    )
    parser.add_argument(
        "--all", action="store_true", dest="refresh_all",
        help="Refresh all 13 cancer types"
    )
    parser.add_argument("--voyage-api-key", default=os.environ.get("VOYAGE_API_KEY", ""))
    parser.add_argument("--lookback-days", type=int, default=15)
    parser.add_argument("--service", default=os.environ.get("CLOUD_RUN_SERVICE", ""))
    parser.add_argument("--region", default=os.environ.get("GCP_REGION", "us-central1"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.bucket:
        parser.error("--bucket or GCS_BUCKET is required")

    # Determine which cancer types to process
    if args.refresh_all:
        cancer_types = list(CANCER_TYPES.keys())
    elif args.cancer_type:
        if args.cancer_type not in CANCER_TYPES:
            valid = ", ".join(sorted(CANCER_TYPES))
            parser.error(f"Unknown cancer type {args.cancer_type!r}. Valid: {valid}")
        cancer_types = [args.cancer_type]
    else:
        # Default: all types (when running as Cloud Run Job)
        cancer_types = list(CANCER_TYPES.keys())

    logger.info("Refreshing %d cancer type(s): %s", len(cancer_types), ", ".join(cancer_types))

    results = []
    for ct in cancer_types:
        result = process_cancer_type(
            cancer_type=ct,
            bucket=args.bucket,
            voyage_api_key=args.voyage_api_key,
            lookback_days=args.lookback_days,
            dry_run=args.dry_run,
        )
        results.append(result)

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("INCREMENTAL UPDATE SUMMARY")
    logger.info("=" * 60)
    any_changes = False
    for r in results:
        status = "skipped" if r["skipped"] else f"+{r['n_new']} new, ~{r['n_updated']} updated"
        logger.info("  %-25s %s (total: %d)", r["cancer_type"], status, r["total"])
        if not r["skipped"]:
            any_changes = True

    # Trigger rolling restart once (not per type)
    if any_changes and args.service and not args.dry_run:
        trigger_restart(args.service, args.region)
    elif args.service and any_changes:
        logger.info("[DRY RUN] Would restart service %s", args.service)
    elif not any_changes:
        logger.info("No changes across any cancer type — no restart needed")


if __name__ == "__main__":
    main()
