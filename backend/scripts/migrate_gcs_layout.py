"""One-time migration: copy existing breast cancer data to the new per-cancer-type layout.

Old layout:
    gs://berrybio-rag-data/data/breast_cancer_trials.csv
    gs://berrybio-rag-data/data/embeddings_cache/*.npz

New layout:
    gs://berrybio-rag-data/data/breast_cancer/trials.csv
    gs://berrybio-rag-data/data/breast_cancer/embeddings_cache/*.npz

Usage:
    python -m scripts.migrate_gcs_layout --bucket berrybio-rag-data
    python -m scripts.migrate_gcs_layout --bucket berrybio-rag-data --dry-run
"""
import argparse
import logging

from google.cloud import storage as gcs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("migrate_gcs_layout")

OLD_CSV = "data/breast_cancer_trials.csv"
OLD_EMBEDDINGS_PREFIX = "data/embeddings_cache/"

NEW_CSV = "data/breast_cancer/trials.csv"
NEW_EMBEDDINGS_PREFIX = "data/breast_cancer/embeddings_cache/"


def main():
    parser = argparse.ArgumentParser(description="Migrate GCS data to per-cancer-type layout")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    client = gcs.Client()
    bucket = client.bucket(args.bucket)

    # 1. Copy CSV
    old_blob = bucket.blob(OLD_CSV)
    if old_blob.exists():
        if args.dry_run:
            logger.info("[DRY RUN] Would copy %s -> %s", OLD_CSV, NEW_CSV)
        else:
            bucket.copy_blob(old_blob, bucket, NEW_CSV)
            logger.info("Copied %s -> %s", OLD_CSV, NEW_CSV)
    else:
        logger.warning("Old CSV not found at %s", OLD_CSV)

    # 2. Copy embeddings
    blobs = list(bucket.list_blobs(prefix=OLD_EMBEDDINGS_PREFIX))
    for blob in blobs:
        filename = blob.name.split("/")[-1]
        if not filename:
            continue
        new_name = f"{NEW_EMBEDDINGS_PREFIX}{filename}"
        if args.dry_run:
            logger.info("[DRY RUN] Would copy %s -> %s", blob.name, new_name)
        else:
            bucket.copy_blob(blob, bucket, new_name)
            logger.info("Copied %s -> %s", blob.name, new_name)

    logger.info("Migration complete. Old files left in place (delete manually when verified).")


if __name__ == "__main__":
    main()
