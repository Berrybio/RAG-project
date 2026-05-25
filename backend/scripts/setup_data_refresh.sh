#!/usr/bin/env bash
#
# One-time setup for bi-weekly incremental data refresh on GCP.
#
# Creates:
#   1. A GCS bucket for trial data + embeddings
#   2. Uploads the initial CSV and embeddings cache
#   3. A Cloud Run Job that runs the incremental update script
#   4. A Cloud Scheduler job that triggers it bi-weekly
#
# Prerequisites:
#   - gcloud CLI authenticated with appropriate permissions
#   - The backend Docker image already built/pushed, or use --source build
#
# Usage:
#   chmod +x scripts/setup_data_refresh.sh
#   ./scripts/setup_data_refresh.sh

set -euo pipefail

# --- Configuration (edit these) ---
PROJECT="rag-project-494802"
REGION="us-central1"
BUCKET="berrybio-rag-data"
SERVICE="berrybio-backend"
JOB_NAME="berrybio-data-refresh"
SCHEDULER_NAME="biweekly-data-refresh"
# Bi-weekly: 2 AM UTC on the 1st and 15th of each month
SCHEDULE="0 2 1,15 * *"

DATA_DIR="$(cd "$(dirname "$0")/../data" && pwd)"
CSV_FILE="breast_cancer_trials_full_2026-04-14.csv"
CSV_BLOB="data/breast_cancer_trials.csv"
EMBEDDINGS_PREFIX="data/embeddings_cache"

echo "============================================"
echo "  Bi-weekly Data Refresh Setup"
echo "============================================"
echo "  Project:  $PROJECT"
echo "  Region:   $REGION"
echo "  Bucket:   $BUCKET"
echo "  Service:  $SERVICE"
echo "  Schedule: $SCHEDULE"
echo "============================================"

# 1. Create GCS bucket (idempotent)
echo ""
echo "--- Step 1: GCS Bucket ---"
if gsutil ls "gs://$BUCKET" &>/dev/null; then
    echo "Bucket gs://$BUCKET already exists."
else
    gsutil mb -p "$PROJECT" -l "$REGION" "gs://$BUCKET"
    echo "Created bucket gs://$BUCKET"
fi

# 2. Upload initial data
echo ""
echo "--- Step 2: Upload Initial Data ---"
if [ -f "$DATA_DIR/$CSV_FILE" ]; then
    echo "Uploading CSV ($CSV_FILE)..."
    gsutil -m cp "$DATA_DIR/$CSV_FILE" "gs://$BUCKET/$CSV_BLOB"
else
    echo "WARNING: $DATA_DIR/$CSV_FILE not found. Skipping CSV upload."
fi

if [ -d "$DATA_DIR/embeddings_cache" ]; then
    echo "Uploading embeddings cache..."
    gsutil -m cp -r "$DATA_DIR/embeddings_cache/"*.npz "gs://$BUCKET/$EMBEDDINGS_PREFIX/"
else
    echo "WARNING: embeddings_cache directory not found. Skipping."
fi

# 3. Create Cloud Run Job
echo ""
echo "--- Step 3: Cloud Run Job ---"
echo "Building and deploying Cloud Run Job from Dockerfile.job..."
cd "$(dirname "$0")/.."

gcloud run jobs create "$JOB_NAME" \
    --source . \
    --dockerfile Dockerfile.job \
    --project "$PROJECT" \
    --region "$REGION" \
    --task-timeout 1800 \
    --max-retries 1 \
    --set-env-vars "GCS_BUCKET=$BUCKET,GCS_CSV_BLOB=$CSV_BLOB,GCS_EMBEDDINGS_PREFIX=$EMBEDDINGS_PREFIX/,CLOUD_RUN_SERVICE=$SERVICE,GCP_REGION=$REGION" \
    --set-secrets "VOYAGE_API_KEY=VOYAGE_API_KEY:latest" \
    --memory 2Gi \
    --cpu 1 \
    2>/dev/null || \
gcloud run jobs update "$JOB_NAME" \
    --source . \
    --dockerfile Dockerfile.job \
    --project "$PROJECT" \
    --region "$REGION" \
    --task-timeout 1800 \
    --max-retries 1 \
    --set-env-vars "GCS_BUCKET=$BUCKET,GCS_CSV_BLOB=$CSV_BLOB,GCS_EMBEDDINGS_PREFIX=$EMBEDDINGS_PREFIX/,CLOUD_RUN_SERVICE=$SERVICE,GCP_REGION=$REGION" \
    --set-secrets "VOYAGE_API_KEY=VOYAGE_API_KEY:latest" \
    --memory 2Gi \
    --cpu 1

echo "Cloud Run Job '$JOB_NAME' ready."

# 4. Create Cloud Scheduler
echo ""
echo "--- Step 4: Cloud Scheduler ---"
# Get the service account for invoking the job
SA="$(gcloud iam service-accounts list --project "$PROJECT" --filter='displayName:compute' --format='value(email)' | head -1)"

gcloud scheduler jobs create http "$SCHEDULER_NAME" \
    --project "$PROJECT" \
    --location "$REGION" \
    --schedule "$SCHEDULE" \
    --time-zone "UTC" \
    --uri "https://$REGION-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/$PROJECT/jobs/$JOB_NAME:run" \
    --http-method POST \
    --oauth-service-account-email "$SA" \
    2>/dev/null || \
gcloud scheduler jobs update http "$SCHEDULER_NAME" \
    --project "$PROJECT" \
    --location "$REGION" \
    --schedule "$SCHEDULE" \
    --time-zone "UTC" \
    --uri "https://$REGION-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/$PROJECT/jobs/$JOB_NAME:run" \
    --http-method POST \
    --oauth-service-account-email "$SA"

echo "Cloud Scheduler '$SCHEDULER_NAME' set to: $SCHEDULE (UTC)"

echo ""
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  To test manually:"
echo "    gcloud run jobs execute $JOB_NAME --region $REGION"
echo ""
echo "  To update the backend to use GCS:"
echo "    gcloud run services update $SERVICE --region $REGION \\"
echo "      --update-env-vars DATA_SOURCE=gcs,GCS_BUCKET=$BUCKET"
echo "============================================"
