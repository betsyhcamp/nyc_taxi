#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="nyc-taxi-ehc"
SA_NAME="fcst-data-ingress-runner"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
BUCKET="nyc-taxi-ehc--modeling"
AR_REPO="fcst-data-ingress-pipeline"
AR_LOCATION="us-central1"
USER_EMAIL="betsy.h.camp@gmail.com"  # my own account

echo "=== Setting up IAM for ${SA_NAME} in project ${PROJECT_ID} ==="
echo

# create service account
if gcloud iam service-accounts describe "${SA_EMAIL}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "[skip] Service account ${SA_EMAIL} already exists."
else
  echo "[create] Service account ${SA_EMAIL}..."
  gcloud iam service-accounts create "${SA_NAME}" \
    --project="${PROJECT_ID}" \
    --display-name="Runner SA for fcst-data-ingress-pipeline on Vertex"
  echo "[wait] Letting SA propagate (5s)..."
  sleep 5
fi

# Project-scoped roles. roles/bigquery.dataViewer here grants read on every
# dataset in the project (including future ones) — fine for portfolio, narrow
# to dataset scope for enterprise use.
echo "[grant] Project-level roles for ${SA_EMAIL}..."
for ROLE in \
  roles/aiplatform.user \
  roles/logging.logWriter \
  roles/bigquery.jobUser \
  roles/bigquery.dataViewer
do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="${ROLE}" \
    --condition=None
done

# bucket-level write for SA to GCS
echo "[grant] Bucket ${BUCKET}: storage.objectAdmin for ${SA_EMAIL}..."
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/storage.objectAdmin" \
  --condition=None

# Artifact Registry repo-level read for SA to pull images.
echo "[grant] AR repo ${AR_REPO}: artifactregistry.reader for ${SA_EMAIL}..."
gcloud artifacts repositories add-iam-policy-binding "${AR_REPO}" \
  --location="${AR_LOCATION}" \
  --project="${PROJECT_ID}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/artifactregistry.reader" \
  --condition=None

# provide privileges for my personal account to SA
echo "[grant] User ${USER_EMAIL}: aiplatform.user (project) + iam.serviceAccountUser (on SA)..."
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="user:${USER_EMAIL}" \
  --role="roles/aiplatform.user" \
  --condition=None

gcloud iam service-accounts add-iam-policy-binding "${SA_EMAIL}" \
  --project="${PROJECT_ID}" \
  --member="user:${USER_EMAIL}" \
  --role="roles/iam.serviceAccountUser" \
  --condition=None

# verify
echo
echo "=== Verify ==="
echo
echo "--- Project-level bindings for ${SA_EMAIL} ---"

gcloud projects get-iam-policy "${PROJECT_ID}" \
  --flatten="bindings[].members" \
  --filter="bindings.members:${SA_EMAIL}" \
  --format="table(bindings.role)"

echo
echo "--- Bucket bindings for ${SA_EMAIL} on gs://${BUCKET} ---"
echo
# gcloud storage doesn't have --filter flag.
gcloud storage buckets get-iam-policy "gs://${BUCKET}" \
  --format="table(bindings.role,bindings.members)" \
  | { grep "${SA_NAME}" || echo "(no bucket bindings found for ${SA_NAME})"; }

echo
echo "--- AR repo bindings for ${SA_EMAIL} on ${AR_REPO} ---"

gcloud artifacts repositories get-iam-policy "${AR_REPO}" \
  --location="${AR_LOCATION}" --project="${PROJECT_ID}" \
  --flatten="bindings[].members" \
  --filter="bindings.members:${SA_EMAIL}" \
  --format="table(bindings.role)"

echo
echo "=== Done ==="
