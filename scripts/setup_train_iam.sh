#!/usr/bin/env bash
set -euo pipefail

# IAM for the Training pipeline, following scripts/setup_ingress_iam.sh.
#
# Two differences from the ingress script, both deliberate:
#
#   1. No BigQuery roles. Training never touches BigQuery — only Feature does
#      (see config/environments/dev.yaml's source_data comment). Reusing
#      fcst-data-ingress-runner would hand Training bigquery.dataViewer on every
#      dataset in the project for no reason, which is why this is a separate SA
#      even though .env.example allows one SA for all three slices.
#   2. It creates the Artifact Registry repository. Artifact Registry does not
#      auto-create on push, so `task push-train-image` fails without this.
#
# Run once, before `task push-train-image` and before `task submit-train`.
# Every step is idempotent, so re-running is safe.

PROJECT_ID="nyc-taxi-ehc"
SA_NAME="svc-revfcst-train-runner"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
BUCKET="nyc-taxi-ehc--modeling"
AR_REPO="fcst-ml-containers"
AR_LOCATION="us-central1"
USER_EMAIL="betsy.h.camp@gmail.com"  # my own account
# find PROJECT_NUMBER via: gcloud projects describe ${PROJECT_ID} --format="value(projectNumber)"
PROJECT_NUMBER="1083454808980"
VERTEX_SA_AGENT="service-${PROJECT_NUMBER}@gcp-sa-aiplatform-cc.iam.gserviceaccount.com"

echo "=== Setting up IAM for ${SA_NAME} in project ${PROJECT_ID} ==="
echo

# ---------------------------------------------------------------------------
# Artifact Registry repository. One Docker repository holds all three slices'
# images as flat names (feature, train, inference), per
# config/environments/dev.yaml's images block.
# ---------------------------------------------------------------------------
if gcloud artifacts repositories describe "${AR_REPO}" \
     --location="${AR_LOCATION}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "[skip] AR repository ${AR_REPO} already exists."
else
  echo "[create] AR repository ${AR_REPO} (docker, ${AR_LOCATION})..."
  gcloud artifacts repositories create "${AR_REPO}" \
    --repository-format=docker \
    --location="${AR_LOCATION}" \
    --project="${PROJECT_ID}" \
    --description="Container images for the feature, train and inference slices"
fi

# ---------------------------------------------------------------------------
# Service account
# ---------------------------------------------------------------------------
if gcloud iam service-accounts describe "${SA_EMAIL}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "[skip] Service account ${SA_EMAIL} already exists."
else
  echo "[create] Service account ${SA_EMAIL}..."
  gcloud iam service-accounts create "${SA_NAME}" \
    --project="${PROJECT_ID}" \
    --display-name="Revenue Training Pipeline Runner" \
    --description="Service account for running the Vertex AI training pipeline"
  echo "[wait] Letting SA propagate (5s)..."
  sleep 5
fi

# ---------------------------------------------------------------------------
# Project-scoped roles. aiplatform.user lets the job run as this SA;
# logging.logWriter is what makes component stdout reach Cloud Logging.
# ---------------------------------------------------------------------------
echo "[grant] Project-level roles for ${SA_EMAIL}..."
for ROLE in \
  roles/aiplatform.user \
  roles/logging.logWriter
do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="${ROLE}" \
    --condition=None
done

# ---------------------------------------------------------------------------
# Bucket-level write. compose_configs writes its seven artifacts through the
# gcsfuse mount and reads the Feature panel and calendar from the same bucket,
# and KFP writes execution metadata under vertex.pipeline_root — which dev.yaml
# also places in this bucket. Bucket-wide objectAdmin is fine for a portfolio
# project; narrow to a prefix condition for enterprise use.
# ---------------------------------------------------------------------------
echo "[grant] Bucket ${BUCKET}: storage.objectAdmin for ${SA_EMAIL}..."
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/storage.objectAdmin" \
  --condition=None

# ---------------------------------------------------------------------------
# Artifact Registry repo-level read, for two DIFFERENT identities.
#
# Bindings on an AR repository are repo-scoped, so a new repository starts with
# none — the grants on fcst-data-ingress-pipeline do not carry over.
#
# Vertex AI pulls each component's container image using a Google-managed
# service agent, which is NOT the runner SA. With a custom AR repository the
# agent needs explicit reader access or the pipeline fails before any component
# code runs, with "Vertex AI Service Agent ... does not have permission to
# access Artifact Registry repository".
# ---------------------------------------------------------------------------
for MEMBER in \
  "serviceAccount:${SA_EMAIL}" \
  "serviceAccount:${VERTEX_SA_AGENT}"
do
  echo "[grant] AR repo ${AR_REPO}: artifactregistry.reader for ${MEMBER}..."
  gcloud artifacts repositories add-iam-policy-binding "${AR_REPO}" \
    --location="${AR_LOCATION}" \
    --project="${PROJECT_ID}" \
    --member="${MEMBER}" \
    --role="roles/artifactregistry.reader" \
    --condition=None
done

# ---------------------------------------------------------------------------
# The human submitter. iam.serviceAccountUser on the SA is what lets
# job.submit(service_account=...) run the pipeline as the runner SA.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
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
# gcloud storage doesn't have a --filter flag.
gcloud storage buckets get-iam-policy "gs://${BUCKET}" \
  --format="table(bindings.role,bindings.members)" \
  | { grep "${SA_NAME}" || echo "(no bucket bindings found for ${SA_NAME})"; }

echo
echo "--- AR repo bindings on ${AR_REPO} (expect reader for the runner SA and the Vertex agent) ---"
gcloud artifacts repositories get-iam-policy "${AR_REPO}" \
  --location="${AR_LOCATION}" --project="${PROJECT_ID}" \
  --flatten="bindings[].members" \
  --format="table(bindings.role,bindings.members)"

echo
echo "=== Done ==="
echo
echo "Add to .env:"
echo "  FCST_TRAIN_SERVICE_ACCOUNT=${SA_EMAIL}"
