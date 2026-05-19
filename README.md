# Exploring and modeling NYC Yellow Taxi Demand

[![CI](https://github.com/betsyhcamp/nyc_taxi/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/betsyhcamp/nyc_taxi/actions/workflows/ci.yml) ![Python](https://img.shields.io/badge/python-3.12-blue.svg)

______________________________________________________________________

## Purpose

Taxi demand needs to be forecasted to optimize resource allocation, improve operational efficiency, and to enhance customer experience (e.g., reduced rider wait times).

- `TODO`: Need to do exploratory data analysis and
- `TODO`: define forecasting problem along typical axes: quantity to forecast, granularity in time of forecast (hourly, daily, weekly), forecast horizon (how many hours, days, weeks ahead need to be forecasted), geographic granularity (e.g. taxi zone, boro of NYC)

## Data

Base data for NY Yellow Taxi ride data, data dictionaries, and metadata files associated with rides:
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

Weather data can be found here:
https://www.ncei.noaa.gov/pub/data/daily-grids/v1-0-0/averages/2025/

## Setup

### Artifact Registry

For each Vertex AI component (ie, step) with a Docker image, the image is placed in a project specific GCP Artifact Registry

Prerequisite for use of the commands below is a working installation of `gcloud` with application default credentials (ADC) for GCP.

**Create the Artifact Registry (one time per project):**

```{bash}
gcloud artifacts repositories create forecasting-pipeline \
  --repository-format=docker \
  --location=us-central1 \
  --description="Container images for the NYC taxi forecasting pipeline"
```

**Authorize developer machine (one time per developer machine):**

```{bash}
gcloud auth configure-docker us-central1-docker.pkg.dev
```

## Usage

### Building and running the `extract_db_to_bucket` component via Docker

End-to-end workflow for building the component's Docker image, pushing it to Artifact Registry, and verifying it runs against real BigQuery and GCS via `kfp.local.DockerRunner` — the closest local analog to a Vertex AI Pipelines run.

**Prerequisites:**

- ADC configured: `gcloud auth application-default login`
- Artifact Registry setup complete (see Setup section above)
- Working tree clean (so image tags don't include `-dirty`)

**Step 1 — Build and verify the image:**

```{bash}
task build-verify-image
```

Builds the multi-stage `linux/amd64` image and runs an image-health probe that imports `fcstnyctaxi`, `kfp`, and `google.cloud.bigquery`. Expected last line of output:

```
image OK
```

The image is tagged `us-central1-docker.pkg.dev/<project>/forecasting-pipeline/extract-db-to-bucket:<short-sha>`.

**Step 2 — Push the image to Artifact Registry:**

```{bash}
task push-image
```

Confirm the push landed:

```{bash}
gcloud artifacts docker tags list us-central1-docker.pkg.dev/nyc-taxi-ehc/forecasting-pipeline/extract-db-to-bucket
```

The newly pushed tag should appear in the list.

**Step 3 — Update config and wrapper to reference the new image:**

If the image change is meant to become the new default (the SHA in config drives both `verify_kfp_local.py` and future Cycle 4 submission scripts), bump:

- `config/configs_zone_demand_pipeline.yaml` — set `docker.extract_db_to_bucket` to the new tag.
- `src/fcstnyctaxi/components/feature/extract_db_to_bucket_component.py` — update the fallback string in `os.environ.get("FCSTNYCTAXI_EXTRACT_IMAGE", ...)`.

Commit these changes.

**Step 4 — Verify end-to-end via `kfp.local.DockerRunner`:**

```{bash}
uv run python scripts/verify_kfp_local.py
```

Reads the config, sets the `FCSTNYCTAXI_EXTRACT_IMAGE` env var, imports the wrapper, and invokes the component via `kfp.local.DockerRunner`. The container hits real BigQuery and writes both a Parquet snapshot and a SQL sidecar to GCS under a run-scoped prefix.

Expected output (abridged):

```
Found image 'us-central1-docker.pkg.dev/nyc-taxi-ehc/forecasting-pipeline/extract-db-to-bucket:<sha>'

[KFP Executor ...]: Looking for component `extract_db_to_bucket` in ...
...
Task 'extract-db-to-bucket' finished with status SUCCESS

Component completed.
    uri: gs://nyc-taxi-ehc--modeling/dev/initial_datapull/<run_id>/manhattan_daily_zone_pickups.parquet
    metadata: {
        'gcs_uri': 'gs://nyc-taxi-ehc--modeling/dev/initial_datapull/<run_id>/manhattan_daily_zone_pickups.parquet',
        'sql_sha256': '3434b01110b4940eeab5ffa847e7a2576b5b2f3639c737b0109e0c5097f11555',
        'sql_sidecar_gcs_uri': 'gs://nyc-taxi-ehc--modeling/dev/initial_datapull/<run_id>/query.sql',
        'extracted_at': '<utc-iso-timestamp>',
        'row_count': 194392.0,
        'git_sha': None,
        'bq_query_job_id': '<bq-job-uuid>',
        'file_size_bytes': 367770.0,
        'run_id': '<run_id>',
    }
```

`<run_id>` is a sortable UTC microsecond timestamp (e.g. `20260519T053021093666Z`).

**Step 5 — Confirm GCS artifacts:**

```{bash}
gsutil ls gs://nyc-taxi-ehc--modeling/dev/initial_datapull/<run_id>/
```

Should list two objects:

- `manhattan_daily_zone_pickups.parquet` — the source snapshot (size matches `file_size_bytes` in metadata).
- `query.sql` — the rendered SQL text used to produce the snapshot.
