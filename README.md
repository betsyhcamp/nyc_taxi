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
gcloud artifacts repositories create fcst-data-ingress-pipeline \
  --repository-format=docker \
  --location=us-central1 \
  --description="Container image for F (data ingress) pipeline of NYC taxi forecasting system"
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

The image is tagged `us-central1-docker.pkg.dev/<project>/fcst-data-ingress-pipeline/extract-db-to-bucket:<short-sha>`.

**Step 2 — Push the image to Artifact Registry:**

```{bash}
task push-image
```

Confirm the push landed:

```{bash}
gcloud artifacts docker tags list us-central1-docker.pkg.dev/nyc-taxi-ehc/fcst-data-ingress-pipeline/extract-db-to-bucket
```

The newly pushed tag should appear in the list.

**Step 3 — Update config and wrapper to reference the new image:**

If the image change is meant to become the new default (the SHA in config drives both `verify_kfp_local.py` and future Cycle 4 submission scripts), bump:

- `config/configs_zone_demand_pipeline.yaml` — set `docker.extract_db_to_bucket` to the new tag.
- `src/fcstnyctaxi/components/feature/extract_db_to_bucket_component.py` — update the fallback string in `os.environ.get("FCST_EXTRACT_IMAGE", ...)`.

Commit these changes.

**Step 4 — Verify end-to-end via `kfp.local.DockerRunner`:**

```{bash}
uv run python scripts/verify_kfp_local.py
```

Reads the config, sets the `FCST_EXTRACT_IMAGE` env var, imports the wrapper, and invokes the component via `kfp.local.DockerRunner`. The container hits real BigQuery and writes both a Parquet snapshot and a SQL sidecar to GCS under a run-scoped prefix.

Expected output (abridged):

```
Found image 'us-central1-docker.pkg.dev/nyc-taxi-ehc/fcst-data-ingress-pipeline/extract-db-to-bucket:<sha>'

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

`<run_id>` is a sortable UTC microsecond timestamp (e.g. `20260519t053021093666z`).

**Step 5 — Confirm GCS artifacts:**

```{bash}
gsutil ls gs://nyc-taxi-ehc--modeling/dev/initial_datapull/<run_id>/
```

Should list two objects:

- `manhattan_daily_zone_pickups.parquet` — the source snapshot (size matches `file_size_bytes` in metadata).
- `query.sql` — the rendered SQL text used to produce the snapshot.

### Building, compiling, and submitting the Training pipeline

End-to-end workflow for the Training pipeline: build the image, push it, compile a template against the pushed digest, then submit that template to Vertex AI Pipelines. The DAG composes every config for the run, backtests each model in `model_roles`, one task per model fixed at compile time, scores the challenger against the benchmark, fits the challenger on the whole panel, then registers it as a version in the Vertex AI Model Registry.

Compiling and submitting are separate programs on purpose. The image reference is a compile-time input, since `@dsl.component` binds `base_image` when the module is imported, so a compiled template already names every image it will run. The submitter therefore takes a template and never sees a tag, a digest, or `FCST_TRAIN_IMAGE`. That is also what lets a production trigger, which submits a published template with no compiler anywhere, use the same shape.

**Prerequisites:**

- ADC configured: `gcloud auth application-default login`
- `bash scripts/setup_train_iam.sh` run once per project. It creates the `fcst-ml-containers` repository, which Artifact Registry does not auto-create on push, and the Training runner service account.
- `.env` filled in from `.env.example`, with `FCST_TRAIN_SERVICE_ACCOUNT` set. The submitter reads it with `python-dotenv`, so there is no sourcing step.
- Working tree clean, so image tags do not include `-dirty`
- A published Feature run. `uv run python scripts/publish_feature_stand_in.py --env dev` prints the `--feature-run-id` the submit step needs, with `--panel-uri` and `--calendar-uri` commented below it as optional overrides.

**Step 1. Build and verify the image:**

```{bash}
task build-verify-train-image
```

Builds the `linux/amd64` image and probes it twice: every environment in the baked config tree composes, and that tree matches `config/` file for file. Expected last lines:

```
image OK: ['dev'] compose
config tree OK: 14 files
```

**Step 2. Push the image:**

```{bash}
task push-train-image
```

Pushes the tag step 1 built, then prints the digest as a ready-to-run command:

```
  pushed tag : us-central1-docker.pkg.dev/nyc-taxi-ehc/fcst-ml-containers/train:<short-sha>
  digest     : us-central1-docker.pkg.dev/nyc-taxi-ehc/fcst-ml-containers/train@sha256:<64 hex>

  next: task compile-train IMAGE_REF=<the digest above>
```

This task deliberately does not build. `GIT_HASH` is recomputed on every invocation, so a push-time build could ship bytes that no probe ever saw.

**Step 3. Compile and submit:**

```{bash}
task compile-submit-train IMAGE_REF=<digest from step 2> -- \
  --env dev \
  --feature-run-id <id from publish_feature_stand_in.py>
```

Both artifact URIs are resolved from `--feature-run-id`, by reading the `run_outputs.json` the Feature run wrote at its run root. Pass `--panel-uri` and `--calendar-uri` together to override that; neither is accepted alone.

`compile-submit-train` compiles a fresh template into `build/fcst-train-pipeline.yaml`, then appends its own `--template-path` after your arguments. Argparse is last-wins, so the run always submits what it just compiled.

The compile step refuses to leave a template pinning anything other than the digest you named, and deletes the rejected file rather than leaving something submittable on disk.

To submit a template that already exists, skip the compile:

```{bash}
task submit-train -- --template-path build/fcst-train-pipeline.yaml --env dev ...
```

`task submit-train -- --help` lists every flag, and works on a machine with no `.env`. Its five domain flags are the same ones the local execution mode takes, so the two modes differ only in orchestration:

```{bash}
uv run python -m fcstnyctaxi.pipelines.local_train_pipeline --help
```

The local mode registers only when given `--serving-image`, a digest-pinned image recorded as the runtime that reads the bundle. Without it, a local run stops after the fit and writes no `run_outputs.json`, so routine local runs never add versions to the shared registry.

`--run-id` is optional and generated when absent. Caching is on by default, so resubmitting under the same `--run-id` reuses completed tasks; pass `--no-caching` to re-execute everything. The submitter is fire and forget unless given `--wait`, and it writes the id it submitted to `<tmpdir>/fcstnyctaxi/.last_run_id`.

**Step 4. Confirm the run and its artifacts:**

The submitter logs the console URL for the run. The artifacts land under the run prefix it also logs:

```{bash}
gsutil ls -r gs://nyc-taxi-ehc--modeling/dev/train/<run_id>/
```

`run_identity.json` sits at the run root rather than inside a step directory, because its reader is outside the pipeline and can construct only `<bucket>/<env>/train/<run_id>`. `compose_configs/` holds the five configs plus `manifest.json`, and each model named in `config/train/modeling.yaml`'s `model_roles` gets its own eight-file sidecar under `backtest/<model_name>/`, ending in `backtest_manifest.json`. The Vertex UI names those tasks `backtest-<model_name>`; the compiled template's task keys are positional. `evaluate/` holds the run's four score tables and `evaluate_manifest.json`, scored from the two sidecars `model_roles` names. `final_fit/<model_name>/` holds the challenger's bundle, fitted on the whole panel: `model/`, the `composed_config.yaml` it was fitted under, and `final_fit_manifest.json`. That task runs after `evaluate`, and the Vertex UI names it `final_fit-<model_name>`. The last task, `register_model-<model_name>`, uploads that bundle as a version of the `<model_id_prefix>-<model_name>` model and writes `run_outputs.json` at the run root. Written last, it marks the run finished, and its `published.model_tag` names the exact version, `projects/<number>/locations/<location>/models/<model_id>@<version>`, which is what Inference loads. Resubmitting a finished `--run-id` on the same commit registers nothing; on a different commit the register task fails, naming both commits.

`task build-clean` removes compiled templates; `task scratch-clean` removes the local scratch mirror, `.last_run_id` included.
