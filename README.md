# NYC Taxi Month-End Revenue Forecasting System

[![CI](https://github.com/betsyhcamp/nyc_taxi/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/betsyhcamp/nyc_taxi/actions/workflows/ci.yml) ![Python](https://img.shields.io/badge/python-3.12-blue.svg)

Batch ML pipelines on Vertex AI for forecasting each Manhattan taxi zone's fare revenue for the current and next fiscal month, refreshed weekly.

______________________________________________________________________

## Business problem

A hypothetical fleet operator runs yellow cabs across Manhattan, earns a percentage of fares and closes its books at the end of each fiscal month. Every week, finance needs an updated forecast of each taxi zone's fare revenue for the current fiscal month and the next fiscal month, to plan cash flow and to see which zones are ahead of or behind plan. The same problem, a weekly refreshed forecast of revenue by segment on a fiscal calendar, shows up wherever a business plans monthly against weekly actuals.

## At a glance

- **Built:** a Training pipeline on Vertex AI Pipelines that backtests each candidate model, scores the challenger against a naive benchmark and registers it (selection rule planned) in the Vertex AI Model Registry, plus a Streamlit dashboard for reviewing backtests.
- **Models:** global gradient-boosted trees trained across all zones at once with the Naive forecast as a benchmark.
- **Evaluation:** rolling-origin backtests scored by a revenue-weighted skill score against the benchmark, broken out by revenue tier.
- **Engineering:** containerized components built on Cloud Build and pinned by image digest, configs validated by Pydantic schemas, run lineage checked across steps, CI on GitHub Actions.
- **Stack:** Python, Kubeflow Pipelines, Vertex AI, BigQuery, Cloud Storage, Cloud Build, Artifact Registry, Docker, XGBoost, LightGBM, MLForecast, StatsForecast, Pydantic, Streamlit, GitHub Actions and uv, plus [tsbricks](https://github.com/betsyhcamp/tsbricks), my open-source time-series backtesting library.

## System architecture

*Architecture diagram coming soon.*

<!-- TODO: replace the line above with this image once docs/images/architecture.drawio.svg exists.
![Architecture: Cloud Scheduler triggers a Cloud Workflow that runs the Feature, Training and Inference pipelines on Vertex AI Pipelines, connected through Google Cloud Storage and the Vertex AI Model Registry.](docs/images/architecture.drawio.svg)
-->

The system is three batch pipelines on Vertex AI Pipelines, each with its own container image. A Cloud Scheduler job triggers a Cloud Workflow that runs the three in order.

- **Feature** turns raw trip records in BigQuery into weekly revenue per zone, builds the weekly fiscal calendar and creates exogenous features.
- **Training** backtests each candidate model, scores the challenger against a naive benchmark, fits the challenger on the full history and registers it in the Vertex AI Model Registry.
- **Inference** loads that registered model and forecasts the current and next fiscal month.

The pipelines hand off through files in Google Cloud Storage, not shared state. Each pipeline run writes a small `run_output.json` at its run root naming what it produced, and the next pipeline finds its inputs from that file given only a run id.

| Component          | Status  |
| ------------------ | ------- |
| Feature pipeline   | Partial |
| Training pipeline  | Built   |
| Inference pipeline | Planned |
| Cloud Workflow     | Planned |

## Forecasting approach

Every week, the model forecasts each zone's revenue for the weeks ahead, and each month-end forecast is the completed weeks' actual revenue plus the forecast for the weeks still to come.

- **Target:** month-end fare revenue for each of Manhattan's 68 yellow-taxi zones, for the current fiscal month and, separately, for the next fiscal month.
- **Fiscal calendar:** 4-4-5, so each quarter is two four-week months followed by a five-week month. Weeks start on Sunday, and in a 53-week year the last month gets a sixth week.
- **Refresh:** weekly. The current month's forecast tightens as its weeks become actuals; the next month is forecast in full.
- **Model:** a single global GBDT model trained across all zones with an absolute-error objective. Features are lags of 1 and 52 weeks, a 4-week rolling mean and fiscal-calendar features: fiscal week of month, fiscal month, weeks in the month and workdays.
- **Benchmark:** a naive forecast that repeats the last observed week, rolled up to month-end totals the same way as the model's forecasts.
- **Framings explored:** the pipeline sums weekly forecasts into the month. Two alternatives predict at the month level, using month-to-date actuals as features: one predicts the month-end total directly (prototyped in a notebook and scored on the same leaderboard), and the other predicts only the revenue remaining in the month and adds month-to-date actuals back (in development).

## Evaluation

Each challenger is scored against the naive benchmark on the same backtest time period. Scores are weighted so large zones count more without drowning out small ones, and broken out by revenue tier so a good overall score can't hide a bad tail.

- **Backtest:** forecasts from every weekly origin across multiple evaluation periods, each covering the current and next fiscal month, scored only on months whose actuals are complete.
- **Skill score:** weighted relative MAE, `sum(w * |challenger error|) / sum(w * |benchmark error|)`, reported separately for the current and next fiscal month. Below 1 means the challenger beats the benchmark. Summing errors before dividing keeps one zone with a near-perfect benchmark from swinging the score.
- **Weights:** the cube root of each zone's trailing 26-week revenue. Bigger zones count more, but a zone with 100 times the revenue gets about 4.6 times the weight.
- **Revenue tiers:** five equal-sized tiers by mean positive weekly revenue over the trailing 52 weeks. Every metric is reported per tier as well as overall.
- **No look-ahead:** weights and tiers are recomputed at every origin from data available at that origin.
- **Also reported:** WAPE, weighted signed bias, a weighted average of per-zone skill ratios and a per-zone comparison table.
- **Planned selection rule:** a challenger qualifies only if it beats the benchmark in every revenue tier; among qualifiers, the best overall skill score wins, subject to human review.

## Engineering design decisions

Each choice below closes off a specific way a batch ML system can quietly produce a wrong answer. All are implemented in the Training pipeline; Feature and Inference will adopt them as they're built.

- **Logic separated from I/O.** Each step's logic is a pure function, dataframes in and results out, wrapped by a thin layer that reads, checks and writes files; the Kubeflow component only wraps that layer. The full test suite runs in CI with no cloud credentials.
- **Imports flow one way.** By convention, imports run from pipelines to components to step implementations to shared code to schemas and never back up, and Kubeflow is imported only by components and pipelines, so step logic never depends on the orchestrator.
- **Lineage is read, never passed.** The first step writes `run_identity.json` at the run root, and every later step reads it instead of taking run ids as parameters, so no two steps can disagree about which run or inputs they belong to.
- **Completion marker removed, then written last.** Once a step's checks pass, it deletes its manifest before touching any outputs and writes it only after all of them exist, so a failed rerun can't leave a directory that looks finished.
- **Images verified, then pinned by digest.** Cloud Build pushes an image only after import and config probes pass, and the compiler pins that image by digest and refuses to emit a template naming anything else, so what runs on Vertex is exactly what was built and probed.
- **Config versioned with code.** The config tree is baked into each image, so one git hash versions both. Project-owned configs are validated by Pydantic schemas that reject unknown keys, so a typo fails validation instead of being silently ignored.
- **Pipelines find each other's outputs by run id alone.** Training reads Feature's `run_output.json` and takes its inputs by role, so it never depends on Feature's internal step names.
- **One module builds every storage path.** Paths follow `<env>/<pipeline>/<run_id>/<step>/` and are built in one place, so the local and Vertex execution modes can't write to different locations.

## Repository layout

```
src/fcstnyctaxi/
  pipelines/          Kubeflow pipeline definitions, plus local modes that run the same steps without Vertex
  components/         Kubeflow component wrappers, one per step; they never open a file
  core/               Step implementations: a pure function plus the layer that reads, checks and writes
  lib/                Shared logic: metrics, monthly rollup, config composition, storage paths
  models/             Model callables the pipeline fits and saves
  schemas/            Pydantic schemas for configs, run identity and pipeline outputs
  image_checks/       Probes Cloud Build runs inside an image before pushing it
config/               Layered YAML configs, baked into the Docker images
dashboard/            Streamlit app for reviewing and monitoring pipeline output
notebooks/            Exploratory analysis and prototypes of the forecasting framings
experimental_models/  Model callables used only by the notebooks
queries/              BigQuery DDL and source queries
scripts/              Compile, submit and one-time setup scripts
docker/               Dockerfiles and the shared image probe
tests/                pytest suite, mirroring the source tree
docs/                 Runbook, roadmap and other documentation
```

## Running it

Walkthroughs for each pipeline, with screenshots of real runs, are in `docs/RUNBOOK.md` (to be created).

## Documentation

- `docs/RUNBOOK.md` (to be created): one-time GCP setup, then building, submitting and checking each pipeline run.
- `docs/ROADMAP.md` (to be created): planned work and open design questions.

## Data

- **Trip records:** yellow taxi trips from the [NYC Taxi and Limousine Commission (TLC) trip record data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), loaded into BigQuery. Only trips that start in Manhattan are used, and each trip's revenue is attributed to its pickup zone.
- **Taxi zones:** the TLC's taxi zone lookup, which maps each pickup location to a zone and borough.
- **Fiscal calendar:** a 4-4-5 calendar generated in this repo, with workdays that exclude New York State holidays.

## License

Released into the public domain under [the Unlicense](LICENSE).
