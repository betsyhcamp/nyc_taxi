# Local ML Pipeline

The goal of this `local/` directory is to eliminate any need of infrastructure and get the whole project running locally.

I (Eric) found myself confounded by two simultaneous learning curves: sagemaker and timeseries forecasting in general.

Running everything locally, and without an orchestrating framework like Airflow, Metaflow, SageMaker, etc. will make the infra part much more straightforward.

## Overview

1. Download taxi ride data as parquet files idempotently

    ```bash
    just download-taxi-data \
        --year-month-start=2025-01 \
        --year-month-end-inclusive=2025-12
    ```

    - skips already downloaded files
    - defaults to last 3 months, acknowledging that the NYC TLC publishes data with a 2-month delay

2. Load into duckdb use SQL to get to gold layer

    ```bash
    just transform-taxi-data \
        --year-month-start=2025-01 \
        --year-month-end-inclusive=2025-12
    ```

    - runs SQL queries parameterized by date against data in these files, uses merge into for idempotency
    - defaults to most recent 3 months, currently in the lakehouse.duckdb

3. Train model

    ```bash
    just train-model \
        --run-id=1 \
        --holdout-year-month-start=2025-10 \
        --train-year-month-start=2025-01
    ```

    - trains model on the training data
    - cross validation goes here

4. Evaluate model

    ```bash
    just evaluate-model \
        --run-id=1 \
        --holdout-year-month-start=2025-10 \
        --holdout-year-month-end=2025-11
    ```

    - loads model from the run ID
    - evaluates model on the holdout data IN AN ALGORITHM AGNOSTIC WAY
      - case 1: model is recursive
      - case 2: model is not recursive
    - creates visualizations
      - prediction intervals
      - actual vs. forecast
      - bias and variation
    - logs RMSE

    [Reddit thread](https://www.reddit.com/r/learnmachinelearning/comments/xs2ish/why_on_earth_do_most_online_examples_show_cross/) on when to use cross-validation vs use a holdout set.

5. Deploy model

  > Question: Which deployment paradigm should we use?

  Options:
  1. train once, e.g. every 7 days, and forecast daily
  2. retrain daily and immediately forecast

  One preference: never transiiton a model from "non-prod" to "prod". Always re-train the model in prod once the code is merged.