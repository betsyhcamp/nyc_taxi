# Local ML Pipeline

The goal of this `local/` directory is to eliminate any need of infrastructure and get the whole project running locally.

I (Eric) found myself confounded by two simultaneous learning curves: sagemaker and timeseries forecasting in general.

Running everything locally, and without an orchestrating framework like Airflow, Metaflow, SageMaker, etc. will make the infra part much more straightforward.

Note: 
- an ML system has a rigidly defined output. No matter what experimental approaches you take, they are only valid if the output is in the correct format. E.g. a series of predictions on the grain `pu_location`-`date`.
- but an experiment can infinitely vary the inputs.

## Overview

1. Download taxi ride + zone + weather data idempotently

    ```bash
    just download-taxi-data \
        --year-month-start=2025-01 \
        --year-month-end-inclusive=2025-12

    just download-taxi-zone-dimension

    # Weather data from 4 Meteostat stations near Manhattan
    just download-weather-station-dimension
    just download-weather-data --start-date=2015-01-01 --end-date=2025-12-31
    ```

    - skips already downloaded files (use `--force` to overwrite)
    - taxi data defaults to last 3 months, acknowledging that the NYC TLC publishes data with a 2-month delay
    - weather stations: KNYC0 (Yorkville), KTEB0 (Teterboro), KJRB0 (Wall Street), 72502 (Newark). We would have taken the closest weather station to Central Park only, but 

2. Load into DuckDB and build curated layer

    ```bash
    just run-ddl-queries

    just merge-taxi-data \
        --year-month-start=2025-01 \
        --year-month-end-inclusive=2025-12

    just merge-taxi-zone-dimension

    just merge-curated-date-dim \
        --start-date=2025-01-01 \
        --end-date=2025-12-31

    just build-taxi-curated-layer \
        --year-month-start=2025-01 \
        --year-month-end-inclusive=2025-12

    # Weather: merge raw then build curated with priority-based coalescing
    just merge-weather-station-dimension
    just merge-weather-data --start-date=2015-01-01 --end-date=2025-12-31
    just build-weather-curated-layer
    ```

    | pickup_taxi_zone_id | pickup_date | number_ride_pickups | ... |
    | --- | --- | --- | --- |
    | 4 | 2025-01-01 | 269 | ... |
    | 4 | 2025-01-02 | 52 | ... |
    | 4 | 2025-01-03 | 88 | ... |

    - runs SQL queries parameterized by date against data in these files, uses merge into for idempotency
    - defaults to most recent 3 months, currently in the lakehouse.duckdb
    - curated weather tables coalesce 4 stations into one row per timestamp

**Consideration:** Weather forecast data is not the same as realized weather data.

1. When you are at a date D and you get a weather forecast for D + 1, D + 2, etc., there is uncertainty about how accurate that forecast will actually be. In other words, a forecasted temperature or humidity or wind speed or Air Quality Index reading will have some amount of error (residuals) compared to what it actually ends up being.
2. Realized historical weather data is just correct. It was what it was (except for measurement error in instruments). So, when training a model, there is a difference between using realized weather data and forecasted weather data.

How should we handle this? If nothing else, communicate that using "historical weather data" is cheating, because it is essentially being able to see the future. In other words, it is "future data leakage". Maybe we just accept this and say that in a real world scenario, it is ideal to have a backfilled historical dataset of what the weather FORECAST was on a date, not what the ACTUAL weather would be.

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

## Data Lineage

*To update or generate these diagrams, use the prompt hidden at this point as a comment in the raw README.md*

<!--

PROMPT FOR GENERATING DATA LINEAGE DIAGRAMS

Use this prompt with an AI assistant to generate or update data lineage diagrams for this project.

## Style Guide

### Flowchart Direction

- Use `flowchart TD` (top-down) for most diagrams
- Use `flowchart LR` (left-right) only when showing many parallel DDL-to-table mappings

### Color Scheme (classDef)

Always include these three class definitions at the top of each diagram:

```
classDef script fill:#e0f2fe,stroke:#0284c7
classDef file fill:#fef9c3,stroke:#ca8a04
classDef table fill:#dcfce7,stroke:#16a34a
```

- **Scripts** (blue): Python scripts, SQL query files, API calls
- **Files** (yellow): Local artifacts like .csv, .parquet, .zip files
- **Tables** (green): Database tables (raw.*, staging.*, curated.*)

### Structure

1. **Data Sources**: External APIs or services (e.g., "NYC TLC CloudFront", "Meteostat API")
2. **Script Boxes**: Group related scripts in labeled subgraphs (e.g., `subgraph download [10_download_*.py]`)
3. **File Artifacts**: Show intermediate files with full paths (e.g., `data/taxi/yellow/YYYY-MM.parquet`)
4. **Tables**: Show final database tables with schema prefix (e.g., `raw.taxi_zone_dim`)

### Node Naming

- Use descriptive IDs: `script_10`, `lookup_csv`, `raw_zone`
- Apply class with `:::script`, `:::file`, or `:::table`
- For scripts inside boxes, show the SQL file or action (e.g., `merge__raw__taxi_zone_dim.sql`)

### Flow

- Arrows show data flow: source → script → file → script → table
- Keep flows top-to-bottom or left-to-right consistently within a diagram

## Example

Use arrows (double-hyphen greater-than) to connect nodes:

    API[External API]:::script
    subgraph download [download_script.py]
        dl[download data]:::script
    end
    file[data/folder/file.parquet]:::file
    table[raw.table_name]:::table

    API (arrow) dl (arrow) file (arrow) table

-->

### 0* Scripts - DDL (Schema Creation)

```mermaid
flowchart LR
    classDef script fill:#e0f2fe,stroke:#0284c7
    classDef table fill:#dcfce7,stroke:#16a34a

    subgraph scripts_0 [00_run_ddl_queries.py]
        ddl1[ddl__raw__taxi_yellow_tripdata_fact.sql]:::script
        ddl2[ddl__raw__taxi_zone_dim.sql]:::script
        ddl3[ddl__staging_taxi_zone_centroids.sql]:::script
        ddl4[ddl__curated__date_dim.sql]:::script
        ddl5[ddl__curated__datetime_hour_dim.sql]:::script
        ddl6[ddl__raw__weather_station_dim.sql]:::script
        ddl7[ddl__raw__weather_hourly_fact.sql]:::script
        ddl8[ddl__raw__weather_daily_fact.sql]:::script
        ddl9[ddl__curated__weather_hourly_fact.sql]:::script
        ddl10[ddl__curated__weather_daily_fact.sql]:::script
    end

    subgraph tables_0 [Tables Created]
        raw_taxi[raw.taxi_yellow_tripdata_fact]:::table
        raw_zone[raw.taxi_zone_dim]:::table
        stg_centroid[staging.taxi_zone_centroids_dim]:::table
        cur_date[curated.date_dim]:::table
        cur_hour[curated.datetime_hour_dim]:::table
        raw_station[raw.weather_station_dim]:::table
        raw_whourly[raw.weather_hourly_fact]:::table
        raw_wdaily[raw.weather_daily_fact]:::table
        cur_whourly[curated.weather_hourly_fact]:::table
        cur_wdaily[curated.weather_daily_fact]:::table
    end

    ddl1 --> raw_taxi
    ddl2 --> raw_zone
    ddl3 --> stg_centroid
    ddl4 --> cur_date
    ddl5 --> cur_hour
    ddl6 --> raw_station
    ddl7 --> raw_whourly
    ddl8 --> raw_wdaily
    ddl9 --> cur_whourly
    ddl10 --> cur_wdaily
```

### 1* Scripts - Taxi Rides

```mermaid
flowchart TD
    classDef script fill:#e0f2fe,stroke:#0284c7
    classDef file fill:#fef9c3,stroke:#ca8a04
    classDef table fill:#dcfce7,stroke:#16a34a

    subgraph download [10_download_taxi_rides_yyyy_mm.py]
        script_10[NYC TLC CloudFront]:::script
    end

    parquet[data/taxi/yellow/YYYY-MM.parquet]:::file

    subgraph merge [11_merge_taxi_rides_into_duckdb.py]
        script_11[merge__raw__taxi_yellow_tripdata_fact.sql]:::script
    end

    raw_taxi[raw.taxi_yellow_tripdata_fact]:::table

    script_10 --> parquet --> script_11 --> raw_taxi
```

### 2* Scripts - Taxi Zone Dimension

```mermaid
flowchart TD
    classDef script fill:#e0f2fe,stroke:#0284c7
    classDef file fill:#fef9c3,stroke:#ca8a04
    classDef table fill:#dcfce7,stroke:#16a34a

    TLC[NYC TLC CloudFront]:::script

    subgraph download [20, 21, 22 Download and Extract]
        script_20[20: download taxi_zone_lookup.csv]:::script
        script_21[21: download taxi_zones.zip]:::script
        script_22[22: extract centroids csv]:::script
    end

    lookup_csv[data/taxi/zones/taxi_zone_lookup.csv]:::file
    shapes_zip[data/taxi/zones/taxi_zones.zip]:::file
    centroids_csv[data/taxi/zones/taxi_zone_centroids.csv]:::file

    subgraph merge [23, 24 Merge Scripts]
        script_23[23: merge__raw__taxi_zone_dim.sql]:::script
        script_24[24: merge__raw__taxi_zone_centroids_dim.sql]:::script
    end

    raw_zone[raw.taxi_zone_dim]:::table
    stg_centroid[staging.taxi_zone_centroids_dim]:::table

    TLC --> script_20 --> lookup_csv --> script_23 --> raw_zone
    TLC --> script_21 --> shapes_zip --> script_22 --> centroids_csv --> script_24 --> stg_centroid
```

### 3* Scripts - Taxi Curated Layer

```mermaid
flowchart TD
    classDef script fill:#e0f2fe,stroke:#0284c7
    classDef table fill:#dcfce7,stroke:#16a34a

    subgraph inputs [Dependencies from 1* and 2*]
        raw_taxi[raw.taxi_yellow_tripdata_fact]:::table
        raw_zone[raw.taxi_zone_dim]:::table
        stg_centroid[staging.taxi_zone_centroids_dim]:::table
    end

    subgraph script_30 [30_merge_curated_date_dim.py]
        merge_date[merge__curated__date_dim.sql]:::script
        merge_datetime[merge__curated__datetime_hour_dim.sql]:::script
        cur_date[curated.date_dim]:::table
        cur_datetime[curated.datetime_hour_dim]:::table
    end

    subgraph script_31 [31_build_taxi_curated_layer.py]
        view_stg[view__staging__taxi_yellow_tripdata_fact.sql]:::script
        view_zone[view__curated__taxi_zone_dim.sql]:::script
        view_fact[view__curated__taxi_yellow_tripdata_fact.sql]:::script
        stg_taxi[staging.taxi_yellow_tripdata_fact]:::table
        cur_zone[curated.taxi_zone_dim]:::table
        cur_fact[curated.taxi_yellow_tripdata_fact]:::table
    end

    merge_date --> cur_date
    merge_datetime --> cur_datetime
    raw_taxi --> view_stg --> stg_taxi --> view_fact --> cur_fact
    cur_date --> view_fact
    raw_zone --> view_zone --> cur_zone
    stg_centroid --> view_zone
```

### 4* Scripts - Weather Data

```mermaid
flowchart TD
    classDef script fill:#e0f2fe,stroke:#0284c7
    classDef file fill:#fef9c3,stroke:#ca8a04
    classDef table fill:#dcfce7,stroke:#16a34a

    subgraph download [40, 42 Download Scripts]
        script_40[40_download_weather_station_dim.py]:::script
        script_42a[42: Meteostat Hourly API]:::script
        script_42b[42: Meteostat Daily API]:::script
    end

    subgraph data_folders [data/weather/]
        station_csv[station_dim/weather_stations.csv]:::file
        hourly_pq[hourly/*.parquet]:::file
        daily_pq[daily/*.parquet]:::file
    end

    subgraph merge_scripts [41, 43 Merge Scripts]
        merge_station[41: merge__raw__weather_station_dim.sql]:::script
        merge_hourly[43: merge__raw__weather_hourly_fact.sql]:::script
        merge_daily[43: merge__raw__weather_daily_fact.sql]:::script
    end

    subgraph raw_tables [Raw Tables]
        raw_station[raw.weather_station_dim]:::table
        raw_hourly[raw.weather_hourly_fact]:::table
        raw_daily[raw.weather_daily_fact]:::table
    end

    subgraph curate_44 [44_build_weather_curated_layer.py]
        view_hourly[view__curated__weather_hourly_fact.sql]:::script
        view_daily[view__curated__weather_daily_fact.sql]:::script
    end

    subgraph curated_tables [Curated Tables]
        cur_hourly[curated.weather_hourly_fact]:::table
        cur_daily[curated.weather_daily_fact]:::table
    end

    script_40 --> station_csv --> merge_station --> raw_station
    script_42a --> hourly_pq --> merge_hourly --> raw_hourly --> view_hourly --> cur_hourly
    script_42b --> daily_pq --> merge_daily --> raw_daily --> view_daily --> cur_daily
```

## Local Folder Structure

```
local/
  scripts/
    00_run_ddl_queries.py - run all ddl__*.sql files
    10_download_taxi_rides_yyyy_mm.py - download monthly yellow taxi parquet files
    11_merge_taxi_rides_into_duckdb.py - merge taxi ride parquet files into raw table
    20_download_taxi_zone_lookup.py - download TLC taxi zone lookup CSV
    21_download_taxi_zone_shapes.py - download TLC taxi zone shapefile ZIP
    22_extract_taxi_zone_centroids_csv.py - extract centroids CSV from shapefile ZIP
    23_merge_taxi_zone_lookup_into_duckdb.py - merge taxi zone lookup CSV into raw table
    24_merge_taxi_zone_centroids_into_duckdb.py - merge centroid rows into staging table
    30_merge_curated_date_dim.py - merge curated.date_dim for a date range
    31_build_taxi_curated_layer.py - build staged and curated taxi tables
    40_download_weather_station_dim.py - download weather station metadata from Meteostat
    41_merge_weather_station_dim.py - merge weather station dim into DuckDB
    42_download_weather_data.py - download hourly + daily weather data from Meteostat
    43_merge_weather_into_duckdb.py - merge weather parquet into raw tables
    44_build_weather_curated_layer.py - build curated weather tables with coalesce
  queries/
    ddl__raw__taxi_yellow_tripdata_fact.sql - raw yellow taxi trips table DDL
    merge__raw__taxi_yellow_tripdata_fact.sql - merge raw taxi rides from parquet files
    view__staging__taxi_yellow_tripdata_fact.sql - staging table built from raw rides
    view__curated__taxi_yellow_tripdata_fact.sql - curated taxi rides table
    ddl__raw__taxi_zone_dim.sql - raw taxi zone dimension DDL
    merge__raw__taxi_zone_dim.sql - merge taxi zone lookup CSV into raw dim
    ddl__staging_taxi_zone_centroids.sql - staging centroids dimension DDL
    merge__raw__taxi_zone_centroids_dim.sql - merge computed centroids into staging
    view__curated__taxi_zone_dim.sql - curated taxi zone dimension table
    ddl__curated__date_dim.sql - curated date dimension DDL
    merge__curated__date_dim.sql - merge date dimension for a date range
    ddl__curated__datetime_hour_dim.sql - curated hour dimension DDL
    merge__curated__datetime_hour_dim.sql - merge datetime hour dimension for a date range
    ddl__raw__weather_station_dim.sql - raw weather station dimension DDL
    ddl__raw__weather_daily_fact.sql - raw daily weather fact DDL
    ddl__raw__weather_hourly_fact.sql - raw hourly weather fact DDL
    ddl__curated__weather_daily_fact.sql - curated daily weather fact DDL
    ddl__curated__weather_hourly_fact.sql - curated hourly weather fact DDL
    merge__raw__weather_station_dim.sql - merge weather station CSV into raw dim
    merge__raw__weather_daily_fact.sql - merge daily weather parquet into raw
    merge__raw__weather_hourly_fact.sql - merge hourly weather parquet into raw
    view__curated__weather_daily_fact.sql - curated daily weather with coalesce
    view__curated__weather_hourly_fact.sql - curated hourly weather with coalesce
    initial_daily_taxi_rides.sql - daily pickup series per zone for modeling
```