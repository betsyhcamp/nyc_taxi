# %%
import uuid
from google.cloud import bigquery, storage
import pandas as pd
from pathlib import Path

from datetime import datetime, timezone
from fcstnyctaxi.lib.utils import get_project_root_dir, load_config_file
from fcstnyctaxi.lib.dataio import read_sql

# %%
# get config file
config_filename = "config_daily_zone.yaml"
project_root_path = get_project_root_dir()
config_path = project_root_path / "config" / config_filename
configs = load_config_file(config_path)

# %%
project_root_path

# %%
# hard code configs used herein (for initial development work)
project_id = "nyc-taxi-ehc"
bq_location = "us-central1"


sql = """
WITH max_pickup_date_cte AS (
    SELECT
        DATE_SUB(
            DATE_ADD(
                MAX(
                    DATE(
                        REGEXP_EXTRACT(sourcefile_name, r'(\d{4}-\d{2})') || '-01'
                    )
                ),
                INTERVAL 1 MONTH
            ),
            INTERVAL 1 DAY
        ) AS max_pickup_date
    FROM `nyc-taxi-ehc.curated.taxi_yellow_tripdata_fact`
),

daily_timeseries_cte AS (
    SELECT
        trip.pickup_taxi_zone_id,
        trip.pickup_date,
        COUNT(trip.trip_id) AS number_ride_pickups
    FROM `nyc-taxi-ehc.curated.taxi_yellow_tripdata_fact` AS trip
    LEFT JOIN `nyc-taxi-ehc.curated.taxi_zone_dim` AS zone
        ON trip.pickup_taxi_zone_id = zone.taxi_zone_id
    WHERE
        zone.borough = 'Manhattan'
        AND NOT (trip.fare_amount = 0 OR trip.total_amount = 0)
        AND NOT (trip.trip_distance = 0 AND trip.total_amount <= 0)
        AND pickup_date <= (
            SELECT max_pickup_date
            FROM max_pickup_date_cte
        )
    GROUP BY
        pickup_taxi_zone_id,
        pickup_date
),

min_date_bound_cte AS (
    SELECT
        pickup_taxi_zone_id,
        MIN(pickup_date) AS min_pickup_date
    FROM daily_timeseries_cte
    GROUP BY pickup_taxi_zone_id
),

pickup_taxi_zone_id_cal AS (
    SELECT
        mn.pickup_taxi_zone_id,
        d.calendar_date
    FROM min_date_bound_cte AS mn
    CROSS JOIN max_pickup_date_cte AS mx
    INNER JOIN `nyc-taxi-ehc.curated.date_dim` AS d
        ON
            mn.min_pickup_date <= d.calendar_date
            AND mx.max_pickup_date >= d.calendar_date
)

SELECT
    p.pickup_taxi_zone_id,
    p.calendar_date AS pickup_date,
    d.calendar_year,
    d.calendar_month,
    d.day_of_week,
    d.day_of_week_name,
    d.is_weekend,
    d.is_holiday,
    d.holiday_name,
    d.is_daylight_savings,
    COALESCE(t.number_ride_pickups, 0) AS number_ride_pickups
FROM pickup_taxi_zone_id_cal AS p
LEFT JOIN daily_timeseries_cte AS t
    ON
        p.calendar_date = t.pickup_date
        AND p.pickup_taxi_zone_id = t.pickup_taxi_zone_id
LEFT JOIN `nyc-taxi-ehc.curated.date_dim` AS d
    ON p.calendar_date = d.calendar_date
ORDER BY
    p.pickup_taxi_zone_id,
    p.calendar_date

"""

# %%
utc_now = datetime.now(timezone.utc)
ts = utc_now.strftime("UTC%Y%m%d_%H%M%S")
print(ts)

# %%
# run BQ
bq_client = bigquery.Client(project=project_id, location=bq_location)
job = bq_client.query(sql)
print("Billed project (job.project):", job.project)
df = job.result().to_dataframe()

# %%
df

# %%
# run BQ
bq_client = bigquery.Client(project=project_id, location=bq_location)
job = bq_client.query(sql)
df = job.result().to_dataframe()


# %%
# Optionally set datatypes (TO DO: make function for this)

# write to local .parquet file w/ string in the filename: UUID_utctimenow
dataset_id = uuid.uuid4().hex[:8]
ts = datetime.now(timezone.utc).strftime("UTC%Y%m%d_%H%M%S")
filename = f"manhattan_daily_pickups_{dataset_id}_UTC{ts}.parquet"

tmp_dir = project_root_path / "tmp_data"
tmp_dir.mkdir(parents=False, exist_ok=True)
local_path = tmp_dir / filename
df.to_parquet(local_path, index=False)


# %%
# upload to GCS


# %%
def extract_db_to_bucket(
    config_filename: str,
    project_id: str,
    db_location: str,
    bucket_name: str,
    prefix: str = "dev/initial_datapull",
) -> str:
    # TO DO: create datatype conversion function

    return gcs_uri


# %%
