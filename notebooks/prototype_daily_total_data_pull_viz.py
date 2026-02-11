# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: nyc_taxi (3.12.9)
#     language: python
#     name: python3
# ---

# %%
from google.cloud import bigquery
import pandas as pd
import pathlib

import matplotlib.pyplot as plt
from coreforecast.scalers import boxcox, inv_boxcox, boxcox_lambda
# from lib.utils import get_project_root_dir, load_config_file
# from lib.dataio import read_sql

# %%
project = "nyc-taxi-ehc"
location = "us-central1"

# %%
sql_text = """
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
client = bigquery.Client(project=project, location=location)
job_config = bigquery.QueryJobConfig(
    use_legacy_sql=False,
)

# %%
job = client.query(sql_text, job_config=job_config)

# %%
df = job.result().to_dataframe()

# %%
df

# %%
df.info()

# %%
df["pickup_date"] = pd.to_datetime(df["pickup_date"])

# %%
total_daily_df = (
    df.groupby("pickup_date")["number_ride_pickups"].sum().reset_index(drop=False)
)
total_daily_df

# %%
total_daily_df.to_parquet(
    "../data/preprocessed/daily_total_timeseries.parquet", index=False, engine="pyarrow"
)

# %%
# visualize entire time series
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(total_daily_df["pickup_date"], total_daily_df["number_ride_pickups"])

ax.set_title("Daily Yellow Taxi Pickups in Manhattan")
ax.set_xlabel("Date")
ax.set_ylabel("Number of Pickups")
ax.grid(alpha=0.2)
fig.autofmt_xdate()

plt.tight_layout()

plt.show()

# %% [markdown]
# Peak-to-peak variation of seasonality before COVID much larger than post COVID. Specifically, magnitude of peak-to-peak sesonal variation changes with the level of the time series. A forecasting model type that than handle multiplicative seasonality needed or do Box-Cox transform prior to forecasting is needed. Or, for baseline, a seasonal naive would be fine since changes in level are not significant over the 30 day forecast horizon.

# %%
# Just choosing several months to visualize weekly periodicity
mask = (total_daily_df["pickup_date"] >= "2024-01-01") & (
    total_daily_df["pickup_date"] <= "2024-12-31"
)
total_daily_df_2024 = total_daily_df.loc[mask].sort_values("pickup_date")

fig, ax = plt.subplots(figsize=(12, 4))  # you can tweak the size
ax.plot(total_daily_df_2024["pickup_date"], total_daily_df_2024["number_ride_pickups"])

ax.set_title("Daily Yellow Taxi Pickups in Manhattan")
ax.set_xlabel("Date")
ax.set_ylabel("Number of Pickups")
ax.grid(alpha=0.2)
# Optional: nicer date formatting / rotation
fig.autofmt_xdate()

plt.tight_layout()
plt.show()

# %%
bc_lambda = boxcox_lambda(
    total_daily_df["number_ride_pickups"].values,
    method="guerrero",
    season_length=7,
)
bc_ts = boxcox(total_daily_df["number_ride_pickups"].values, bc_lambda)

# %%
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(total_daily_df["pickup_date"], bc_ts)

ax.set_title("Daily Yellow Taxi Pickups in Manhattan")
ax.set_xlabel("Date")
ax.set_ylabel("Box Cox (Number of Pickups)")
ax.grid(alpha=0.2)
fig.autofmt_xdate()
plt.tight_layout()

plt.show()

# %% [markdown]
# Box Cox transform using Guerrero not appropriate. Issue likely due to multiple seasonality so MLE finds poor value for parameter `lambda`

# %%
bc_lambda = boxcox_lambda(
    total_daily_df["number_ride_pickups"].values,
    method="loglik",
    season_length=7,
)
bc_ts = boxcox(total_daily_df["number_ride_pickups"].values, bc_lambda)

# %%
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(total_daily_df["pickup_date"], bc_ts)

ax.set_title("Daily Yellow Taxi Pickups in Manhattan")
ax.set_xlabel("Date")
ax.set_ylabel("Box Cox (Number of Pickups)")
ax.grid(alpha=0.2)
fig.autofmt_xdate()
plt.tight_layout()

plt.show()

# %%
total_daily_df.info()

# %%
total_daily_df["boxcox_lambda"] = bc_lambda
total_daily_df["box_cox_timeseries"] = bc_ts


# %%
total_daily_df

# %%
df_long = total_daily_df.melt(
    id_vars="pickup_date",
    value_vars=["number_ride_pickups", "box_cox_timeseries"],
    var_name="unique_id",
    value_name="y",
)

# %%
df_long

# %%
df_long = df_long[["unique_id", "pickup_date", "y"]].rename(
    columns={"pickup_date": "ds"}
)

# %%
df_long.info()

# %%
df_long

# %%
from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA,
    AutoETS,
    AutoCES,
    AutoTheta,
    TBATS,
    SeasonalNaive,
)

# from utilsforecast.evaluation import evaluate
# from utilsforecast.losses import rmse

# %%
import scipy, statsmodels

print("scipy", scipy.__version__, "statsmodels", statsmodels.__version__)

# %%

# %%
