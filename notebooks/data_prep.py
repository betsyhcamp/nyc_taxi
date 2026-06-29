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
import sys
from pathlib import Path

import pandas as pd

from google.cloud import bigquery
from tsbricks.blocks.dataio import read_sql, query_to_dataframe, write_df_to_gcs

from fcstnyctaxi.lib.utils import get_project_root_dir

# %%
project = "nyc-taxi-ehc"
location = "us-central1"
input_sql_filename = "initial_daily_taxi_revenue.sql"

project_root = get_project_root_dir()

queries_dir = project_root / "queries"

# %%
sql_str = read_sql(queries_dir / input_sql_filename)

# %%
client = bigquery.Client(project=project, location=location)
df, *_ = query_to_dataframe(sql_str, client=client)

# %%
dtype_map = {
    "pickup_taxi_zone_id": "int64",
    "pickup_date":"datetime64[ns]",
    "day_of_fiscal_month": "int64",
    "fiscal_week": "int64",
    "fiscal_week_of_month": "int64",
    "fiscal_month": "int64",
    "fiscal_year": "int64",
    "fiscal_year_month": "int64",
    "day_of_week": "int64",
    "day_of_week_name": "object",
    "is_weekend": "bool",
    "is_holiday": "bool",
    "holiday_name": "object",
    "is_daylight_savings": "bool",
    "fiscal_week_start_date": "datetime64[ns]",
    "fiscal_year_week": "int64",
    "number_ride_pickups": "int64",
}

df = df.astype(dtype_map)

# %%
df.info()
df.head()

# %%
ts_df = (
    df
    .groupby(["fiscal_week_start_date", "pickup_taxi_zone_id"])
    ["number_ride_pickups"]
    .sum()
    .reset_index(drop=False)
    .rename(columns={
        "fiscal_week_start_date":"ds",
        "pickup_taxi_zone_id":"unique_id",
        "number_ride_pickups":"y"
    })
)

# %%
# TODO: Move this calculation to SQL
df["weeks_in_month"] = (
    df
    .groupby("fiscal_year_month")["fiscal_week_of_month"]
    .transform("max")
)

# %%
df.head()

# %%
df[["day_of_week_name", "is_weekend"]].drop_duplicates()

# %%
day_col = ["pickup_date",
    "is_weekend",
    "is_holiday"]
cal_col = [
    "fiscal_week_start_date",
    "fiscal_year_month",
    "fiscal_year_week",
    "fiscal_week_of_month",
    "weeks_in_month",
]
cal_df = df[day_col+cal_col].drop_duplicates().copy()

cal_df["is_workday"] = (
    (~cal_df["is_weekend"])
    & (~cal_df["is_holiday"])
)

cal_df = cal_df.groupby(cal_col)["is_workday"].sum().reset_index(drop=False)

# %%
cal_df.info()
cal_df.head()


# %%
uri_time_series = "gs://nyc-taxi-ehc--modeling/dev/backtests/data/time_series.parquet"
gcs_result_ts = write_df_to_gcs(ts_df, uri_time_series)

# %%
uri_cal = "gs://nyc-taxi-ehc--modeling/dev/backtests/data/fiscal_calendar.parquet"
gcs_result_cal = write_df_to_gcs(cal_df, uri_cal)
