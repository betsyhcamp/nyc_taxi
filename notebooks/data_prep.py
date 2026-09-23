# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: nyc_taxi (3.12.9.final.0)
#     language: python
#     name: python3
# ---

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from google.cloud import bigquery
from tsbricks.blocks.dataio import read_sql, query_to_dataframe, write_df_to_gcs

from fcstnyctaxi.lib.utils import get_project_root_dir

# %%
project = "nyc-taxi-ehc"
location = "us-central1"
input_sql_filename = "initial_daily_taxi_revenue.sql"
calendar_sql_filename = "daily_fiscal_calendar.sql"

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
# TODO : Could consider putting the "ds", "unique_id", "y" alias in SQL
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
cal_df, *_ = query_to_dataframe(
    read_sql(queries_dir / calendar_sql_filename), client=client
)

# %%
# Load-bearing: a BigQuery DATE arrives as object or dbdate, and a nullable bool
# sums to Int64 rather than int64.
cal_df = cal_df.astype({
    "calendar_date": "datetime64[ns]",
    "fiscal_week_start_date": "datetime64[ns]",
    "fiscal_week_of_month": "int64",
    "fiscal_month": "int64",
    "fiscal_year": "int64",
    "fiscal_year_month": "int64",
    "fiscal_year_week": "int64",
    "is_weekend": "bool",
    "is_holiday": "bool",
})

# %%
cal_df.info()
cal_df.head()

# %%
# TODO: Move these calculations to SQL
cal_df["weeks_in_month"] = (
    cal_df
    .groupby("fiscal_year_month")["fiscal_week_of_month"]
    .transform("max")
)

cal_df["origin_month_fraction_elapsed"] = (
    cal_df["fiscal_week_of_month"]
    / cal_df["weeks_in_month"]
)

# %%
day_col = ["calendar_date",
    "is_weekend",
    "is_holiday"]
cal_col = [
    "fiscal_week_start_date",
    "fiscal_year_month",
    "fiscal_year",
    "fiscal_month",
    "fiscal_year_week",
    "fiscal_week_of_month",
    "weeks_in_month",
    "origin_month_fraction_elapsed"
]
cal_df = cal_df[day_col+cal_col].drop_duplicates().copy()

cal_df["is_workday"] = (
    (~cal_df["is_weekend"])
    & (~cal_df["is_holiday"])
)

cal_df["is_weekday_holiday"] = (
    cal_df["is_holiday"]
    & (~cal_df["is_weekend"])
)

cal_df = (
    cal_df
    .groupby(cal_col)[["is_workday", "is_weekday_holiday"]]
    .sum()
    .reset_index(drop=False)
)

# %%
cal_df 

# %%
# TODO : Could consider putting the "ds" alias in SQL
cal_df = cal_df.rename(columns={
    "fiscal_week_start_date": "ds",
    "is_workday":"count_workdays",
    "is_weekday_holiday":"holiday_days_in_week"
    }
)

# %%
# The weekly holiday count rides on the calendar's own group-by, so the two frames
# hold the same weeks by construction. The calendar keeps the columns it has today.
weekly_df = cal_df[["ds", "holiday_days_in_week"]].copy()
cal_df = cal_df.drop(columns="holiday_days_in_week")

# %%
cal_df.info()
cal_df.head()

# %%
week_angle = 2 * np.pi * weekly_df["ds"].dt.dayofyear / 365.25
weekly_df["week_sin"] = np.sin(week_angle)
weekly_df["week_cos"] = np.cos(week_angle)

# The history side is the panel's own keys, so a series that starts late starts
# where it starts. Every series runs to the panel's last week, so the weeks after
# it are the same grid for all of them.
future_weeks = weekly_df.loc[weekly_df["ds"] > ts_df["ds"].max(), ["ds"]]
exog_df = pd.concat(
    [
        ts_df[["unique_id", "ds"]],
        ts_df[["unique_id"]].drop_duplicates().merge(future_weeks, how="cross"),
    ],
    ignore_index=True,
)
# A duplicated week would fan the merge out silently: the cast below catches a
# missing count, not a repeated one.
exog_df = exog_df.merge(
    weekly_df[["ds", "holiday_days_in_week", "week_sin", "week_cos"]],
    on="ds",
    how="left",
    validate="many_to_one",
)

# Raises rather than filling when a count is missing, which would mean the weekly
# frame lost a row.
exog_df = exog_df.astype(
    {"holiday_days_in_week": "int64", "week_sin": "float64", "week_cos": "float64"}
)

# %%
exog_df.info()
exog_df.head()


# %%
uri_time_series = "gs://nyc-taxi-ehc--modeling/dev/backtests/data/time_series.parquet"
gcs_result_ts = write_df_to_gcs(ts_df, uri_time_series)

# %%
uri_cal = "gs://nyc-taxi-ehc--modeling/dev/backtests/data/fiscal_calendar.parquet"
gcs_result_cal = write_df_to_gcs(cal_df, uri_cal)

# %%
uri_exog = "gs://nyc-taxi-ehc--modeling/dev/backtests/data/exogenous_features.parquet"
gcs_result_exog = write_df_to_gcs(exog_df, uri_exog)
