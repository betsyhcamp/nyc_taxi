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
from datetime import datetime, timezone
from google.cloud import bigquery
import pandas as pd
import holidays

# %%
INITIAL_DATE = "2018-01-01"
HOLIDAY_VERSION = "v0.1.0"

# %%
initial_date = pd.to_datetime(INITIAL_DATE)
current = datetime.now()
current_utc = datetime.now(timezone.utc)
end_date = pd.to_datetime(current) + pd.DateOffset(months=3)

# %%
daily_cal_df = pd.DataFrame(
    {
        "calendar_date": pd.date_range(
            start=initial_date, end=end_date, freq="D", tz=None
        )
    }
)

# %%
daily_cal_df["calendar_year"] = daily_cal_df["calendar_date"].dt.year
daily_cal_df["calendar_month"] = daily_cal_df["calendar_date"].dt.month
daily_cal_df["month_name"] = daily_cal_df["calendar_date"].dt.month_name()
daily_cal_df["calendar_day"] = daily_cal_df["calendar_date"].dt.day
daily_cal_df["day_of_week"] = daily_cal_df["calendar_date"].dt.dayofweek + 1
daily_cal_df["day_of_week_name"] = daily_cal_df["calendar_date"].dt.day_name()
daily_cal_df["is_weekend"] = daily_cal_df["day_of_week"].isin([6, 7])

# %%
holidays_country_code = "US"
holidays_subdivision = "NY"
holiday_type = "observed"


years = range(
    daily_cal_df["calendar_date"].dt.year.min(),
    daily_cal_df["calendar_date"].dt.year.max() + 1,
)
us_ny_holidays = holidays.US(
    state=holidays_subdivision, years=years, observed=bool(holiday_type)
)

holidays_series = daily_cal_df["calendar_date"].dt.date.map(us_ny_holidays)
daily_cal_df["is_holiday"] = holidays_series.notnull()
daily_cal_df["holiday_name"] = holidays_series
daily_cal_df["holiday_country_code"] = holidays_country_code

daily_cal_df["calendar_ny_timezone"] = daily_cal_df["calendar_date"].dt.tz_localize(
    "America/New_York"
)
daily_cal_df["dst_offset"] = daily_cal_df["calendar_ny_timezone"].map(
    lambda ts: ts.dst()
)
daily_cal_df["is_daylight_savings"] = daily_cal_df["dst_offset"] != pd.Timedelta(0)

# %%
daily_cal_df["holiday_calendar_version"] = "-".join(
    [holidays_country_code, holidays_subdivision, holiday_type, HOLIDAY_VERSION]
)

# %%
daily_cal_df["load_timestamp_utc"] = current_utc

# %%
daily_cal_df.info()
daily_cal_df.head()

# %%
daily_cal_df = daily_cal_df.drop(columns=["calendar_ny_timezone", "dst_offset"])

# %%
# TO DO : Put in code to get hourly calendar

# %%
table_id = "nyc-taxi-ehc.curated.date_dim"
client = bigquery.Client()
client.query(f"""
  TRUNCATE TABLE `{table_id}`
""").result()

# %%
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",
)

# %%
job = client.load_table_from_dataframe(daily_cal_df, table_id, job_config=job_config)
job.result()
