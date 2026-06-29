# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: .venv-etl (3.12.9)
#     language: python
#     name: python3
# ---

# %%
from datetime import datetime, timezone
from google.cloud import bigquery
import pandas as pd
import numpy as np
import holidays

# %%
INITIAL_DATE = '2017-12-31'
HOLIDAY_VERSION = 'v0.1.0'

# %%
initial_date = pd.to_datetime(INITIAL_DATE)
current = datetime.now()
current_utc = datetime.now(timezone.utc)
end_date = pd.to_datetime(current) + pd.DateOffset(months=3)


# %%
def make_year_arrays(year: int, month_lengths: list[int])-> np.ndarray:
    total_days = sum(month_lengths)

    if total_days % 7 !=0:
        raise ValueError("Yearly fiscal days must be divisible by 7; got remainder.")

    fiscal_days = np.concatenate([
        np.arange(1, days_in_month+1)
        for days_in_month in month_lengths
    ])

    weeks_in_year = total_days // 7

    fiscal_weeks =np.repeat(np.arange(1, weeks_in_year +1), 7)

    fiscal_week_of_month = np.concatenate([ 
        np.repeat(np.arange(1, (days_in_month // 7)+1), 7)
        for days_in_month in month_lengths
    ])

    fiscal_months = np.concatenate([
        np.repeat(month, days_in_month)
        for month, days_in_month in enumerate(month_lengths, start=1)
    ])
    
    fiscal_years = np.repeat(year, len(fiscal_days))

    return np.column_stack([
        fiscal_days,
        fiscal_weeks,
        fiscal_week_of_month,
        fiscal_months,
        fiscal_years
    ])


# %%
regular_month_lengths = [28, 28, 35] * 4
long_year_month_lengths = [28, 28, 35] * 3 + [28, 28, 42]

long_years = {2020, 2026}

fiscal_year_arrays = []

for year in range(2018, 2028):
    if year in long_years:
        month_lengths = long_year_month_lengths
    else:
        month_lengths = regular_month_lengths

    fiscal_year_arrays.append(
        make_year_arrays(year, month_lengths)
    )

fiscal_array = np.vstack(fiscal_year_arrays)

# %%
fiscal_df = pd.DataFrame(
    fiscal_array, 
    columns=["day_of_fiscal_month", "fiscal_week", "fiscal_week_of_month", "fiscal_month", "fiscal_year"]
)

# %%
fiscal_df["fiscal_year_month"]=(
    fiscal_df["fiscal_year"].astype(str)
    + fiscal_df["fiscal_month"].astype(str).str.zfill(2)
).astype(int)

# %%
fiscal_df.head()

# %%
daily_cal_df = pd.DataFrame({'calendar_date':pd.date_range(start=initial_date, end=end_date, freq='D', tz=None)})

# %%
daily_cal_df = daily_cal_df.merge(fiscal_df, left_index=True, right_index=True, how="left")

# %%
daily_cal_df['calendar_year'] = daily_cal_df['calendar_date'].dt.year
daily_cal_df['calendar_month'] = daily_cal_df['calendar_date'].dt.month
daily_cal_df['month_name'] = daily_cal_df['calendar_date'].dt.month_name()
daily_cal_df['calendar_day'] = daily_cal_df['calendar_date'].dt.day

# pandas uses Sunday as day 6 and Monday as day 0; desire Sunday as day 0
daily_cal_df['day_of_week'] = (daily_cal_df['calendar_date'].dt.dayofweek + 1)%7
daily_cal_df['day_of_week_name'] = daily_cal_df['calendar_date'].dt.day_name()
daily_cal_df['is_weekend']=daily_cal_df['day_of_week'].isin([0,6])

# %%
holidays_country_code = 'US'
holidays_subdivision = 'NY'
holiday_type = 'observed'


years = range(
    daily_cal_df["calendar_date"].dt.year.min(),
    daily_cal_df["calendar_date"].dt.year.max() + 1,
)
us_ny_holidays = holidays.US(state=holidays_subdivision, years=years, observed=bool(holiday_type))

holidays_series = daily_cal_df["calendar_date"].dt.date.map(us_ny_holidays)
daily_cal_df["is_holiday"] = holidays_series.notnull()
daily_cal_df["holiday_name"] = holidays_series
daily_cal_df['holiday_country_code']=holidays_country_code

daily_cal_df["calendar_ny_timezone"] = daily_cal_df["calendar_date"].dt.tz_localize("America/New_York")
daily_cal_df["dst_offset"] = daily_cal_df["calendar_ny_timezone"].map(lambda ts: ts.dst())
daily_cal_df["is_daylight_savings"] = daily_cal_df["dst_offset"] != pd.Timedelta(0)

# %%
daily_cal_df['holiday_calendar_version'] = '-'.join(
    [
        holidays_country_code,
        holidays_subdivision,
        holiday_type, 
        HOLIDAY_VERSION 
    ]
)

# %%
daily_cal_df['load_timestamp_utc'] = current_utc

# %%
daily_cal_df = daily_cal_df.drop(columns=['calendar_ny_timezone', 'dst_offset'])

# %%
daily_cal_df.info()
daily_cal_df.head()

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
