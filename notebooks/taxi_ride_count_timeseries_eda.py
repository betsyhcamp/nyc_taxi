# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: nyc_taxi
#     language: python
#     name: python3
# ---

# %%
from pathlib import Path

import pandas as pd
from pandas.tseries.offsets import MonthEnd, MonthBegin

pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 250)

# %%
DATA_DIR_NAME = "data"
DATA_SUBDIR_NAME = "raw"
DATA_CLEANED_SUBDIR_NAME = "cleaned"
TAXI_RIDE_FILENAME = "yellow_tripdata_2025-03.parquet"
TAXI_ZONE_LOOKUP = "taxi_zone_lookup.csv"

# %%
try:
    base_dir = Path(__file__).parent
except NameError:
    base_dir = Path.cwd()

# %%
taxi_zone_lookup_filepath = (
    base_dir.parent / DATA_DIR_NAME / DATA_SUBDIR_NAME / TAXI_ZONE_LOOKUP
)
taxi_zone_df = pd.read_csv(taxi_zone_lookup_filepath)

# %%
filepath_list = [
    file_path
    for file_path in (base_dir.parent / DATA_DIR_NAME / DATA_CLEANED_SUBDIR_NAME).glob(
        "*.parquet"
    )
    if "aggregated" in str(file_path).split("/")[-1]
]

# %%
ride_ts_df = pd.DataFrame(columns=["pickup_date", "PULocationID", "count"])

for filepath in filepath_list:
    temp_df = pd.read_parquet(filepath)
    source_year_month_str = (
        str(filepath).split("/")[-1].split("_")[-1].rstrip(".parquet").strip()
    )
    temp_df["source_year_month"] = source_year_month_str

    ride_ts_df = pd.concat([ride_ts_df, temp_df])

ride_ts_df = ride_ts_df.reset_index(drop=True)
ride_ts_df["source_year_month"] = pd.to_datetime(ride_ts_df["source_year_month"])
ride_ts_df["pickup_date"] = pd.to_datetime(ride_ts_df["pickup_date"])

# %%
ride_ts_df["pickup_date"] = pd.to_datetime(ride_ts_df["pickup_date"])

# %%
ride_ts_df.head()
ride_ts_df.info()

# %%
min_source_datetime = ride_ts_df["source_year_month"].min()
max_source_plus_one_month_datetime = ride_ts_df["source_year_month"].max() + MonthBegin(
    1
)

# %%
# remove partial months
ride_ts_df = ride_ts_df[
    (ride_ts_df["pickup_date"] >= min_source_datetime)
    & (ride_ts_df["pickup_date"] < max_source_plus_one_month_datetime)
]

# %%
zero_filled_ts_df = (
    pd.DataFrame(
        pd.date_range(
            start=ride_ts_df["pickup_date"].min(),
            end=ride_ts_df["pickup_date"].max(),
            freq="D",
        )
    )
    .rename(columns={0: "pickup_date"})
    .join(taxi_zone_df[["LocationID"]].drop_duplicates(), how="cross")
    .rename(columns={"LocationID": "PULocationID"})
)
zero_filled_ts_df["count"] = 0

# %%
ride_ts_df = (
    pd.concat([ride_ts_df, zero_filled_ts_df])
    .groupby(["pickup_date", "PULocationID"])["count"]
    .sum()
    .reset_index(drop=False)
)

# %%
ride_ts_df.info()
ride_ts_df.head()

# %%
ride_ts_df = ride_ts_df.merge(
    taxi_zone_df[["LocationID", "Borough"]].drop_duplicates(),
    how="left",
    left_on=["PULocationID"],
    right_on=["LocationID"],
)

ride_ts_df = ride_ts_df.drop(columns="LocationID")

# %%
ride_ts_df = ride_ts_df[ride_ts_df["Borough"].isin(["Manhattan"])].reset_index(
    drop=True
)

# %%
ride_ts_df["count"] = ride_ts_df["count"].astype(float)

# %%
ride_ts_df["pickup_date"].dt.year.unique()

# %%
min_ride_count = ride_ts_df["count"].min()
max_ride_count = ride_ts_df["count"].max()

# %%


for year in ride_ts_df["pickup_date"].dt.year.unique():
    temp_df = ride_ts_df[ride_ts_df["pickup_date"].dt.year == year]
    temp_df["pickup_date"] = temp_df["pickup_date"].dt.date
    temp_pivot_df = temp_df.pivot_table(
        index="PULocationID", columns="pickup_date", values="count"
    )
    plt.figure(figsize=(16, 16))
    sns.heatmap(temp_pivot_df, cmap="viridis", vmin=min_ride_count, vmax=max_ride_count)
    plt.title(f"Count taxi ride over year {year}")
    plt.xlabel("date")
    # plt.ylabel('Pickup location ID')
    plt.show()

# %%
temp_df = ride_ts_df[ride_ts_df["pickup_date"].dt.year == year]
temp_df["count"] = temp_df["count"].astype(float)

# %%
temp_df.info()

# %%
temp_pivot_df.dtypes

# %%
plt.figure(figsize=(16, 16))
sns.heatmap(temp_pivot_df, cmap="viridis")
# plt.title('Heatmap of Values by Hour and Day')
# plt.xlabel('date')
# plt.ylabel('Pickup location ID')
plt.show()

# %%
temp_pivot_df.head()
temp_pivot_df.info()

# %%
ride_ts_df.pivot_table(index="PULocationID", columns="")

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Example: Create sample time series data
data = {
    "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
    "hour": [9, 14, 10, 15],
    "value": [25, 30, 28, 32],
}
df = pd.DataFrame(data)

# Extract month and year for pivoting
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day

# Pivot the data
pivot_df = df.pivot_table(index="hour", columns="day", values="value")

# %%
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_df, cmap="viridis", annot=True, fmt=".1f")
plt.title("Heatmap of Values by Hour and Day")
plt.xlabel("Day of Month")
plt.ylabel("Hour of Day")
plt.show()


# %%

# %%
