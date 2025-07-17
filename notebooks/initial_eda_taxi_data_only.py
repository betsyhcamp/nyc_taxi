# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
from pathlib import Path

import pandas as pd
from pandas.tseries.offsets import MonthEnd, MonthBegin
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 250)

# %%
DATA_DIR_NAME = "data"
DATA_SUBDIR_NAME = "raw"
TAXI_RIDE_FILENAME = "yellow_tripdata_2025-03.parquet"
TAXI_ZONE_LOOKUP = "taxi_zone_lookup.csv"

# %%
try: 
    base_dir = Path(__file__).parent
except NameError:
    base_dir=Path.cwd()

taxi_ride_filepath = base_dir.parent / DATA_DIR_NAME / DATA_SUBDIR_NAME / TAXI_RIDE_FILENAME
taxi_zone_lookup_filepath = base_dir.parent / DATA_DIR_NAME / DATA_SUBDIR_NAME / TAXI_ZONE_LOOKUP

# %%
file_month_start_datetime = pd.to_datetime(TAXI_RIDE_FILENAME.rstrip(".parquet").split("_")[2])
file_next_month_start_datetime = file_month_start_datetime + MonthBegin(1)

# %%
taxi_df = pd.read_parquet(taxi_ride_filepath)
taxi_zone_df = pd.read_csv(taxi_zone_lookup_filepath)
taxi_df.info()
taxi_df.head()

# %%
taxi_df.describe()

# %%
null_mask = taxi_df.isnull()
null_counts = null_mask.sum()
null_counts

# %% [markdown]
# For demand forecasting, only really require columns:
# * [critical not null] tpep_pickup_datetime: The datetime, time meter started
# * [critical not null] PULocationID: TLC Taxi Zone in which taximeter engaged
#
# Have min `tpep_pickup_datetime` min timestamp that is from a different year. Need to check and exclude.
#
# Probably not coincidental that columns with missing values (`passenger_count`, `RatecodeID`, `store_and_forward_flag`, `congestion_surcharge`, `Airport_fee`). Maybe a particular VendorID doesn't provide that info or certain cars don't have a computer that stores that info.

# %%
groupby_col =['VendorID'] 
null_mask = null_mask.drop(columns=groupby_col)
null_count_per_vendor = taxi_df[groupby_col].merge(null_mask, left_index=True, right_index=True).groupby(groupby_col).sum()

# %%
null_count_per_vendor

# %%
taxi_df["difference_minutes_pickup_month_start"] = (taxi_df["tpep_pickup_datetime"] - file_month_start_datetime).dt.total_seconds()/60

# %%
taxi_df["difference_minutes_pickup_month_next_start"] = (taxi_df["tpep_pickup_datetime"] - file_next_month_start_datetime).dt.total_seconds()/60

# %%
taxi_df[taxi_df["difference_minutes_pickup_month_start"] <0]

# %%
taxi_df[taxi_df["difference_minutes_pickup_month_next_start"]>=0]

# %%
start_cutoff_pickup  = file_month_start_datetime + MonthBegin(-1)
end_cutoff_pickup  = file_month_start_datetime + MonthBegin(2)
temp_mask = (taxi_df["tpep_pickup_datetime"]< start_cutoff_pickup ) | (taxi_df["tpep_pickup_datetime"]>end_cutoff_pickup)

temp_df = taxi_df[temp_mask]

taxi_df = taxi_df[~temp_mask].reset_index(drop=True)


# %%
taxi_df["trip_duration_minutes"] = (taxi_df["tpep_dropoff_datetime"] - taxi_df["tpep_pickup_datetime"]).dt.total_seconds()/60

# %%
less_than_equal_zero_duration = taxi_df[taxi_df["trip_duration_minutes"]<=0]
print(less_than_equal_zero_duration.shape[0]/taxi_df.shape[0] *100)
less_than_equal_zero_duration

# %%
less_than_equal_zero_duration.describe()

# %%
less_than_equal_zero_total_amount = taxi_df[(taxi_df["total_amount"]<0)]
print(less_than_equal_zero_total_amount.shape[0]/taxi_df.shape[0] *100)
less_than_equal_zero_total_amount

# %%
less_than_equal_zero_distance = taxi_df[taxi_df["trip_distance"] <=0]
print(less_than_equal_zero_distance.shape[0]/taxi_df.shape[0] *100)
less_than_equal_zero_distance 

# %%
less_than_equal_zero_distance.describe()

# %%
#zero_trip_distance_fare_amount_mask = 



#zero_trip_distance_fare_amount_df = taxi_df[zero_total_amount_mask]
#zero_trip_distance_fare_amount_df



taxi_df[(taxi_df["trip_distance"] ==0) & ((taxi_df["fare_amount"] ==0)| (taxi_df["total_amount"] ==0))]

# %%
taxi_df["tpep_pickup_datetime"].dt.date

# %%
taxi_df["pickup_date"] =  taxi_df["tpep_pickup_datetime"].dt.date
taxi_df["count"]=0


# %%
taxi_agg_df =  taxi_df[["pickup_date", "PULocationID", "count"]].groupby(["pickup_date", "PULocationID"]).count().reset_index(drop=False)

# %%
taxi_agg_df.head(100)

# %%
import pandas as pd

df = pd.DataFrame({'Director': ['Steven Spielberg', 'Martin Scorsese', 'Steven Spielberg', 'Quentin Tarantino', 'Martin Scorsese', 'Steven Spielberg'],
                   'Movie': ['Jaws', 'Goodfellas', 'Jurassic Park', 'Pulp Fiction', 'Raging Bull', 'E.T.'],
                  'year':[1990, 1991, 1992, 1993, 1994, 1995]})

# %%
df

# %%
df['count']=0


# %%
df[['Director', 'count']].groupby('Director').count().reset_index(drop=False)

# %%
