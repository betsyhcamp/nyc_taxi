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
from pandas.tseries.offsets import MonthBegin
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 250)

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
    base_dir=Path.cwd()

taxi_ride_filepath = base_dir.parent / DATA_DIR_NAME / DATA_SUBDIR_NAME / TAXI_RIDE_FILENAME
taxi_zone_lookup_filepath = base_dir.parent / DATA_DIR_NAME / DATA_SUBDIR_NAME / TAXI_ZONE_LOOKUP

# %%
taxi_zone_df = pd.read_csv(taxi_zone_lookup_filepath)

# %%
for file_path in (base_dir.parent / DATA_DIR_NAME / DATA_SUBDIR_NAME).glob('*.parquet'):
    print(file_path)
    filename = str(file_path).split('/')[-1]
    output_filename = 'cleaned_'+filename
    output_aggregated_filename = 'cleaned_aggregated'+filename
    file_month_start_datetime = pd.to_datetime(filename.rstrip(".parquet").split("_")[2])
    file_next_month_start_datetime = file_month_start_datetime + MonthBegin(1)
    
    taxi_df = pd.read_parquet(file_path)
    
    start_cutoff_pickup  = file_month_start_datetime + MonthBegin(-1)
    end_cutoff_pickup  = file_month_start_datetime + MonthBegin(2)
    
    temp_mask = (taxi_df["tpep_pickup_datetime"]< start_cutoff_pickup ) | (taxi_df["tpep_pickup_datetime"]>end_cutoff_pickup)
    temp_df = taxi_df[temp_mask]
    nrows, _ = temp_df.shape
    print(f"number of rows with timestamps outside of bounds{nrows}") 
    print(temp_df)
    taxi_df = taxi_df[~temp_mask].reset_index(drop=True)
    
    taxi_df = taxi_df[~((taxi_df["fare_amount"] ==0)| (taxi_df["total_amount"] ==0))]
    taxi_df = taxi_df[~((taxi_df["trip_distance"] ==0) & (taxi_df["total_amount"] <=0))]
    taxi_df.to_parquet(base_dir.parent / DATA_DIR_NAME / DATA_CLEANED_SUBDIR_NAME/output_filename)
    
    taxi_df["pickup_date"] =  taxi_df["tpep_pickup_datetime"].dt.date
    taxi_df["count"]=0
    taxi_agg_df =  taxi_df[["pickup_date", "PULocationID", "count"]].groupby(["pickup_date", "PULocationID"]).count().reset_index(drop=False)
    taxi_agg_df.to_parquet(base_dir.parent / DATA_DIR_NAME / DATA_CLEANED_SUBDIR_NAME/output_aggregated_filename) 
    
    
