# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python (.venv) (Local)
#     language: python
#     name: venv
# ---

# %%
from google.cloud import bigquery
from google.cloud import storage
from datetime import datetime, timezone
from decimal import Decimal
import pandas as pd
import numpy as np


# %%
def list_gcs_files(bucket_name: str, prefix: str):
    """List all blobs in a GCS bucket under a given prefix."""
    client = storage.Client()

    # prefix should NOT include 'gs://bucket-name/'
    # e.g., prefix = "raw/taxi_yellow/" or "" for whole bucket
    blobs = client.list_blobs(bucket_name, prefix=prefix)

    paths = [blob.name for blob in blobs]
    return paths


# %%
bucket_name = "nyc-taxi-ehc--data-ingest"
prefix_in_bucket = "raw/nyc_tlc_taxi/yellow_taxi_ride_data/"

required_columns_dtypes = {
    "vendorid": "Int64",
    "tpep_pickup_datetime": "datetime",
    "tpep_dropoff_datetime": "datetime",
    "passenger_count": "Int64",
    "trip_distance": float,
    "ratecodeid": "Int64",
    "store_and_fwd_flag": str,
    "pulocationid": "Int64",
    "dolocationid": "Int64",
    "payment_type": "Int64",
    "fare_amount": float,
    "extra": float,
    "mta_tax": float,
    "tip_amount": float,
    "tolls_amount": float,
    "improvement_surcharge": float,
    "total_amount": float,
    "congestion_surcharge": float,
    "airport_fee": float,
    "cbd_congestion_fee": float,
}

# %%
load_timestamp_utc = datetime.now(timezone.utc)

required_columns = set(required_columns_dtypes.keys())
col_datatypes = set(required_columns_dtypes.values())

# %%
table_id = "nyc-taxi-ehc.raw.taxi_yellow_tripdata_fact"
client = bigquery.Client()
client.query(f"""
  TRUNCATE TABLE `{table_id}`
""").result()

# %%
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",
)

# %%
files_list = [
    file for file in list_gcs_files(bucket_name, prefix_in_bucket) if ".parquet" in file
]


# %%
# files_list = [files[-1]]

# %%
for file in files_list:
    gcs_file_path = f"gs://{bucket_name}/{file}"
    print(gcs_file_path)
    sourcefile_name = file.split("/")[-1]
    file_df = pd.read_parquet(gcs_file_path)

    file_df.columns = [col.lower() for col in file_df.columns]

    if "cbd_congestion_fee" not in file_df.columns:
        file_df["cbd_congestion_fee"] = np.nan

    columns_not_in_intersection = required_columns.symmetric_difference(
        set(file_df.columns)
    )
    if list(columns_not_in_intersection):
        raise ValueError(f"Columns not correct: {columns_not_in_intersection}")

    for col, datatype in required_columns_dtypes.items():
        if datatype == "datetime":
            file_df[col] = pd.to_datetime(file_df[col])
        else:
            file_df[col] = file_df[col].astype(datatype)

    file_df["sourcefile_name"] = sourcefile_name

    file_df["load_timestamp_utc"] = load_timestamp_utc

    job = client.load_table_from_dataframe(file_df, table_id, job_config=job_config)
    job.result()
