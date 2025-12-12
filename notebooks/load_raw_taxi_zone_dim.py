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
import pandas as pd

# %%
gcs_path_file = (
    "gs://nyc-taxi-ehc--data-ingest/raw/nyc_tlc_taxi/taxi_metadata/taxi_zone_lookup.csv"
)

taxi_zone_df = pd.read_csv(gcs_path_file)

# %%
taxi_zone_df.head()
taxi_zone_df.info()

# %%
taxi_zone_df

# %%
taxi_zone_df["load_timestamp_utc"] = datetime.now(timezone.utc)

# %%
table_id = "nyc-taxi-ehc.raw.taxi_zone_dim"
client = bigquery.Client()
client.query(f"""
  TRUNCATE TABLE `{table_id}`
""").result()

# %%
job_config = bigquery.LoadJobConfig(
    write_disposition="WRITE_APPEND",
)

# %%
job = client.load_table_from_dataframe(taxi_zone_df, table_id, job_config=job_config)
job.result()
