# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: venv-m5 (Local)
#     language: python
#     name: .venv
# ---

# %%
from google.cloud import bigquery

# %%
DDL_PATH = "../queries/"
DDL_FILENAMES = [
    "ddl__curated__date_dim.sql",
    "ddl__curated__datetime_hour_dim.sql",
    "ddl__curated__taxi_yellow_tripdata_fact.sql",
    "ddl__raw__taxi_yellow_tripdata_fact.sql",
    "ddl__raw__weather_daily_fact.sql",
    "ddl__raw__weather_hourly_fact.sql",
]

# %%
for file in DDL_FILENAMES:
    filepath = DDL_PATH + file
    print(filepath)
    with open(filepath, "r") as file_object:
        sql_script = file_object.read()
        print(sql_script)
        client = bigquery.Client()
        client.query(sql_script).result()
