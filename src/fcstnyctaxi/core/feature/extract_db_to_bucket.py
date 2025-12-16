# %%
import uuid
from google.cloud import bigquery, storage
import pandas as pd
from pathlib import Path

from datetime import datetime, timezone
from fcstnyctaxi.lib.utils import get_project_root_dir, load_config_file
from fcstnyctaxi.lib.dataio import read_sql


# %%
def extract_db_to_bucket(
    project_id: str,
    db_location: str,
    bucket_name: str,
    prefix: str,
    sql_query_str: str,
) -> str:
    # put in pipeline file
    # project_root_path = get_project_root_dir()
    # put in pipeline file
    # sql = read_sql(project_root_path / "queries" / query_filename)

    # run BQ
    bq_client = bigquery.Client(project=project_id, location=db_location)
    job = bq_client.query(sql_query_str)
    print("Billed project (job.project):", job.project)
    df = job.result().to_dataframe()

    # consider later transfering queried data directly to GCS rather than
    # going through pandas and a local write

    # write to local .parquet file w/ string in the filename: UUID_utctimenow
    dataset_id = uuid.uuid4().hex[:8]
    timestamp = datetime.now(timezone.utc).strftime("UTC%Y%m%d_%H%M%S")
    filename = f"manhattan_daily_zone_pickups_{dataset_id}_UTC{timestamp}.parquet"
    tmp_dir = Path("/tmp/data")
    tmp_dir.mkdir(parents=False, exist_ok=True)
    local_path = tmp_dir / filename
    df.to_parquet(local_path, index=False)

    # upload to GCS
    gcs_uri = f"gs://{bucket_name}/{prefix}/{filename}"
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"{prefix}/{filename}")
    blob.upload_from_filename(local_path)

    # TO DO: any print statement should be part of logging
    print(f"Wrote {len(df)} rows to {gcs_uri}")

    return gcs_uri


# %%
if __name__ == "__main__":
    # local smoke test
    from fcstnyctaxi.lib.utils import get_project_root_dir, load_config_file

    # load configs
    config_filename = "config_daily_zone.yaml"
    project_root_path = get_project_root_dir()
    config_path = project_root_path / "config" / config_filename
    configs = load_config_file(config_path)

    env = configs["project_settings"].get("env")
    gcs_prefix = configs["extract_db_to_bucket"].get("gcs_prefix")
    gcs_prefix = env + "/" + gcs_prefix

    query_filename = (configs["extract_db_to_bucket"].get("query_filename"),)
    sql_query_str = read_sql(project_root_path / "queries" / query_filename)
    # put in pipeline file
    # project_root_path = get_project_root_dir()
    # put in pipeline file
    # sql = read_sql(project_root_path / "queries" / query_filename)

    extract_db_to_bucket(
        project_id=configs["project_settings"].get("project_id"),
        db_location=configs["project_settings"].get("location"),
        bucket_name=configs["project_settings"].get("gcs_bucket_name"),
        prefix=gcs_prefix,
        sql_query_str=sql_query_str,
    )
