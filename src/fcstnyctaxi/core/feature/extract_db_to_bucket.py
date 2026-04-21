from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
from google.cloud import bigquery
from tsbricks.blocks.dataio import (
    read_sql,
    render_sql_template,
    query_to_dataframe,
    BigQueryQueryStats,
    write_df_to_gcs_parquet,
)


def extract_db_to_bucket_impl(
    sql_path: Path,
    sql_params: Mapping[str, object],
    bq_client: bigquery.Client,
    bucket_name: str,
    gcs_prefix: str,
    job_config: bigquery.QueryJobConfig | None = None,
) -> tuple[str, BigQueryQueryStats, Mapping[str, Any]]:
    """Extract data from BigQuery and write the result as a Parquet file to GCS.

    Composes read_sql, render_sql_template, query_to_dataframe, and
    write_df_to_gcs_parquet into a single callable so the component stays thin.
    Documented exception to the core/ no-GCP-clients rule: this step has no
    business logic — it exists solely to compose IO functions.

    Args:
        sql_path (Path): Path to the .sql file containing the query to execute.
        sql_params (Mapping[str, object]): Parameters to substitute into the SQL
            template as a mapping of placeholder name to value. Pass an empty
            mapping if the SQL contains no Jinja placeholders; rendering is
            skipped in that case.
        bq_client (bigquery.Client): Authenticated BigQuery client used to run
            the query.
        bucket_name (str): Name of the destination GCS bucket.
        gcs_prefix (str): GCS object prefix (path between the bucket name and
            the filename, without leading or trailing slashes).
        job_config (bigquery.QueryJobConfig | None, optional): Optional BigQuery
            job configuration (e.g., maximum_bytes_billed, dry_run). Defaults to
            None.

    Raises:
        RuntimeError: If the BigQuery query returns zero rows.

    Returns:
        tuple[str, BigQueryQueryStats, Mapping[str, Any]]: The destination GCS
            URI of the written Parquet file, BigQuery query execution
            statistics, and GCS write confirmation metadata.
    """
    sql_text = read_sql(sql_path)

    if sql_params:
        sql_text = render_sql_template(sql_text, sql_params)

    df, bq_stats = query_to_dataframe(sql_text, client=bq_client, job_config=job_config)

    if bq_stats.total_rows == 0:
        raise RuntimeError(
            f"BigQuery query returned zero rows (job_id={bq_stats.job_id})."
            f"SQL file: {sql_path}."
        )

    filename = "manhattan_daily_zone_pickups.parquet"

    gs_uri = f"gs://{bucket_name}/{gcs_prefix}/{filename}"

    write_metadata = write_df_to_gcs_parquet(df, gs_uri)

    return gs_uri, bq_stats, write_metadata
