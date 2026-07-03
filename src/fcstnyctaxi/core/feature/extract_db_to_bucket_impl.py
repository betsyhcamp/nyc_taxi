from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from google.cloud import bigquery
from tsbricks.blocks.dataio import (
    BigQueryQueryStats,
    query_to_dataframe,
    write_df_to_gcs_parquet,
)


def extract_db_to_bucket_impl(
    sql_query: str,
    output_gcs_uri: str,
    bq_client: bigquery.Client,
    job_config: bigquery.QueryJobConfig | None = None,
) -> tuple[BigQueryQueryStats, Mapping[str, Any]]:
    """Extract data from BigQuery and write the result as a Parquet file to GCS.

    Documented exception to the core/ no-GCP-clients rule: this step composes BigQuery
    read and GCS write IO functions. SQL query template rendering and URI construction
    are in lib/io.py called by the component wrapper or local pipeline runner.

    Args:
        sql_query (str): Rendered SQL query text to execute
        output_gcs_uri (str): GCS URI location to write SQL query output to as fully
            formed URI.
        bq_client (bigquery.Client): Authenticated BigQuery client used to run
            the query.
        job_config (bigquery.QueryJobConfig | None, optional): Optional BigQuery
            job configuration (e.g., maximum_bytes_billed, dry_run). Defaults to
            None.

    Raises:
        RuntimeError: If the BigQuery query returns zero rows.

    Returns:
        tuple[BigQueryQueryStats, Mapping[str, Any]]: The BigQuery query execution
            statistics and GCS write confirmation metadata.
    """
    df, bq_stats = query_to_dataframe(
        sql_query, client=bq_client, job_config=job_config
    )

    if bq_stats.total_rows == 0:
        raise RuntimeError(
            f"BigQuery query returned zero rows (job_id={bq_stats.job_id})."
        )

    write_metadata = write_df_to_gcs_parquet(df, output_gcs_uri)

    return bq_stats, write_metadata
