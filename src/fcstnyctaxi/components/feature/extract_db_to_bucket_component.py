from __future__ import annotations

from kfp import dsl
from kfp.dsl import Dataset, Output


@dsl.component(
    base_image="python:3.12",
    packages_to_install=[],
)  # placeholders; Docker image set up in next PR
def extract_db_to_bucket(
    project_id: str,
    bq_location: str,
    sql_query: str,
    sql_sha256: str,
    output_gcs_uri: str,
    sql_sidecar_gcs_uri: str,
    run_id: str,
    snapshot: Output[Dataset],
) -> None:
    """Thin KFP component wrapper for extract_db_to_bucket_impl(), handles infra.

    Reads `sql_query` (already-rendered SQL text) and writes a Parquet
    snapshot to `output_gcs_uri`. After a successful query+write, writes
    the SQL text as a sidecar object to `sql_sidecar_gcs_uri`.
    Populates `snapshot.metadata` for downstream lineage.
    """
    from datetime import datetime, timezone

    from google.cloud import bigquery

    from fcstnyctaxi.core.feature.extract_db_to_bucket_impl import (
        extract_db_to_bucket_impl,
    )
    from fcstnyctaxi.lib.io import write_text_to_gcs

    bq_client = bigquery.Client(project=project_id, location=bq_location)

    bq_stats, write_meta = extract_db_to_bucket_impl(
        sql_query=sql_query,
        output_gcs_uri=output_gcs_uri,
        bq_client=bq_client,
    )
    write_text_to_gcs(text=sql_query, gcs_uri=sql_sidecar_gcs_uri)
    extracted_at_utc = datetime.strftime(
        datetime.now(timezone.utc), format="%Y-%m-%dT%H:%M:%S.%fZ"
    )

    snapshot.uri = output_gcs_uri
    snapshot.metadata.update(
        {
            "row_count": bq_stats.total_rows,
            "bq_query_job_id": bq_stats.job_id,
            "file_size_bytes": write_meta.get("size"),
            "run_id": run_id,
            "gcs_uri": output_gcs_uri,
            "extracted_at": extracted_at_utc,
            "sql_sha256": sql_sha256,
            "sql_sidecar_gcs_uri": sql_sidecar_gcs_uri,
            "git_sha": None,
        }
    )
