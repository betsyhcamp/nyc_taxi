import os

from kfp import dsl
from kfp.dsl import Dataset, Output

# Fallback is used only when callers don't set FCST_EXTRACT_IMAGE.
# verify_kfp_local.py and th submission script always set it from
# config.docker.extract_db_to_bucket. Bump this string only when emergency
# ad-hoc imports need a different default.
_IMAGE = os.environ.get(
    "FCST_EXTRACT_IMAGE",
    "us-central1-docker.pkg.dev/nyc-taxi-ehc/fcst-data-ingress-pipeline/extract-db-to-bucket:f286db5",
)


@dsl.component(
    base_image=_IMAGE,
    packages_to_install=[],
)
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
    from datetime import UTC, datetime

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
    extracted_at_utc = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

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
