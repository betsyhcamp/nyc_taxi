from kfp import dsl

from fcstnyctaxi.components.feature.extract_db_to_bucket_component import (
    extract_db_to_bucket,
)


@dsl.pipeline(
    name="fcst-data-ingress-pipeline",
    description="Data ingress pipeline (F) in FTI architecture in NYC Taxi forecasting",
)
def fcst_data_ingress_pipeline(
    project_id: str,
    bq_location: str,
    sql_query: str,
    sql_sha256: str,
    output_gcs_uri: str,
    sql_sidecar_gcs_uri: str,
    run_id: str,
) -> None:
    """Data ingress pipeline (F) in FTI architecture in NYC Taxi forecasting. Callers
    pass concrete values for every parameter at submission time"""

    extract_db_to_bucket(  # type: ignore[call-arg]
        project_id=project_id,
        bq_location=bq_location,
        sql_query=sql_query,
        sql_sha256=sql_sha256,
        output_gcs_uri=output_gcs_uri,
        sql_sidecar_gcs_uri=sql_sidecar_gcs_uri,
        run_id=run_id,
    )
