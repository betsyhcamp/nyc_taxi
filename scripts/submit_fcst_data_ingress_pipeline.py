"""Compile and submit fcst-data-ingress-pipeline to Vertex AI Pipelines."""

import os

from google.cloud import aiplatform
from kfp import compiler

from fcstnyctaxi.lib.config.loading import load_pipeline_config
from fcstnyctaxi.lib.io import (
    build_run_scoped_uri,
    prepare_sql,
)
from fcstnyctaxi.lib.utils import generate_run_id, get_project_root_dir


def main() -> None:
    project_root = get_project_root_dir()

    config_path = project_root / "config" / "configs_zone_demand_pipeline.yaml"

    config = load_pipeline_config(config_path)

    run_id = generate_run_id()

    sql_path = project_root / "queries" / config.extract_db_to_bucket.sql_filename
    preparedsql = prepare_sql(
        sql_path=sql_path, sql_params=config.extract_db_to_bucket.sql_params
    )

    output_gcs_uri = build_run_scoped_uri(
        bucket=config.project_settings.bucket_name,
        prefix=config.extract_db_to_bucket.gcs_prefix,
        run_id=run_id,
        filename=config.extract_db_to_bucket.output_filename,
    )

    sql_sidecar_gcs_uri = build_run_scoped_uri(
        bucket=config.project_settings.bucket_name,
        prefix=config.extract_db_to_bucket.gcs_prefix,
        run_id=run_id,
        filename="query.sql",
    )

    # MUST set env var BEFORE importing pipeline. Setting after import has no effect.
    os.environ["FCST_EXTRACT_IMAGE"] = config.docker.extract_db_to_bucket
    from fcstnyctaxi.pipelines.fcst_data_ingress_pipeline import (
        fcst_data_ingress_pipeline,
    )

    compiled_pipeline_path = project_root / "build" / "fcst-data-ingress-pipeline.yaml"
    compiled_pipeline_path.parent.mkdir(parents=True, exist_ok=True)
    compiler.Compiler().compile(
        pipeline_func=fcst_data_ingress_pipeline,  # type: ignore[arg-type]
        package_path=str(compiled_pipeline_path),
    )

    aiplatform.init(
        project=config.project_settings.project_id,
        location=config.project_settings.location,
    )

    job = aiplatform.PipelineJob(
        display_name=f"{config.vertex.display_name_prefix}-{run_id}",
        template_path=str(compiled_pipeline_path),
        pipeline_root=config.vertex.pipeline_root,
        parameter_values={
            "project_id": config.project_settings.project_id,
            "bq_location": config.project_settings.location,
            "sql_query": preparedsql.sql_text,
            "sql_sha256": preparedsql.sha256,
            "output_gcs_uri": output_gcs_uri,
            "sql_sidecar_gcs_uri": sql_sidecar_gcs_uri,
            "run_id": run_id,
        },
        enable_caching=False,  # Keep False until trust in pipeline established
    )

    job.submit(service_account=config.vertex.pipeline_service_account)

    print(f"Submitted: {job.display_name}")
    print(f"Resource: {job.resource_name}")
    print(f"Console: {job._dashboard_uri()}")  # private SDK method; replace if breaks


if __name__ == "__main__":
    main()
