"""Manual smoke test (not unit test) of local KFP subprocess runner hitting
GCP resources via credentials set up in Google Application Default Credentials"""

import os

import kfp.local

from fcstnyctaxi.lib.config.loading import load_pipeline_config
from fcstnyctaxi.lib.io import (
    build_run_scoped_uri,
    prepare_sql,
)
from fcstnyctaxi.lib.utils import generate_run_id, get_project_root_dir

_ADC_PATH_IN_CONTAINER = "/kfp-workspace/application_default_credentials.json"


def main() -> None:
    project_root = get_project_root_dir()

    config_path = project_root / "config" / "configs_zone_demand_pipeline.yaml"
    config = load_pipeline_config(config_path)

    # MUST set env var BEFORE importing the wrapper — the @dsl.component decorator
    # captures base_image at import time. Setting after import has no effect.
    os.environ["FCST_EXTRACT_IMAGE"] = config.docker.extract_db_to_bucket

    from fcstnyctaxi.components.feature.extract_db_to_bucket_component import (
        extract_db_to_bucket,
    )

    kfp.local.init(
        runner=kfp.local.DockerRunner(
            environment={
                "GOOGLE_APPLICATION_CREDENTIALS": _ADC_PATH_IN_CONTAINER,
                "GOOGLE_CLOUD_PROJECT": "nyc-taxi-ehc",  # silences warning
            },
        ),
        workspace_root=os.path.expanduser("~/.config/gcloud"),
    )

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

    task = extract_db_to_bucket(  # type: ignore[call-arg]
        project_id=config.project_settings.project_id,
        bq_location=config.project_settings.location,
        sql_query=preparedsql.sql_text,
        sql_sha256=preparedsql.sha256,
        output_gcs_uri=output_gcs_uri,
        sql_sidecar_gcs_uri=sql_sidecar_gcs_uri,
        run_id=run_id,
    )

    snapshot = task.outputs["snapshot"]
    print("Component completed.")
    print(f"    uri: {snapshot.uri}\n    metadata: {snapshot.metadata}")


if __name__ == "__main__":
    main()
