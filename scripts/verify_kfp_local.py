"""Manual smoke test (not unit test) of local KFP subprocess runner
hitting GCP resources via redentials set up in Google Application Default Credentials"""

import kfp.local

from fcstnyctaxi.components.feature.extract_db_to_bucket_component import (
    extract_db_to_bucket,
)
from fcstnyctaxi.lib.config import load_pipeline_config
from fcstnyctaxi.lib.io import (
    build_run_scoped_uri,
    prepare_sql,
)
from fcstnyctaxi.lib.utils import generate_run_id, get_project_root_dir


def main() -> None:
    kfp.local.init(runner=kfp.local.SubprocessRunner(use_venv=False))

    project_root = get_project_root_dir()

    config_path = project_root / "config" / "configs_zeon_demand_pipeline.yaml"
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

    # TODO: Take care of Pylance warning that "snapshot" param is missing
    task = extract_db_to_bucket(
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
    print(f"  uri: {snapshot.uri}\n  metadata: {snapshot.metadata}")


if __name__ == "__main__":
    main()
