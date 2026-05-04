import logging
import time
from pathlib import Path

from google.cloud import bigquery

from fcstnyctaxi.core.feature.extract_db_to_bucket_impl import extract_db_to_bucket_impl
from fcstnyctaxi.lib.config import load_pipeline_config
from fcstnyctaxi.lib.io import PreparedSql, build_run_scoped_uri, prepare_sql
from fcstnyctaxi.lib.utils import generate_run_id, get_project_root_dir
from fcstnyctaxi.schemas.config_schemas import PipelineConfig

logger = logging.getLogger(__name__)


def prepare_extract_step_inputs(
    config: PipelineConfig, run_id: str, project_root: Path
) -> tuple[PreparedSql, str]:
    """Prepare the SQL and output URI for the extract step."""
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
    return preparedsql, output_gcs_uri


def main() -> None:
    logging.Formatter.converter = time.gmtime
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03dZ %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    run_id = generate_run_id()

    project_root = get_project_root_dir()
    config_path = project_root / "config" / "configs_zone_demand_pipeline.yaml"
    config = load_pipeline_config(config_path)

    bq_client = bigquery.Client(
        project=config.project_settings.project_id,
        location=config.project_settings.location,
    )

    preparedsql, output_gcs_uri = prepare_extract_step_inputs(
        config=config, run_id=run_id, project_root=project_root
    )

    bq_stats, write_meta = extract_db_to_bucket_impl(
        sql_query=preparedsql.sql_text,
        output_gcs_uri=output_gcs_uri,
        bq_client=bq_client,
    )

    logger.info(
        "extract_db_to_bucket complete: rows=%d job_id=%s uri=%s size=%s "
        "run_id=%s sqlsha=%s",
        bq_stats.total_rows,
        bq_stats.job_id,
        output_gcs_uri,
        write_meta.get("size"),
        run_id,
        preparedsql.sha256,
    )


if __name__ == "__main__":
    main()
