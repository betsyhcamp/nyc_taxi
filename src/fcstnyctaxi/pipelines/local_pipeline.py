import logging
import time

from google.cloud import bigquery

from fcstnyctaxi.core.feature.extract_db_to_bucket import extract_db_to_bucket_impl
from fcstnyctaxi.lib.config import load_pipeline_config
from fcstnyctaxi.lib.utils import get_project_root_dir

logger = logging.getLogger(__name__)


def main() -> None:
    logging.Formatter.converter = time.gmtime
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03dZ %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    project_root = get_project_root_dir()
    config_path = project_root / "config" / "configs_zone_demand_pipeline.yaml"
    config = load_pipeline_config(config_path)

    bq_client = bigquery.Client(
        project=config.project_settings.project_id,
        location=config.project_settings.location,
    )

    sql_path = project_root / "queries" / config.extract_db_to_bucket.sql_filename

    gcs_uri, bq_stats, write_meta = extract_db_to_bucket_impl(
        sql_path=sql_path,
        sql_params=config.extract_db_to_bucket.sql_params,
        bq_client=bq_client,
        bucket_name=config.project_settings.bucket_name,
        gcs_prefix=config.extract_db_to_bucket.gcs_prefix,
    )

    logger.info(
        "extract_db_to_bucket complete: rows=%d job_id=%s uri=%s size=%s",
        bq_stats.total_rows,
        bq_stats.job_id,
        gcs_uri,
        write_meta.get("size"),
    )


if __name__ == "__main__":
    main()
