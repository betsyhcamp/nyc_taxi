"""Republish the fixed-path Data Preparation artifacts in the F->T contract form.

The Training pipeline needs a panel and a fiscal calendar carrying
`feature_run_id` at run-scoped paths, and the Feature pipeline that will write
them does not exist yet. This reads the two artifacts `notebooks/data_prep.py`
already wrote, adds the column, writes them under `build_run_prefix`'s
convention, and prints the flags the local training runner takes.

Disposable by construction: delete it the day the Feature pipeline publishes
these artifacts.
"""

import argparse

import pandas as pd

from fcstnyctaxi.core.train.compose_configs_impl import compose_train_static_configs
from fcstnyctaxi.lib.io import build_run_prefix
from fcstnyctaxi.lib.utils import (
    generate_run_id,
    get_project_root_dir,
    require_path_safe_run_id,
)

# Hardcoded because they are the pre-convention locations this script migrates
# away from, and they exist only under dev whatever --env says.
_SOURCE_PANEL_URI = "gs://nyc-taxi-ehc--modeling/dev/backtests/data/time_series.parquet"
_SOURCE_CALENDAR_URI = (
    "gs://nyc-taxi-ehc--modeling/dev/backtests/data/fiscal_calendar.parquet"
)

# Hardcoded values; Step named for the notebook it imitates, not for any Feature step.
_STEP = "data_prep"
_PANEL_FILENAME = "time_series.parquet"
_CALENDAR_FILENAME = "fiscal_calendar.parquet"

_LINEAGE_COLUMN = "feature_run_id"


def _parse_args() -> argparse.Namespace:
    """Environment selector, and an optional id so a run can be republished."""
    parser = argparse.ArgumentParser(
        description="Republish the fixed-path artifacts with feature_run_id added."
    )
    parser.add_argument(
        "--env",
        required=True,
        choices=["dev"],
        help="Environment selector. Dev only: sources are set to dev only, presently.",
    )
    parser.add_argument(
        "--feature-run-id",
        default=None,
        help="Reuse an existing id, overwriting its artifacts; one is generated "
        "when absent.",
    )
    return parser.parse_args()


def _with_lineage_column(frame: pd.DataFrame, feature_run_id: str) -> pd.DataFrame:
    """Return `frame` carrying `feature_run_id` as dtype "string", never null.

    The dtype and the null policy are the contract, not an implementation
    detail; the consumer checks neither, which leaves both to this producer.
    """
    frame = frame.copy()
    frame[_LINEAGE_COLUMN] = feature_run_id
    frame[_LINEAGE_COLUMN] = frame[_LINEAGE_COLUMN].astype("string")
    return frame


def main() -> None:
    """Republish both artifacts under one feature run id, then print the flags.

    Raises:
        ValueError: If `--env` has no `environments/<env>.yaml`, if
            `--feature-run-id` is not path-safe, or on any composition failure.
    """
    args = _parse_args()
    feature_run_id = (
        generate_run_id() if args.feature_run_id is None else args.feature_run_id
    )
    require_path_safe_run_id(feature_run_id, "--feature-run-id")

    environment, _, _ = compose_train_static_configs(
        get_project_root_dir() / "config", args.env
    )
    run_prefix = build_run_prefix(
        bucket=environment.config.storage.bucket_name,
        env=args.env,
        slice_name="feature",
        run_id=feature_run_id,
    )
    panel_uri = f"{run_prefix}{_STEP}/{_PANEL_FILENAME}"
    calendar_uri = f"{run_prefix}{_STEP}/{_CALENDAR_FILENAME}"

    panel_df = _with_lineage_column(pd.read_parquet(_SOURCE_PANEL_URI), feature_run_id)
    calendar_df = _with_lineage_column(
        pd.read_parquet(_SOURCE_CALENDAR_URI), feature_run_id
    )
    panel_df.to_parquet(panel_uri, index=False)
    calendar_df.to_parquet(calendar_uri, index=False)

    print(f"--feature-run-id {feature_run_id} \\")
    print(f"--panel-uri {panel_uri} \\")
    print(f"--calendar-uri {calendar_uri}")


if __name__ == "__main__":
    main()
