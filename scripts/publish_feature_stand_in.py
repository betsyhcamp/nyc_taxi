"""Republish the fixed-path Data Preparation artifacts in the F->T contract form.

The Training pipeline needs a panel carrying `feature_run_id`, a fiscal calendar
and an exogenous features file, at run-scoped paths, and the Feature pipeline that
will write them does not exist yet. This reads the three artifacts
`notebooks/data_prep.py` already wrote, adds the column to the panel as Feature
does, writes them under `resolve_run_prefix`'s convention, and prints the
`--feature-run-id` both training callers resolve from.

Disposable by construction: delete it the day the Feature pipeline publishes
these artifacts.
"""

import argparse
from datetime import UTC, datetime

import pandas as pd
from tsbricks.blocks.metadata import get_git_hash

from fcstnyctaxi.lib.io import write_text_to_gcs
from fcstnyctaxi.lib.storage_layout import resolve_run_outputs_uri, resolve_run_prefix
from fcstnyctaxi.lib.utils import (
    generate_run_id,
    get_project_root_dir,
    require_path_safe_run_id,
)
from fcstnyctaxi.schemas.run_identity import LINEAGE_COLUMN
from fcstnyctaxi.schemas.run_outputs import FeatureArtifacts, FeatureRunOutputs

# Hardcoded because they are the pre-convention locations this script migrates
# away from, and they exist only under dev whatever --env says.
_SOURCE_PANEL_URI = "gs://nyc-taxi-ehc--modeling/dev/backtests/data/time_series.parquet"
_SOURCE_CALENDAR_URI = (
    "gs://nyc-taxi-ehc--modeling/dev/backtests/data/fiscal_calendar.parquet"
)
_SOURCE_EXOG_URI = (
    "gs://nyc-taxi-ehc--modeling/dev/backtests/data/exogenous_features.parquet"
)

# Hardcoded values; Step named for the notebook it imitates, not for any Feature step.
_STEP = "data_prep"
_PANEL_FILENAME = "time_series.parquet"
_CALENDAR_FILENAME = "fiscal_calendar.parquet"
_EXOG_FILENAME = "exogenous_features.parquet"

# The version the reader expects. Written although the shipped producer file has
# none: the stand-in imitates the contract being asked for, not today's gap.
_SCHEMA_VERSION = "0.1.0"


def _parse_args() -> argparse.Namespace:
    """Environment selector, and an optional id so a run can be republished."""
    parser = argparse.ArgumentParser(
        description="Republish the three fixed-path artifacts, the panel with "
        "feature_run_id added."
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
    frame[LINEAGE_COLUMN] = feature_run_id
    frame[LINEAGE_COLUMN] = frame[LINEAGE_COLUMN].astype("string")
    return frame


def main() -> None:
    """Republish all three artifacts under one feature run id, then print that id.

    Raises:
        ValueError: If `--env` has no `environments/<env>.yaml`, if
            `--feature-run-id` is not path-safe, or on any composition failure.
    """
    args = _parse_args()
    feature_run_id = (
        generate_run_id() if args.feature_run_id is None else args.feature_run_id
    )
    require_path_safe_run_id(feature_run_id, "--feature-run-id")

    config_dir = get_project_root_dir() / "config"
    run_prefix = resolve_run_prefix(config_dir, args.env, "feature", feature_run_id)
    panel_uri = f"{run_prefix}{_STEP}/{_PANEL_FILENAME}"
    calendar_uri = f"{run_prefix}{_STEP}/{_CALENDAR_FILENAME}"
    exog_uri = f"{run_prefix}{_STEP}/{_EXOG_FILENAME}"

    panel_df = _with_lineage_column(pd.read_parquet(_SOURCE_PANEL_URI), feature_run_id)
    # No lineage column on these two: Feature stamps the panel alone.
    calendar_df = pd.read_parquet(_SOURCE_CALENDAR_URI)
    exog_df = pd.read_parquet(_SOURCE_EXOG_URI)
    panel_df.to_parquet(panel_uri, index=False)
    calendar_df.to_parquet(calendar_uri, index=False)
    exog_df.to_parquet(exog_uri, index=False)

    # Written last, after every parquet file: its presence is what marks the run
    # complete, so a URI stamped before the artifacts exist would be a promise.
    outputs = FeatureRunOutputs(
        schema_version=_SCHEMA_VERSION,
        feature_run_id=feature_run_id,
        env=args.env,
        git_hash=get_git_hash(),
        completed_at=datetime.now(UTC).isoformat(),
        published=FeatureArtifacts(
            panel_uri=panel_uri, calendar_uri=calendar_uri, exogenous_uri=exog_uri
        ),
        # Computed, never hardcoded. No series_admitted, series_dropped or
        # exogenous_columns: this script admits nothing, and nothing in Training
        # reads them.
        panel={
            "rows": len(panel_df),
            "series": panel_df["unique_id"].nunique(),
            # pd.Timestamp first: `ds` arrives as datetime64, object, or the
            # `dbdate` extension dtype a BigQuery DATE becomes, and `.min()`
            # returns a plain `datetime.date` for the latter two, which has
            # no `.date()`.
            "first_ds": str(pd.Timestamp(panel_df["ds"].min()).date()),
            "last_ds": str(pd.Timestamp(panel_df["ds"].max()).date()),
        },
    )
    write_text_to_gcs(
        outputs.model_dump_json(indent=2, exclude_none=True),
        resolve_run_outputs_uri(config_dir, args.env, "feature", feature_run_id),
    )

    # One pasteable line, so resolution is the path every run takes by default.
    print(f"--feature-run-id {feature_run_id}")
    print("\n# override, normally unnecessary:")
    print(f"#   --panel-uri {panel_uri}")
    print(f"#   --calendar-uri {calendar_uri}")
    print(f"#   --additional-exog-uri {exog_uri}")


if __name__ == "__main__":
    main()
