import pandas as pd

from fcstnyctaxi.schemas.run_identity import LINEAGE_COLUMN


def require_columns(df: pd.DataFrame, required: list[str], frame_name: str) -> None:
    """Raise if df lacks any required column, naming the frame and what is missing.

    Checks a subset, not an exact column set, so extra columns pass. An empty
    required list passes vacuously.

    Args:
        df: Frame to check.
        required: Column names that must be present.
        frame_name: Name of the frame as the caller knows it, used in the error
            message so a missing column names its own source rather than
            surfacing as an error from somewhere downstream.

    Raises:
        ValueError: If any required column is absent, listing every missing
            name rather than only the first.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")


def require_single_feature_run_id(frame: pd.DataFrame, frame_name: str) -> str:
    """The one non-null `feature_run_id` a frame carries.

    Nulls are rejected separately because `nunique()` skips them, and
    `require_columns` names which of the two frames lacks the column.
    """
    require_columns(frame, [LINEAGE_COLUMN], frame_name)
    column = frame[LINEAGE_COLUMN]

    null_count = int(column.isna().sum())
    if null_count:
        raise ValueError(
            f"{frame_name} has {null_count} row(s) with a null "
            f"{LINEAGE_COLUMN}, so those rows carry no provenance."
        )

    distinct = sorted(column.unique())
    if len(distinct) != 1:
        raise ValueError(
            f"{frame_name} mixes Feature runs: {len(distinct)} distinct "
            f"{LINEAGE_COLUMN} values {distinct[:5]}."
        )

    return str(distinct[0])


def require_matching_feature_run_id(
    panel_df: pd.DataFrame, calendar_df: pd.DataFrame, expected: str
) -> str:
    """The id both frames carry, checked against the one the caller declared.

    Consistency between frames is checked first, since a shared value must exist
    before it can be compared. `expected` is a claim; the observation gets stamped.
    """
    panel_id = require_single_feature_run_id(panel_df, "panel")
    calendar_id = require_single_feature_run_id(calendar_df, "calendar")

    if panel_id != calendar_id:
        raise ValueError(
            f"panel feature_run_id {panel_id!r} != calendar {calendar_id!r}; "
            f"origins and actuals would come from different Feature runs."
        )
    if panel_id != expected:
        raise ValueError(
            f"Frames carry feature_run_id {panel_id!r}, not the declared "
            f"{expected!r}: wrong URIs, or wrong bytes at the requested location."
        )

    return panel_id
