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
    `require_columns` names the frame lacking the column.
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


def require_matching_feature_run_id(panel_df: pd.DataFrame, expected: str) -> str:
    """The id the panel carries, checked against the one the caller declared.

    The panel alone: Feature stamps no other artifact, so their lineage is the common
    source of their URIs. `expected` is a claim; the observation gets stamped.
    """
    panel_id = require_single_feature_run_id(panel_df, "panel")
    if panel_id != expected:
        raise ValueError(
            f"panel carries feature_run_id {panel_id!r}, not the declared "
            f"{expected!r}: wrong URI, or wrong bytes at the requested location."
        )
    return panel_id


def trim_to_allowlist(
    frame: pd.DataFrame,
    *,
    required: tuple[str, ...],
    allowed: tuple[str, ...] | None = None,
    frame_name: str,
) -> pd.DataFrame:
    """Keep `allowed`'s columns; raise unless every `required` one is present.

    `allowed` defaults to `required`; the reasoning is in `schemas/run_outputs.py`.
    """
    require_columns(frame, list(required), frame_name)
    keep = required if allowed is None else allowed
    return frame[[column for column in keep if column in frame.columns]]
