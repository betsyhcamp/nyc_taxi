import pandas as pd
from pandas.api.types import is_numeric_dtype

from fcstnyctaxi.schemas.run_outputs import JOIN_KEYS


def _require_unique_calendar_dates(calendar_df: pd.DataFrame) -> None:
    """A repeated date duplicates a training row for every series."""
    repeated = calendar_df.loc[calendar_df["ds"].duplicated(), "ds"].unique()
    if len(repeated):
        raise ValueError(f"calendar repeats ds {list(repeated[:5])}.")


def _require_comparable_series_ids(
    panel_df: pd.DataFrame, additional_exog_df: pd.DataFrame
) -> None:
    """Numeric against text is the one key pairing pandas refuses to merge."""
    panel_ids = panel_df["unique_id"]
    exog_ids = additional_exog_df["unique_id"]
    if is_numeric_dtype(panel_ids) != is_numeric_dtype(exog_ids):
        raise ValueError(
            f"panel unique_id is {panel_ids.dtype} and the additional exogenous "
            f"file's is {exog_ids.dtype}: the two artifacts key series differently."
        )


def _require_unique_exog_keys(exog_df: pd.DataFrame) -> None:
    """Checked on the built frame: the additional artifact makes this a real join."""
    duplicated = exog_df.duplicated(list(JOIN_KEYS))
    if duplicated.any():
        sample = exog_df.loc[duplicated, list(JOIN_KEYS)].head(3)
        raise ValueError(
            f"assembled frame has {int(duplicated.sum())} duplicate key row(s): "
            f"{sample.to_dict('records')}."
        )


def _require_panel_rows_survive(joined: pd.DataFrame, panel_df: pd.DataFrame) -> None:
    """Against the panel's count: a ragged panel has fewer rows than the frame."""
    if len(joined) != len(panel_df):
        raise ValueError(
            f"exogenous join changed the panel from {len(panel_df)} to "
            f"{len(joined)} rows."
        )


def _require_complete_coverage(
    joined: pd.DataFrame, exog_features: tuple[str, ...]
) -> None:
    """A fit accepts NaN as a value, so a gap degrades the numbers silently."""
    if not exog_features:
        return

    incomplete = joined[list(exog_features)].isna().any(axis=1)
    if incomplete.any():
        sample = joined.loc[incomplete, list(JOIN_KEYS)].head(3)
        raise ValueError(
            f"{int(incomplete.sum())} panel row(s) have no exogenous values: "
            f"{sample.to_dict('records')}."
        )


def build_exog_frame(
    panel_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    additional_exog_df: pd.DataFrame,
    *,
    exog_features: tuple[str, ...],
) -> pd.DataFrame:
    """Assemble one exogenous frame, the file driving and the calendar broadcast.

    One frame, not several: the runner forwards exactly one optional exogenous
    argument. The additional file decides the row set and the calendar is merged
    onto it on `ds`; the reverse inflates the calendar and destroys the `ds`
    uniqueness tiering depends on. The keys are structural; `exog_features` names
    only what rides on them.

    Args:
        panel_df: The trimmed panel, whose keys the coverage check is made against.
        calendar_df: The trimmed fiscal calendar, unique on `ds`.
        additional_exog_df: The trimmed exogenous features file, which drives.
        exog_features: Columns to carry, from either contract, in declaration order.

    Raises:
        ValueError: On a repeated calendar date, a panel and a file that key series
            differently, a duplicate assembled key, a changed panel row count, or a
            coverage gap.

    Returns:
        pd.DataFrame: Columns `unique_id`, `ds`, then `exog_features`.
    """
    _require_unique_calendar_dates(calendar_df)
    _require_comparable_series_ids(panel_df, additional_exog_df)

    # Both frames arrive trimmed to their own contracts, which share no column but
    # ds, so the merged frame is the two contracts side by side and exog_features
    # selects across it without knowing which artifact declares which name.
    exog_df = additional_exog_df.merge(calendar_df, on="ds", how="left")
    exog_df = exog_df[[*JOIN_KEYS, *exog_features]]
    _require_unique_exog_keys(exog_df)

    # The join the model will perform, so its faults are named here. No validate=:
    # the key check above owns that fault and would mask the row-count check below.
    joined = panel_df.merge(exog_df, on=list(JOIN_KEYS), how="left")
    _require_panel_rows_survive(joined, panel_df)
    _require_complete_coverage(joined, exog_features)

    return exog_df
