import pandas as pd

from fcstnyctaxi.schemas.run_outputs import JOIN_KEYS


def _require_unique_calendar_dates(calendar_df: pd.DataFrame) -> None:
    """A repeated date duplicates a training row for every series."""
    repeated = calendar_df.loc[calendar_df["ds"].duplicated(), "ds"].unique()
    if len(repeated):
        raise ValueError(f"calendar repeats ds {list(repeated[:5])}.")


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
    *,
    exog_features: tuple[str, ...],
) -> pd.DataFrame:
    """Broadcast the calendar across the panel's series, keyed for the model merge.

    One frame, not several: the runner forwards exactly one optional exogenous
    argument. It spans every calendar date, so it serves both the training merge
    and the horizon a fold predicts into. The keys are structural; `exog_features`
    names only what rides on them.

    Args:
        panel_df: The trimmed panel, supplying the series spine.
        calendar_df: The trimmed fiscal calendar, unique on `ds`.
        exog_features: Calendar columns to carry, in declaration order.

    Raises:
        ValueError: On a repeated calendar date, a duplicate assembled key, a
            changed panel row count, or a coverage gap.

    Returns:
        pd.DataFrame: Columns `unique_id`, `ds`, then `exog_features`.
    """
    _require_unique_calendar_dates(calendar_df)

    spine = panel_df[["unique_id"]].drop_duplicates()
    exog_df = spine.merge(calendar_df[["ds", *exog_features]], how="cross")
    _require_unique_exog_keys(exog_df)

    # The join the model will perform, so its faults are named here. No validate=:
    # the key check above owns that fault and would mask the row-count check below.
    joined = panel_df.merge(exog_df, on=list(JOIN_KEYS), how="left")
    _require_panel_rows_survive(joined, panel_df)
    _require_complete_coverage(joined, exog_features)

    return exog_df
