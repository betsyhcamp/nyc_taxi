import numpy as np
import pandas as pd


def _build_future_calendar_df(
    unique_ids: np.ndarray,
    last_ds: pd.Timestamp,
    calendar_df: pd.DataFrame,
    horizon: int,
    cal_cols: list[str],
) -> pd.DataFrame:
    """Build one row per (unique_id, ds) for the horizon calendar dates
    immediately following last_ds, carrying cal_cols.

    Args:
        unique_ids: Series identifiers to cross-join against the future dates.
        last_ds: Last observed date in the training panel; future dates are
            strictly after this.
        calendar_df: Fiscal calendar table with a "ds" column and cal_cols.
        horizon: Number of future dates required.
        cal_cols: Calendar columns to attach from calendar_df.

    Returns:
        DataFrame with columns unique_id, ds, and cal_cols. Exactly
        len(unique_ids) * horizon rows.

    Raises:
        ValueError: If calendar_df does not extend horizon dates past last_ds.
    """
    future_dates = (
        calendar_df.loc[calendar_df["ds"] > last_ds, "ds"]
        .sort_values()
        .head(horizon)
        .values
    )

    if len(future_dates) != horizon:
        raise ValueError(
            f"Horizon is {horizon} but not equal to number of "
            f"future_dates is {len(future_dates)} in calendar_df"
        )

    return (
        pd.DataFrame({"unique_id": unique_ids})
        .merge(pd.DataFrame({"ds": future_dates}), how="cross")
        .merge(calendar_df[["ds"] + cal_cols], how="left", on="ds")
    )
