from typing import cast

import numpy as np
import pandas as pd
from mlforecast import MLForecast


def _require_columns(df: pd.DataFrame, required: list[str], frame_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")


def trim_incomplete_series_months(
    panel_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop (series, month) pairs that lack all weeks_in_month fiscal weeks.

    Framing-specific cleaning for direct-month targets: a full month target and
    weeks_actualized indexing both presume every week of the month is present.

    Args:
        panel_df: Weekly panel with columns (unique_id, ds, y).
        calendar_df: Fiscal calendar with columns (ds, fiscal_year_month,
            fiscal_week_of_month, weeks_in_month).

    Returns:
        (trimmed_panel_df, dropped_pairs). trimmed_panel_df carries panel_df's
        three columns. dropped_pairs has one row per dropped (series, month),
        with columns (unique_id, fiscal_year_month, weeks_present,
        weeks_in_month).

    Raises:
        ValueError: If either frame lacks a required column; if calendar_df does
            not cover every panel date, since an unlabeled row would otherwise
            be dropped silently and never appear in dropped_pairs; or if a
            partial pair survives the trim.
    """
    _require_columns(
        df=panel_df, required=["unique_id", "ds", "y"], frame_name="panel_df"
    )
    _require_columns(
        df=calendar_df,
        required=["ds", "fiscal_year_month", "fiscal_week_of_month", "weeks_in_month"],
        frame_name="calendar_df",
    )

    # precondition: the calendar must cover every panel date
    uncovered = panel_df.loc[~panel_df["ds"].isin(calendar_df["ds"]), "ds"].unique()
    if len(uncovered):
        raise ValueError(
            f"calendar_df does not cover {len(uncovered)} panel date(s); "
            f"first few: {sorted(uncovered)[:5]}"
        )

    labeled = panel_df.merge(
        calendar_df[
            ["ds", "fiscal_year_month", "fiscal_week_of_month", "weeks_in_month"]
        ],
        on="ds",
        how="left",
    )
    weeks_present = labeled.groupby(["unique_id", "fiscal_year_month"])[
        "fiscal_week_of_month"
    ].transform("nunique")
    keep_row_mask = weeks_present == labeled["weeks_in_month"]
    trimmed_panel_df = labeled.loc[keep_row_mask, ["unique_id", "ds", "y"]]
    dropped_pairs = (
        labeled.loc[~keep_row_mask]
        .groupby(["unique_id", "fiscal_year_month"], as_index=False)
        .agg(
            weeks_present=("fiscal_week_of_month", "nunique"),
            weeks_in_month=("weeks_in_month", "first"),
        )
    )

    # check that we have correctly trimmed months that were incomplete
    check = trimmed_panel_df.merge(
        calendar_df[
            ["ds", "fiscal_year_month", "fiscal_week_of_month", "weeks_in_month"]
        ],
        on="ds",
        how="left",
    )
    weeks_present = check.groupby(["unique_id", "fiscal_year_month"])[
        "fiscal_week_of_month"
    ].transform("nunique")
    survivors = check.loc[weeks_present != check["weeks_in_month"]]
    if not survivors.empty:
        raise ValueError("trimmed panel still has partial (series, month) pairs")

    return trimmed_panel_df, dropped_pairs


def build_weekly_features(
    panel: pd.DataFrame,
    freq: str,
    *,
    lags: list[int],
    lag_transforms: dict[int, list],
) -> pd.DataFrame:
    """Build Layer 1 of the modeling table: weekly lag features per series.

    A feature factory, not a model — MLForecast is built with models=[] and
    only preprocess() is called. dropna=False is load-bearing: it keeps each
    series' early weeks, where lags are still NaN, so Layer 3's left join
    finds a row for every origin. Native MLForecast feature names are kept;
    the framing declares the same names in MLF_FEATURES and
    assert_preprocess_feature_drift gates the agreement.

    Args:
        panel: Weekly panel (requires unique_id, ds, y); should be the trimmed panel
            from trim_incomplete_series_months.
        freq: Pandas offset alias for the series frequency, e.g. "W-SUN".
        lags: Lag periods to emit, in units of freq.
        lag_transforms: Lag period -> transforms applied at that lag, e.g.
            {1: [RollingMean(window_size=4)]}.

    Returns:
        DataFrame with (unique_id, feature_ds, y, + one column per native
        MLForecast feature name), one row per panel row. ds is renamed
        feature_ds, the join key Layer 3 uses.
    """
    mlf = MLForecast(models=[], freq=freq, lags=lags, lag_transforms=lag_transforms)
    feats = cast(
        pd.DataFrame, mlf.preprocess(panel, dropna=False)
    )  # unique_id, ds, y, + native feature names ; cast() to satisfy type checking
    return feats.rename(columns={"ds": "feature_ds"})


def enumerate_origins(calendar_df: pd.DataFrame) -> pd.DataFrame:
    """Build the origin spine: one row per forecast origin, at origin grain.

    An origin is a fiscal week end, and it normally targets its own month. The
    exception carries the subtlety: when origin_month_fraction_elapsed == 1 the
    origin's own month has already completed, so it targets the NEXT month with
    weeks_actualized = 0 which is the only value of weeks_actualized the calendar's
    fiscal_week_of_month (which starts at 1) never contains.

    Origins whose target would be the calendar's first month are dropped, since
    no prior month exists to learn from, as is the final week, whose shifted
    target month is NaN.

    Args:
        calendar_df: Fiscal calendar with columns (ds, fiscal_year_month,
            fiscal_week_of_month, weeks_in_month, origin_month_fraction_elapsed).

    Returns:
        DataFrame with (target_month, forecast_origin_date, weeks_actualized,
        weeks_in_month), one row per origin, unique on
        (target_month, weeks_actualized).
    """
    _require_columns(
        df=calendar_df,
        required=[
            "ds",
            "fiscal_year_month",
            "fiscal_week_of_month",
            "weeks_in_month",
            "origin_month_fraction_elapsed",
        ],
        frame_name="calendar_df",
    )
    cal_df = calendar_df.copy().sort_values(by="ds").reset_index(drop=True)
    is_month_end = cal_df["origin_month_fraction_elapsed"] == 1

    cal_df["target_month"] = np.where(
        is_month_end, cal_df["fiscal_year_month"].shift(-1), cal_df["fiscal_year_month"]
    )
    cal_df["weeks_in_month"] = np.where(
        is_month_end, cal_df["weeks_in_month"].shift(-1), cal_df["weeks_in_month"]
    )
    cal_df["weeks_actualized"] = np.where(
        is_month_end, 0, cal_df["fiscal_week_of_month"]
    )

    cols_keep = [
        "target_month",
        "forecast_origin_date",
        "weeks_actualized",
        "weeks_in_month",
    ]
    first_month = cal_df["fiscal_year_month"].min()
    return (
        cal_df[
            (cal_df["target_month"].notna()) & (cal_df["target_month"] != first_month)
        ]
        .rename(columns={"ds": "forecast_origin_date"})
        .astype({"weeks_in_month": int, "target_month": int})[cols_keep]
        .reset_index(drop=True)
    )


def attach_workday_progress(
    origin_spine: pd.DataFrame, calendar_df: pd.DataFrame
) -> pd.DataFrame:
    """Attach number_workdays, workdays_elapsed and workdays_remaining, at origin grain.

    The workday lookup is a cumulative sum over fiscal_week_of_month, which starts
    at 1 so month end origins (weeks_actualized = 0) have nothing to join to.
    An explicit week-0 row per month supplies that value, since zero weeks elapsed
    means zero workdays elapsed. When lookup is joined, a one-to-one validation plus a
    null check bound protect agains fan-out either side of the join.

    Args:
        origin_spine: Output of enumerate_origins; requires (target_month,
            weeks_actualized). Not modified.
        calendar_df: Fiscal calendar with columns (fiscal_year_month,
            fiscal_week_of_month, count_workdays).

    Returns:
        origin_spine's columns plus (number_workdays, workdays_elapsed,
        workdays_remaining), preserving origin_spine's row order.

    Raises:
        ValueError: If either frame lacks a required column, or if any origin has
            no workday match.
    """
    _require_columns(
        df=calendar_df,
        required=["fiscal_year_month", "count_workdays", "fiscal_week_of_month"],
        frame_name="calendar_df",
    )
    _require_columns(
        df=origin_spine,
        required=["target_month", "weeks_actualized"],
        frame_name="origin_spine",
    )

    cal_df = calendar_df.copy().sort_values(
        by=["fiscal_year_month", "fiscal_week_of_month"]
    )
    origin_df = origin_spine.copy()
    # workday lookups
    number_workdays_by_month = cal_df.groupby("fiscal_year_month")[
        "count_workdays"
    ].sum()
    workdays_elapsed_lookup = cal_df.assign(
        workdays_elapsed=cal_df.groupby("fiscal_year_month")["count_workdays"].cumsum()
    ).rename(
        columns={
            "fiscal_year_month": "target_month",
            "fiscal_week_of_month": "weeks_actualized",
        }
    )[["target_month", "weeks_actualized", "workdays_elapsed"]]
    week_zero = pd.DataFrame(
        {"target_month": workdays_elapsed_lookup["target_month"].unique()}
    ).assign(weeks_actualized=0, workdays_elapsed=0)

    workdays_elapsed_lookup = pd.concat(
        [week_zero, workdays_elapsed_lookup], ignore_index=True
    )

    # attach workdays and workday-related quantities
    origin_df["number_workdays"] = origin_df["target_month"].map(
        number_workdays_by_month
    )
    origin_df = origin_df.merge(
        workdays_elapsed_lookup,
        on=["target_month", "weeks_actualized"],
        how="left",
        validate="1:1",
    )
    if origin_df["workdays_elapsed"].isna().any():
        raise ValueError("Error: 'workdays_elapsed' column has NaN rows")
    if origin_df["number_workdays"].isna().any():
        raise ValueError("Error: 'number_workdays' column has NaN rows")

    origin_df["workdays_remaining"] = (
        origin_df["number_workdays"] - origin_df["workdays_elapsed"]
    )
    return origin_df
