from __future__ import annotations

from collections.abc import Callable
from datetime import date

import numpy as np
import pandas as pd


def _get_trailing_dates(
    calendar_df: pd.DataFrame,
    origin_date: date,
    trailing_weeks: int,
    calendar_time_col: str = "ds",
) -> pd.Series:
    """
    Return the calendar_time_col values for the trailing_weeks fiscal weeks ending
    at origin_date.
    """
    return (
        calendar_df[calendar_df[calendar_time_col] <= pd.Timestamp(origin_date)]
        .sort_values(by=calendar_time_col)
        .tail(trailing_weeks)
        .loc[:, calendar_time_col]
    )


def generate_origins_for_periods(
    start_months: list[int],
    forecast_horizon_months: int,
    calendar_df: pd.DataFrame,
) -> list[str]:
    """
        Return deduplicated list of fiscal_week_start_date strings to use as forecast
        origins.

    Args:
        start_months: YYYYMM integers, e.g. [202505, 202506]. Each defines one
            evaluation period starting at that fiscal month.
        forecast_horizon_months: 2 for weekly_to_monthly / direct_monthly;
            1 for remaining_month.
        calendar_df: fiscal calendar with columns fiscal_year_month (int YYYYMM)
            and fiscal_week_start_date (date).

    Returns:
        Sorted, deduplicated list of ISO date strings (Sundays) representing
        forecast origins. Each date appears exactly once across all periods.
    """

    cal_dates = pd.to_datetime(calendar_df["fiscal_week_start_date"])

    def get_weeks_list(month_val):
        return sorted(
            cal_dates[calendar_df["fiscal_year_month"] == month_val]
            .dt.date.unique()
            .tolist()
        )

    fiscal_week_start_set = set()
    cal_months = sorted(calendar_df["fiscal_year_month"].unique().tolist())
    for month in start_months:
        month_idx = cal_months.index(month)
        if month_idx == 0:
            raise ValueError(
                f"start_month {month} is the first month in the calendar — "
                "no preceding month available for first_origin"
            )

        month_before_idx = month_idx - 1
        month_before_val = cal_months[month_before_idx]
        month_end_idx = month_before_idx + forecast_horizon_months

        if month_end_idx >= len(cal_months):
            raise ValueError(
                f"start_month={month} forecast_horizon_months={forecast_horizon_months}"
                " extends beyond the last month in the calendar"
            )

        month_end_val = cal_months[month_end_idx]

        first_origin = get_weeks_list(month_before_val)[-1]
        last_origin = get_weeks_list(month_end_val)[-2]

        mask = (cal_dates.dt.date >= first_origin) & (cal_dates.dt.date <= last_origin)
        all_origins = cal_dates[mask].dt.date.unique().tolist()
        fiscal_week_start_set.update(all_origins)
    return sorted(str(o) for o in fiscal_week_start_set)


def assign_tiers(
    train_df: pd.DataFrame,
    origin_date: date,
    calendar_df: pd.DataFrame,
    trailing_weeks: int = 52,
    calendar_time_col: str = "ds",
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    num_tiers: int = 5,
    tier_labels: tuple[str, ...] = ("very_low", "low", "middle", "high", "very_high"),
) -> pd.DataFrame:
    """Assign a revenue tier to each series based on trailing training data.

    Args:
        train_df: Per-fold observed data (weeks up to and including origin_date).
        origin_date: The fold's forecast origin (last observed fiscal week start date).
        calendar_df: Fiscal calendar; used to identify the trailing window of weeks.
        trailing_weeks: Number fiscal weeks to look back from origin_date. Default 52.
        calendar_time_col: Date column in calendar_df. Default "ds".
        id_col: Series identifier column. Default "unique_id".
        time_col: Date column in train_df. Default "ds".
        target_col: Revenue column. Default "y".
        num_tiers: Number of quantile bins. Default 5.
        tier_labels: Tier label names ordered lowest to highest. Must match num_tiers.
            Default ("very_low", "low", "middle", "high", "very_high").

    Returns:
        DataFrame with columns (id_col, "tier").
        Tier labels: very_high, high, middle, low, very_low (Q5 → Q1).
        Series with no positive revenue weeks in the trailing window → very_low.
    """
    trailing_dates = _get_trailing_dates(
        calendar_df, origin_date, trailing_weeks, calendar_time_col
    )

    trailing_train_df = train_df[train_df[time_col].isin(trailing_dates)]
    mean_pos = (
        trailing_train_df[trailing_train_df[target_col] > 0]
        .groupby(id_col)[target_col]
        .mean()
    )
    if mean_pos.empty:
        return pd.DataFrame({id_col: train_df[id_col].unique(), "tier": "very_low"})

    effective_tiers = min(num_tiers, mean_pos.nunique())
    effective_labels = tier_labels[:effective_tiers]

    return (
        pd.qcut(mean_pos, q=effective_tiers, labels=effective_labels, duplicates="drop")
        .rename("tier")
        .reindex(train_df[id_col].unique())
        .fillna("very_low")
        .reset_index(drop=False)
    )


def compute_series_weights(
    train_df: pd.DataFrame,
    origin_date: date,
    calendar_df: pd.DataFrame,
    trailing_weeks: int = 26,
    calendar_time_col: str = "ds",
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    weight_fn: Callable[[float], float] = np.sqrt,
) -> pd.DataFrame:
    """Compute dampened revenue weight for each series from trailing training data.

    Args:
        train_df: Per-fold observed data (weeks up to and including origin_date).
        origin_date: The fold's forecast origin (last observed fiscal week start date).
        calendar_df: Fiscal calendar; used to identify the trailing window of weeks.
        trailing_weeks: Number of fiscal weeks to look back from origin_date.
            Default 26.
        calendar_time_col: Date column in calendar_df. Default "ds".
        id_col: Series identifier column. Default "unique_id".
        time_col: Date column. Default "ds".
        target_col: Revenue column. Default "y".
        weight_fn: Dampening function applied to the clipped trailing revenue sum.
            Default np.sqrt. Other candidates: np.cbrt (cube root), np.log1p (log(1+x)).

    Returns:
        DataFrame with columns (id_col, "series_weight").
        series_weight = weight_fn(max(trailing_revenue_sum, 0)).
    """

    trailing_dates = _get_trailing_dates(
        calendar_df, origin_date, trailing_weeks, calendar_time_col
    )

    trailing_sum = (
        train_df[train_df[time_col].isin(trailing_dates)]
        .groupby(id_col)[target_col]
        .sum()
        .reindex(train_df[id_col].unique(), fill_value=0.0)
    )

    return (
        trailing_sum.clip(lower=0)
        .apply(weight_fn)
        .rename("series_weight")
        .reset_index()
    )
