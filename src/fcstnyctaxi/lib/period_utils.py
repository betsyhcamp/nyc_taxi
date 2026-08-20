from __future__ import annotations

from collections.abc import Callable
from datetime import date

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


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


def derive_horizon_label(
    predicted_fiscal_year_month: pd.Series,
    origin_fiscal_year_month: int | pd.Series,
    origin_month_fraction_elapsed: float | pd.Series,
) -> pd.Series:
    """Label each target month by how many fiscal months it sits ahead of
    the last fully-completed month as of the origin.

    horizon_1 = the current fiscal month (or the next one, if the origin
    sits exactly at month-end, since that month has already fully
    completed); horizon_2 = the month after that. "Last fully completed
    month" is the origin's own month when origin_month_fraction_elapsed
    == 1.0, otherwise the month immediately before it.

    Args:
        predicted_fiscal_year_month: Target fiscal months being forecast,
            one per row. Always the per-row-varying input.
        origin_fiscal_year_month: The forecast origin's own fiscal month.
            Accepts a scalar (single-origin callers, e.g.
            calibrate_n_estimators()) or a Series aligned to
            predicted_fiscal_year_month (multi-origin callers, e.g.
            leaderboard.py, which labels many origins in one call).
        origin_month_fraction_elapsed: Fraction of the origin's fiscal
            month already elapsed as of the origin (1.0 = origin sits at
            month-end). Same scalar-or-Series shape as
            origin_fiscal_year_month.

    Returns:
        Series of "horizon_N" string labels, indexed like
        predicted_fiscal_year_month.
    """
    origin = origin_fiscal_year_month
    frac = origin_month_fraction_elapsed
    target = predicted_fiscal_year_month

    year = origin // 100
    month = origin % 100
    prev_fiscal_year_month = np.where(month == 1, (year - 1) * 100 + 12, origin - 1)

    last_completed = np.where(frac == 1.0, origin, prev_fiscal_year_month)

    month_difference = (target // 100 - last_completed // 100) * 12 + (
        target % 100 - last_completed % 100
    )
    return "horizon_" + pd.Series(month_difference, index=target.index).astype(str)


def label_horizon(monthly_series: pd.DataFrame, calendar_df: pd.DataFrame) -> pd.Series:
    """Label each row of a monthly_series frame with its horizon.

    Maps each origin date to its own fiscal month via the calendar, then delegates
    to derive_horizon_label. Requires only columns attach_tier_and_weight already
    produces.

    The origin unit is deliberately unconstrained. Registered sidecars carry mixed
    units and stay mixed, and pandas matches datetime keys across s/ms/ns. What does
    corrupt a label is an origin absent from the calendar, which the null check below
    catches.

    Args:
        monthly_series: Requires (forecast_origin_date, predicted_fiscal_year_month,
            origin_month_fraction_elapsed).
        calendar_df: Fiscal calendar; requires (ds, fiscal_year_month). Rows that
            repeat a ds with the same fiscal month are collapsed; a ds carrying two
            different fiscal months is an error.

    Returns:
        Series of "horizon_N" labels, indexed like monthly_series.

    Raises:
        ValueError: If either frame lacks a required column, if forecast_origin_date
            is not a datetime dtype, if calendar_df maps one ds to two fiscal months,
            or if any origin is absent from the calendar.
    """
    for frame, name, required in (
        (
            monthly_series,
            "monthly_series",
            [
                "forecast_origin_date",
                "predicted_fiscal_year_month",
                "origin_month_fraction_elapsed",
            ],
        ),
        (calendar_df, "calendar_df", ["ds", "fiscal_year_month"]),
    ):
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}")

    if not is_datetime64_any_dtype(monthly_series["forecast_origin_date"]):
        raise ValueError(
            "monthly_series['forecast_origin_date'] must be a datetime dtype, got "
            f"{monthly_series['forecast_origin_date'].dtype}"
        )

    lookup_df = calendar_df[["ds", "fiscal_year_month"]].drop_duplicates()
    if not lookup_df["ds"].is_unique:
        conflicting = lookup_df.loc[lookup_df["ds"].duplicated(), "ds"].unique()
        raise ValueError(
            f"calendar_df maps {len(conflicting)} ds value(s) to more than one "
            f"fiscal_year_month; first few: {sorted(conflicting)[:5]}"
        )

    origin_fiscal_month = monthly_series["forecast_origin_date"].map(
        lookup_df.set_index("ds")["fiscal_year_month"]
    )
    unmapped = origin_fiscal_month.isna()
    if unmapped.any():
        missing_origins = monthly_series.loc[unmapped, "forecast_origin_date"].unique()
        raise ValueError(
            f"{len(missing_origins)} forecast origin(s) absent from calendar_df; "
            f"first few: {sorted(missing_origins)[:5]}"
        )

    return derive_horizon_label(
        predicted_fiscal_year_month=monthly_series["predicted_fiscal_year_month"],
        origin_fiscal_year_month=origin_fiscal_month,
        origin_month_fraction_elapsed=monthly_series["origin_month_fraction_elapsed"],
    )


def generate_origins_for_periods(
    start_months: list[int],
    forecast_horizon_months: int,
    calendar_df: pd.DataFrame,
    calendar_time_col: str = "ds",
    calendar_period_id: str = "fiscal_year_month",
) -> list[dict]:
    """
        Return deduplicated list of {origin, horizon} dicts to use as forecast origins.

    Args:
        start_months: YYYYMM integers, e.g. [202505, 202506]. Each defines one
            evaluation period starting at that fiscal month.
        forecast_horizon_months: 2 for weekly_to_monthly / direct_monthly;
            1 for remaining_month.
        calendar_df: fiscal calendar DataFrame. Must contain a date column
            (calendar_time_col) and an integer period column (calendar_period_id).
        calendar_time_col: column name in calendar_df holding fiscal week start dates.
            Default "ds".
        calendar_period_id: column name in calendar_df holding YYYYMM period integers.
            Values are coerced to int if stored as strings or floats. Default
            "fiscal_year_month".

    Returns:
        Sorted, deduplicated list of dicts with keys "origin" (ISO date string) and
        "horizon" (int weeks). Each origin date appears exactly once; when the same
        origin belongs to overlapping periods the maximum horizon is kept.
    """
    cal_dates = pd.to_datetime(calendar_df[calendar_time_col])
    period_as_int = calendar_df[calendar_period_id].astype(int)
    cal_months = sorted(period_as_int.unique().tolist())

    def get_weeks_list(month_val):
        return sorted(cal_dates[period_as_int == month_val].dt.date.unique().tolist())

    origin_horizon_map: dict[date, int] = {}
    for month in start_months:
        if month not in cal_months:
            raise ValueError(
                f"start_month {month} not found in calendar — "
                f"available months: {cal_months[0]}–{cal_months[-1]}"
            )
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
        last_week = get_weeks_list(month_end_val)[-1]

        first_origin = get_weeks_list(month_before_val)[-1]
        last_origin = get_weeks_list(month_end_val)[-2]

        mask = (cal_dates.dt.date >= first_origin) & (cal_dates.dt.date <= last_origin)
        all_origins = cal_dates[mask].dt.date.unique().tolist()

        for origin in all_origins:
            horizon = (last_week - origin).days // 7
            origin_horizon_map[origin] = max(origin_horizon_map.get(origin, 0), horizon)

    return [
        {"origin": str(o), "horizon": h} for o, h in sorted(origin_horizon_map.items())
    ]


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
            tier_labels[0] is assigned to series with no positive revenue in the
            trailing window. Default ("very_low", "low", "middle", "high", "very_high").

    Returns:
        DataFrame with columns (id_col, "tier").
        Tier labels assigned lowest (tier_labels[0]) to highest (tier_labels[-1]).
        Series with no positive revenue weeks in the trailing window → tier_labels[0].
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
        return pd.DataFrame({id_col: train_df[id_col].unique(), "tier": tier_labels[0]})

    effective_tiers = min(num_tiers, mean_pos.nunique())
    effective_labels = tier_labels[:effective_tiers]

    return (
        pd.qcut(mean_pos, q=effective_tiers, labels=effective_labels, duplicates="drop")
        .rename("tier")
        .reindex(train_df[id_col].unique())
        .fillna(tier_labels[0])
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
    dampening_fn: Callable[[float], float] = np.cbrt,
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
        dampening_fn: Dampening function applied to the clipped trailing revenue sum.
            Default np.cbrt (cube root). Other candidates: np.sqrt, np.log1p (log(1+x)).

    Returns:
        DataFrame with columns (id_col, "series_weight").
        series_weight = dampening_fn(max(trailing_revenue_sum, 0)).
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
        .apply(dampening_fn)
        .rename("series_weight")
        .reset_index()
    )
