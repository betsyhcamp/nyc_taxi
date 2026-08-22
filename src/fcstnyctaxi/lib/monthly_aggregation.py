from typing import cast

import pandas as pd

from fcstnyctaxi.lib.period_utils import assign_tiers, compute_series_weights


def compute_actual_monthly_totals(
    ts_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    period_col: str = "fiscal_year_month",
    time_col: str = "ds",
    id_col: str = "unique_id",
    target_col: str = "y",
) -> pd.DataFrame:
    """Sum realized weekly actuals across the full history per (id_col, period_col).

    Unlike compute_mtd_actuals, which filters to specific target fiscal months
    for one fold, this sums the entire ts_df which is the full realized monthly ground
    truth, computed once and reused across every fold/origin as
    actual_monthly_df.

    Args:
        ts_df: Full historical panel.
        calendar_df: Maps time_col (ds) to period_col (fiscal_year_month).
        period_col: Fiscal period column. Default "fiscal_year_month".
        time_col: Join key between ts_df and calendar_df. Default "ds".
        id_col: Series identifier column. Default "unique_id".
        target_col: Target value column to sum. Default "y".

    Returns:
        DataFrame with columns (id_col, period_col, "actual_monthly_total").
    """

    return (
        ts_df.merge(
            calendar_df[[time_col, period_col]].drop_duplicates(),
            on=time_col,
            how="left",
        )
        .groupby([id_col, period_col])[target_col]
        .sum()
        .reset_index()
        .rename(columns={target_col: "actual_monthly_total"})
    )


def compute_predicted_remaining(
    forecast_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    period_col: str = "fiscal_year_month",
    time_col: str = "ds",
    id_col: str = "unique_id",
    forecast_col: str = "ypred",
) -> pd.DataFrame:
    """Sum forecasted weekly values within target fiscal months per
    (id_col, period_col).

    Args:
        forecast_df: Raw weekly forecast for this fold's horizon.
        calendar_df: Maps time_col (ds) to period_col (fiscal_year_month).
        period_col: Fiscal period column. Default "fiscal_year_month".
        time_col: Join key between forecast_df and calendar_df. Default "ds".
        id_col: Series identifier column. Default "unique_id".
        forecast_col: Predicted value column to sum. Default "ypred".

    Returns:
        DataFrame with columns (id_col, period_col, forecast_col).
    """
    return (
        forecast_df.merge(
            calendar_df[[time_col, period_col]].drop_duplicates(),
            on=time_col,
            how="left",
        )
        .groupby([id_col, period_col])[forecast_col]
        .sum()
        .reset_index()
    )


def compute_mtd_actuals(
    train_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    target_fiscal_months: list,
    period_col: str = "fiscal_year_month",
    time_col: str = "ds",
    id_col: str = "unique_id",
    target_col: str = "y",
) -> pd.DataFrame:
    """Sum observed weekly actuals within target fiscal months per (id_col, period_col).

    Args:
        train_df: Per-fold observed data (weeks up to the fold origin).
        calendar_df: Maps time_col (ds) to period_col (fiscal_year_month).
        target_fiscal_months: Fiscal months covered by the forecast horizon.
        period_col: Fiscal period column. Default fiscal_year_month.
        time_col: Join key between train_df and calendar_df. Default "ds".
        id_col: Series identifier column. Default "unique_id".
        target_col: Target value column to sum. Default "y".

    Returns:
        DataFrame with columns (id_col, period_col, "mtd_actuals").
        Empty when no target-month weeks have been observed yet (e.g. fold_0).
    """

    target_cal = calendar_df.loc[
        calendar_df[period_col].isin(target_fiscal_months), [time_col, period_col]
    ].drop_duplicates()

    return (
        train_df.merge(target_cal, on=time_col, how="inner")
        .groupby([id_col, period_col])[target_col]
        .sum()
        .reset_index()
        .rename(columns={target_col: "mtd_actuals"})
    )


def combine_monthly_forecast(
    mtd_actuals_df: pd.DataFrame,
    predicted_remaining_df: pd.DataFrame,
    period_col: str = "fiscal_year_month",
    forecast_col: str = "ypred",
    id_col: str = "unique_id",
    mtd_actuals_col: str = "mtd_actuals",
) -> pd.DataFrame:
    """Add MTD actuals and predicted remaining to produce a total monthly forecast.

    Args:
        mtd_actuals_df: Output of compute_mtd_actuals; columns
            (id_col, period_col, mtd_actuals_col).
        predicted_remaining_df: Aggregated fold forecasts; columns
            (id_col, period_col, forecast_col).
        period_col: Fiscal period column. Default "fiscal_year_month".
        forecast_col: Predicted remaining column in predicted_remaining_df. Default
            "ypred".
        mtd_actuals_col: MTD actuals column in mtd_actuals_df. Default "mtd_actuals".
        id_col: Series identifier column. Default "unique_id".

    Returns:
        DataFrame with columns (id_col, period_col, "mtd_revenue",
        "predicted_remaining", "monthly_forecast"). The add back components are renamed
        from mtd_actuals_col/forecast_col to the canonical names shared across
        framings; the inputs keep their own column names.

        mtd_revenue is the post-fillna value, not mtd_actuals_df's. The outer
        join with fillna(0) materializes 0 for any (series, period) with no
        observed weeks yet — forecast origins sitting before the target month
        begins, and every next-month row — which compute_mtd_actuals omits
        entirely. Those rows must carry 0, not be absent.
    """

    merged_df = (
        mtd_actuals_df[[id_col, period_col, mtd_actuals_col]]
        .merge(
            predicted_remaining_df[[id_col, period_col, forecast_col]],
            on=[id_col, period_col],
            how="outer",
        )
        .fillna(0)
    )

    merged_df = merged_df.assign(
        monthly_forecast=merged_df[mtd_actuals_col] + merged_df[forecast_col]
    )
    merged_df = merged_df.rename(
        columns={mtd_actuals_col: "mtd_revenue", forecast_col: "predicted_remaining"}
    )
    return merged_df[
        [id_col, period_col, "mtd_revenue", "predicted_remaining", "monthly_forecast"]
    ]


def build_monthly_forecast_vs_actual(
    forecast_df: pd.DataFrame,
    train_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    actual_monthly_df: pd.DataFrame,
    period_col: str = "fiscal_year_month",
    time_col: str = "ds",
    id_col: str = "unique_id",
    forecast_col: str = "ypred",
    target_col: str = "y",
) -> pd.DataFrame:
    """Build the monthly forecast-vs-actual comparison pair for one fold.

    Combines already-observed month-to-date actuals (compute_mtd_actuals)
    with the model's forecast for the remaining weeks of each target fiscal
    month (compute_predicted_remaining) into monthly_forecast, then merges
    in the true realized actual_monthly_total. No tier or weight enrichment
    — see attach_tier_and_weight for that.

    Args:
        forecast_df: Raw weekly forecast for this fold's horizon.
        train_df: Per-fold observed data (weeks up to the fold origin).
        calendar_df: Maps time_col (ds) to period_col (fiscal_year_month).
        actual_monthly_df: Realized monthly totals; columns (id_col,
            period_col, "actual_monthly_total"), precomputed once outside
            the fold loop.
        period_col: Fiscal period column. Default "fiscal_year_month".
        time_col: Join key between forecast_df/train_df and calendar_df.
            Default "ds".
        id_col: Series identifier column. Default "unique_id".
        forecast_col: Predicted value column in forecast_df. Default
            "ypred".
        target_col: Observed actual value column in train_df. Default "y".

    Returns:
        DataFrame with columns (id_col, period_col, "mtd_revenue",
        "predicted_remaining", "monthly_forecast", "actual_monthly_total").
        The two add back components pass through from combine_monthly_forecast so
        callers can persist the decomposition alongside its sum; monthly_forecast is
        mtd_revenue + predicted_remaining on every row.
    """
    predicted_remaining_df = compute_predicted_remaining(
        forecast_df=forecast_df,
        calendar_df=calendar_df,
        period_col=period_col,
        time_col=time_col,
        id_col=id_col,
        forecast_col=forecast_col,
    )

    target_fiscal_months = predicted_remaining_df[period_col].unique().tolist()

    mtd_actuals_df = compute_mtd_actuals(
        train_df=train_df,
        calendar_df=calendar_df,
        target_fiscal_months=target_fiscal_months,
        period_col=period_col,
        time_col=time_col,
        id_col=id_col,
        target_col=target_col,
    )

    monthly_forecast_total_df = combine_monthly_forecast(
        mtd_actuals_df=mtd_actuals_df,
        predicted_remaining_df=predicted_remaining_df,
        period_col=period_col,
        forecast_col=forecast_col,
        id_col=id_col,
    )

    return monthly_forecast_total_df.merge(actual_monthly_df, on=[id_col, period_col])


def attach_tier_and_weight(
    monthly_rows_df: pd.DataFrame,
    tier_df: pd.DataFrame,
    weight_df: pd.DataFrame,
    fold_origin: pd.Timestamp | int,
    origin_month_fraction_elapsed: float,
    period_col: str = "fiscal_year_month",
    id_col: str = "unique_id",
) -> pd.DataFrame:
    """Attach tier and series-weight metadata to produce the final
    leaderboard-ready row.

    Merges in per-series tier and weight (both computed once per fold,
    upstream of this call), then attaches fold-level context
    (forecast_origin_date, origin_month_fraction_elapsed) and renames
    period_col to its final leaderboard column name. Not called by
    calibrate_n_estimators(), which has no use for tier (its scoring splits
    by horizon only, not tier) and recomputes weight independently for
    scoring rather than reading it off this output.

    Args:
        monthly_rows_df: Output of build_monthly_forecast_vs_actual. Requires
            (id_col, period_col, "monthly_forecast", "actual_monthly_total");
            its "mtd_revenue"/"predicted_remaining" columns are unused here and
            are dropped by the final select.
        tier_df: Output of assign_tiers; columns (id_col, "tier").
        weight_df: Output of compute_series_weights; columns
            (id_col, "series_weight").
        fold_origin: This fold's forecast origin.
        origin_month_fraction_elapsed: Fraction of the origin's fiscal
            month elapsed as of fold_origin.
        period_col: Fiscal period column in monthly_rows_df. Default
            "fiscal_year_month".
        id_col: Series identifier column. Default "unique_id".

    Returns:
        DataFrame with columns (forecast_origin_date,
        predicted_fiscal_year_month, id_col, "tier", "monthly_forecast",
        "actual_monthly_total", "series_weight",
        origin_month_fraction_elapsed).
    """

    return (
        monthly_rows_df.merge(tier_df, on=id_col)
        .merge(weight_df, on=id_col)
        .assign(
            forecast_origin_date=fold_origin,
            origin_month_fraction_elapsed=origin_month_fraction_elapsed,
        )
        .rename(
            columns={
                period_col: "predicted_fiscal_year_month",
            }
        )[
            [
                "forecast_origin_date",
                "predicted_fiscal_year_month",
                id_col,
                "tier",
                "monthly_forecast",
                "actual_monthly_total",
                "series_weight",
                "origin_month_fraction_elapsed",
            ]
        ]
    )


def build_monthly_series(
    forecasts_df: pd.DataFrame,
    panel_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble fold predictions into the leaderboard-ready sidecar frame.

    Loops per origin because tiers and series weights are computed from trailing
    data as of each one. A single call over all folds would score late origins
    with information the early ones did not have. Extracted into reusable func
    for parity use cases. period_col is "target_month" attach_tier_and_weight only
    defaults period_col = "fiscal_year_month".

    Args:
        forecasts_df: Fold output. Requires (unique_id, forecast_origin_date,
            target_month, monthly_forecast, actual_monthly_total). The value
            columns arrive already renamed, since what a framing calls its model
            output is its own detail.
        panel_df: The TRIMMED weekly panel which is the same one the modeling table
            was built from. assign_tiers and compute_series_weights derive trailing
            windows from it, so an untrimmed panel yields different tiers and breaks
            benchmark parity. Unverifiable from the frame itself, hence stated. Requires
            (unique_id, ds, y)
        calendar_df: Fiscal calendar; requires (ds, origin_month_fraction_elapsed).

    Returns:
        One row per (origin, series) with columns (forecast_origin_date,
        predicted_fiscal_year_month, unique_id, tier, monthly_forecast,
        actual_monthly_total, series_weight, origin_month_fraction_elapsed).

    Raises:
        ValueError: If either frame lacks a required column.
    """
    required = [
        "unique_id",
        "forecast_origin_date",
        "target_month",
        "monthly_forecast",
        "actual_monthly_total",
    ]
    missing = [c for c in required if c not in forecasts_df.columns]
    if missing:
        raise ValueError(f"forecasts_df is missing required columns: {missing}")

    missing = [c for c in ("unique_id", "ds", "y") if c not in panel_df.columns]
    if missing:
        raise ValueError(f"panel is missing required columns: {missing}")

    missing = [
        c
        for c in ("ds", "origin_month_fraction_elapsed")
        if c not in calendar_df.columns
    ]
    if missing:
        raise ValueError(f"calendar_df is missing required columns: {missing}")

    # assemble forecasts w/ tier, weight; items needed later for scoring of backtests
    fraction_by_origin = calendar_df.set_index("ds")["origin_month_fraction_elapsed"]
    rows = []
    for key_date, origin_group in forecasts_df.groupby("forecast_origin_date"):
        origin_date = cast(pd.Timestamp, key_date)
        monthly_rows_df = origin_group[
            ["unique_id", "target_month", "monthly_forecast", "actual_monthly_total"]
        ]
        tier_df = assign_tiers(
            train_df=panel_df, origin_date=origin_date, calendar_df=calendar_df
        )

        weight_df = compute_series_weights(
            train_df=panel_df, origin_date=origin_date, calendar_df=calendar_df
        )
        origin_group_built = attach_tier_and_weight(
            monthly_rows_df=monthly_rows_df,
            tier_df=tier_df,
            weight_df=weight_df,
            fold_origin=origin_date,
            origin_month_fraction_elapsed=fraction_by_origin[origin_date],
            period_col="target_month",
        )
        rows.append(origin_group_built)
    return pd.concat(rows, ignore_index=True)
