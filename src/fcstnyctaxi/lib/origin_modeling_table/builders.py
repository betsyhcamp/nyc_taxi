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
    """Build the weekly lag features the modeling table is joined against.

    Uses MLForecast as a feature factory, not a model: it is built with models=[] and
    only preprocess() is called. dropna=False is load-bearing, since it keeps each
    series' early weeks where the lags are still NaN, so attach_weekly_features finds
    a row for every origin. Native MLForecast feature names are kept; the framing
    declares the same names in MLF_FEATURES.

    Args:
        panel: Weekly panel (requires unique_id, ds, y); should be the trimmed panel,
            so every month a target is built from is complete.
        freq: Pandas offset alias for the series frequency, e.g. "W-SUN".
        lags: Lag periods to emit, in units of freq.
        lag_transforms: Lag period -> transforms applied at that lag, e.g.
            {1: [RollingMean(window_size=4)]}.

    Returns:
        DataFrame with (unique_id, feature_row_ds, y, + one column per native
        MLForecast feature name), one row per panel row. ds is renamed
        feature_row_ds, as a join key for future use.
    """
    mlf = MLForecast(models=[], freq=freq, lags=lags, lag_transforms=lag_transforms)
    feats = cast(
        pd.DataFrame, mlf.preprocess(panel, dropna=False)
    )  # unique_id, ds, y, + native feature names ; cast() to satisfy type checking
    return feats.rename(columns={"ds": "feature_row_ds"})


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


def build_origin_series_grid(
    origin_spine: pd.DataFrame, actual_monthly_df: pd.DataFrame
) -> pd.DataFrame:
    """Build the (series x origin) grid and attach actual_monthly_total.

    The inner join is foundational since the grid is built from realized actuals rather
    than by cross joining every series against every month. As a result, a series
    appears only from its first active month and no prehistory rows are manufactured.
    A (series, month) absent here is legitimately absent downstream, which is why
    coverage gates must be scoped to active pairs.

    The merge is a deliberate fan out so every origin in a month meets every active
    series in that month so validate= cannot constrain it; the result is checked
    for uniqueness instead. The column actual_monthly_total keeps its name, and a
    framing renames it if its own target is called something else.

    Args:
        origin_spine: Output of enumerate_origins, optionally with workday progress
            attached; requires (target_month, forecast_origin_date).
        actual_monthly_df: Output of compute_actual_monthly_totals; requires
            (unique_id, fiscal_year_month, actual_monthly_total).

    Returns:
        origin_spine's columns plus (unique_id, actual_monthly_total), one row per
        active (series, origin).

    Raises:
        ValueError: If either frame lacks a required column, or if the grid is not
            unique on (unique_id, forecast_origin_date, target_month).
    """
    _require_columns(
        df=origin_spine,
        required=["target_month", "forecast_origin_date"],
        frame_name="origin_spine",
    )
    _require_columns(
        df=actual_monthly_df,
        required=["unique_id", "fiscal_year_month", "actual_monthly_total"],
        frame_name="actual_monthly_df",
    )

    # deliberate fan out so each unique_id get calendar attributes from origin_spine
    grid = origin_spine.merge(
        actual_monthly_df.rename(columns={"fiscal_year_month": "target_month"}),
        on="target_month",
        how="inner",
    )

    # check for duplicate rows in grid
    grid_keys = ["unique_id", "forecast_origin_date", "target_month"]
    duplicates = grid.duplicated(grid_keys)
    if duplicates.any():
        raise ValueError(
            f"{int(duplicates.sum())} duplicate rows on {grid_keys}; "
            "origin_spine or actual_monthly_df has repeated keys"
        )
    return grid


def attach_mtd_revenue(
    grid_df: pd.DataFrame, panel_df: pd.DataFrame, calendar_df: pd.DataFrame
) -> pd.DataFrame:
    """Attach per-series month-to-date revenue at each origin.

    mtd_revenue is a cumulative sum of y within (unique_id, fiscal_year_month),
    keyed to weeks_actualized. The lookup is built from fiscal_week_of_month, which
    starts at 1, so month end origins (weeks_actualized = 0) have nothing to join
    to. The explicit week zero row per active (series, month) supplies that value,
    since no week observed means zero month-to-date. Sourced from the lookup, never
    from the grid so that a (series, month) genuinely absent from the panel must stay
    absent rather than acquire a manufactured zero.

    Args:
        grid_df: Output of build_origin_series_grid; requires (unique_id,
            target_month, weeks_actualized).
        panel_df: The same trimmed panel the grid was built from; requires
            (unique_id, ds, y).
        calendar_df: Fiscal calendar with columns (ds, fiscal_year_month,
            fiscal_week_of_month).

    Returns:
        grid's columns plus mtd_revenue, in grid's row order.

    Raises:
        ValueError: If any frame lacks a required column, or if a grid row has no
            MTD match.
    """
    _require_columns(
        df=grid_df,
        required=["unique_id", "target_month", "weeks_actualized"],
        frame_name="grid_df",
    )
    _require_columns(
        df=panel_df, required=["unique_id", "ds", "y"], frame_name="panel_df"
    )
    _require_columns(
        df=calendar_df,
        required=["ds", "fiscal_year_month", "fiscal_week_of_month"],
        frame_name="calendar_df",
    )

    panel_cal = panel_df.merge(
        calendar_df[["ds", "fiscal_year_month", "fiscal_week_of_month"]],
        on="ds",
        how="left",
    ).sort_values(["unique_id", "fiscal_year_month", "fiscal_week_of_month"])

    panel_cal["mtd_revenue"] = panel_cal.groupby(["unique_id", "fiscal_year_month"])[
        "y"
    ].cumsum()

    mtd_lookup = panel_cal.rename(
        columns={
            "fiscal_year_month": "target_month",
            "fiscal_week_of_month": "weeks_actualized",
        }
    )[["unique_id", "target_month", "weeks_actualized", "mtd_revenue"]]

    week_zero = (
        mtd_lookup[["unique_id", "target_month"]]
        .drop_duplicates()
        .assign(weeks_actualized=0, mtd_revenue=0)
    )
    mtd_lookup = pd.concat([week_zero, mtd_lookup], ignore_index=True)

    grid_df = grid_df.merge(
        mtd_lookup,
        on=["unique_id", "target_month", "weeks_actualized"],
        how="left",
        validate="one_to_one",
    )
    unmatched = grid_df["mtd_revenue"].isna()
    if unmatched.any():
        raise ValueError(f"{int(unmatched.sum())} grid row(s) have no MTD match")

    grid_df["mtd_revenue"] = grid_df["mtd_revenue"].astype(float)
    return grid_df


def attach_weekly_features(
    origin_target_table: pd.DataFrame, weekly_features: pd.DataFrame
) -> pd.DataFrame:
    """Join weekly lag features onto each (series, origin) row, keyed one week ahead.

    The join key has a nuance. forecast_origin_date = W is the last week
    actually observed, so an origin's features must reflect "observed through W" —
    but MLForecast emits lag1 at ds as y[ds - 1 week], so the row where
    lag1 == y[W] is the row at ds = W + 1 week. Hence feature_row_ds = W + 1.

    Returns the full joined superset and selects nothing; bounding the model
    matrix is ModelingTableSchema's job.

    y is required so the drop below is reachable rather than a silent no-op: the
    fetched row's y is y[W + 1], a week that has not happened at the origin, so it
    is never a legitimate feature.

    The one-to-one validation encodes a precondition: each origin targets exactly
    one month. True for both current-month framings, false for a multi-horizon
    one — two target rows would share a feature row, and the merge would need
    many_to_one instead.

    Args:
        origin_target_table: One row per (series, origin); requires (unique_id,
            forecast_origin_date). Not modified.
        weekly_features: Output of build_weekly_features; requires (unique_id,
            feature_row_ds, y).

    Returns:
        origin_target_table's columns plus feature_row_ds and every emitted feature
        name, in origin_target_table's row order.

    Raises:
        ValueError: If either frame lacks a required column.
        MergeError: If either frame repeats an (unique_id, feature_row_ds) key.
    """
    _require_columns(
        df=origin_target_table,
        required=["unique_id", "forecast_origin_date"],
        frame_name="origin_target_table",
    )
    _require_columns(
        df=weekly_features,
        required=["unique_id", "feature_row_ds", "y"],
        frame_name="weekly_features",
    )
    keyed_origins = origin_target_table.copy()
    keyed_origins["feature_row_ds"] = keyed_origins[
        "forecast_origin_date"
    ] + pd.Timedelta(weeks=1)
    modeling = keyed_origins.merge(
        weekly_features,
        on=["unique_id", "feature_row_ds"],
        how="left",
        validate="one_to_one",
    )
    modeling = modeling.drop(columns="y")
    return modeling
