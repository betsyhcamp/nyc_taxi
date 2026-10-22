from collections.abc import Sequence

import numpy as np
import pandas as pd

COMPLETE = "complete"
INCOMPLETE = "incomplete"
GAPPED = "gapped"


def reindex_weeks(snapshot: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Each series on every calendar week from its first non-null week to the
    panel's last, so an absent row becomes a NaN week, as a null `y` already is.

    Without it an absent row leaves no point to break a line at, and Plotly draws
    straight across it.
    """
    first = (
        snapshot[snapshot["y"].notna()]
        .groupby("unique_id", as_index=False)["ds"]
        .min()
        .rename(columns={"ds": "first_week"})
    )
    weeks = calendar.loc[
        calendar["ds"] <= snapshot["ds"].max(), ["ds", "fiscal_year_month"]
    ]
    # A cross join, cut at each series' first week: the reindex has to create the
    # absent weeks, which an inner join to the actuals cannot.
    spans = first.merge(weeks, how="cross")
    spans = spans[spans["ds"] >= spans["first_week"]]
    return spans.merge(snapshot, on=["unique_id", "ds"], how="left")[
        ["unique_id", "ds", "fiscal_year_month", "y"]
    ]


def monthly_totals(weekly: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Each series' months, marked complete, incomplete or gapped, a complete one
    with its total at the month's last fiscal week.

    Incomplete: some of its weeks fall outside the series' span, before its first
    week or after the panel's last. Gapped: every week inside, one without a value.
    Only a complete month carries a total, since a partial sum beside the forecasts
    would read as revenue ramping up or collapsing.

    Args:
        weekly: From `reindex_weeks`.
        calendar: The sidecar's fiscal calendar.

    Returns:
        pd.DataFrame: `unique_id`, `fiscal_year_month`, `ds`, `y`, `status`.
    """
    months = calendar.groupby("fiscal_year_month").agg(
        ds=("ds", "max"), weeks=("weeks_in_month", "first")
    )
    totals = (
        weekly.groupby(["unique_id", "fiscal_year_month"])["y"]
        .agg(in_span="size", observed="count", y="sum")
        .reset_index()
        .join(months, on="fiscal_year_month")
    )
    status = np.select(
        [totals["in_span"] < totals["weeks"], totals["observed"] < totals["in_span"]],
        [INCOMPLETE, GAPPED],
        COMPLETE,
    )
    return totals.assign(status=status, y=totals["y"].where(status == COMPLETE))[
        ["unique_id", "fiscal_year_month", "ds", "y", "status"]
    ]


def fold_set(per_series_comparison: pd.DataFrame, origin: pd.Timestamp) -> tuple:
    """The series with a fold at `origin`, the members `global` sums there.

    Not the run's roster, whose actual would include series the forecast does not,
    so `global` alone would plot an actual other than the one it was scored against.
    """
    at_origin = per_series_comparison[
        per_series_comparison["forecast_origin_date"] == origin
    ]
    return tuple(sorted(at_origin["unique_id"].unique()))


def global_monthly(monthly: pd.DataFrame, members: Sequence) -> pd.DataFrame:
    """`global`'s months: the sum of the members whose own month is complete,
    gapped when any member's is.

    A member's incomplete month is left out, as its own facet leaves it out, so a
    series starting mid-month adds nothing until its first whole month. A month
    no member has whole is incomplete.

    Args:
        monthly: From `monthly_totals`.
        members: From `fold_set`.

    Returns:
        pd.DataFrame: `fiscal_year_month`, `ds`, `y`, `status`.
    """
    rows = monthly[monthly["unique_id"].isin(members)]
    by_month = (
        rows.assign(gapped=rows["status"] == GAPPED, whole=rows["status"] == COMPLETE)
        .groupby("fiscal_year_month")
        .agg(
            ds=("ds", "first"),
            y=("y", "sum"),
            gapped=("gapped", "any"),
            whole=("whole", "any"),
        )
        .reset_index()
    )
    status = np.select(
        [by_month["gapped"], ~by_month["whole"]], [GAPPED, INCOMPLETE], COMPLETE
    )
    return by_month.assign(status=status, y=by_month["y"].where(status == COMPLETE))[
        ["fiscal_year_month", "ds", "y", "status"]
    ]


def forecast_points(
    per_series_comparison: pd.DataFrame,
    origin: pd.Timestamp,
    members: Sequence,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    """Both models' monthly forecasts from `origin`, summed over `members`, each at
    its predicted month's last fiscal week."""
    rows = per_series_comparison[
        (per_series_comparison["forecast_origin_date"] == origin)
        & per_series_comparison["unique_id"].isin(members)
    ]
    totals = (
        rows.assign(horizon=rows["horizon"].astype(str))
        .groupby(["predicted_fiscal_year_month", "horizon"], as_index=False)[
            ["monthly_forecast_ch", "monthly_forecast_bm"]
        ]
        .sum()
    )
    last_week = calendar.groupby("fiscal_year_month")["ds"].max()
    return totals.assign(ds=totals["predicted_fiscal_year_month"].map(last_week))
