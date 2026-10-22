from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dashboard.shared import palette
from fcstnyctaxi.core.train.evaluate_impl import GLOBAL_TIER

COMPLETE = "complete"
INCOMPLETE = "incomplete"
GAPPED = "gapped"

GLOBAL = GLOBAL_TIER
MONTHLY = "monthly"
WEEKLY = "weekly"
ACTUAL = "actual"
ROLES = ("challenger", "benchmark")
TITLE_ANNOTATION = "line_grid_title"
NOTE_ANNOTATION = "line_grid_note"
NO_FORECAST = "no forecast from this origin for this series"

_LINE_WIDTH = 2.1
_ACTUAL_MARKER = 1.5 * _LINE_WIDTH
_RULE_WIDTH = 1.5
_DASH = {"challenger": "solid", "benchmark": "dash"}
_FILL = {"challenger": "", "benchmark": "-open"}
# Measured: a ring the dot's size sits on the dot's edge, so a coincidence reads
# as one dot. A 12px ring around a bare 8px dot leaves white between them.
_MARKER_SIZE = {"challenger": 8, "benchmark": 12}
_MARKER_STROKE = {"challenger": 0, "benchmark": 2}
_WHEN = {
    MONTHLY: "%{customdata[0]}",
    WEEKLY: "%{customdata[0]}, week of %{x|%Y-%m-%d}",
}
"""How a tooltip names its point at each grain."""
_BOUNDARY_TICK = 0.08
"""The month boundary's tick, as a fraction of the facet's height."""
_NOTE_LINE = 14
"""The header line under each title reserved for notes, in px."""


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


@dataclass(frozen=True)
class GridData:
    """The frames every facet reads, built once per page run."""

    calendar: pd.DataFrame
    weekly: pd.DataFrame
    monthly: pd.DataFrame
    per_series_comparison: pd.DataFrame
    raw_cv_forecasts: dict[str, pd.DataFrame]
    """Keyed by role, as is `models`; the same frame twice when one model holds both."""
    models: dict[str, str]


@dataclass(frozen=True)
class Facet:
    """One facet at one grain: `actual` and `forecast` share `ds` and
    `fiscal_year_month`, and `forecast` holds a column per role."""

    title: str
    grain: str
    origin: pd.Timestamp
    actual: pd.DataFrame
    forecast: pd.DataFrame
    notes: tuple[str, ...]


def build_facet(
    data: GridData, series: object, origin: pd.Timestamp, grain: str
) -> Facet:
    """One series, or `global`, from one origin at one grain.

    A series with no fold at `origin` gets its history and a note, never an
    empty frame, which would read as a broken panel.

    Args:
        data: The page's frames.
        series: A `unique_id`, or `GLOBAL` for the fold set's sum.
        origin: A forecast origin of the run.
        grain: `MONTHLY` or `WEEKLY`.

    Returns:
        Facet: Its frames, title and notes, ready for `add_facet`.
    """
    comparison = data.per_series_comparison
    members = fold_set(comparison, origin) if series == GLOBAL else (series,)
    months = (
        global_monthly(data.monthly, members)
        if series == GLOBAL
        else data.monthly[data.monthly["unique_id"] == series]
    )
    if grain == MONTHLY:
        # An incomplete month carries no total, so it draws no point.
        actual = months
        forecast = forecast_points(comparison, origin, members, data.calendar).rename(
            columns={
                "predicted_fiscal_year_month": "fiscal_year_month",
                "monthly_forecast_ch": "challenger",
                "monthly_forecast_bm": "benchmark",
            }
        )
    else:
        actual = _weekly_sum(data.weekly, members)
        forecast = _weekly_forecast(data, origin, members)
    title = (
        f"{GLOBAL} ({len(members)} series at this origin)"
        if series == GLOBAL
        else str(series)
    )
    return Facet(
        title=title,
        grain=grain,
        origin=origin,
        actual=actual[["ds", "fiscal_year_month", "y"]].reset_index(drop=True),
        forecast=forecast[["ds", "fiscal_year_month", "horizon", *ROLES]]
        .sort_values("ds")
        .reset_index(drop=True),
        notes=_notes(months, forecast, pooled=series == GLOBAL),
    )


def add_facet(
    figure: go.Figure,
    facet: Facet,
    models: dict[str, str],
    *,
    row: int,
    col: int,
    legend: bool,
) -> None:
    """Draw one facet: the actual, then each horizon's challenger and benchmark,
    the open benchmark last so a coincidence reads as a ring around a dot.

    Args:
        figure: A subplot grid holding the facet's cell.
        facet: From `build_facet`.
        models: Each role's model name, for the legend and tooltips.
        row: The facet's row in `figure`.
        col: Its column.
        legend: Whether its traces enter the legend.
    """
    when = _WHEN[facet.grain]
    changes = _month_changes(facet.forecast) if facet.grain == WEEKLY else []
    figure.add_trace(_actual_line(facet.actual, when, legend=legend), row=row, col=col)
    for change in changes:
        for trace in _bridges(*change):
            figure.add_trace(trace, row=row, col=col)
    for horizon in sorted(facet.forecast["horizon"].unique()):
        points = facet.forecast[facet.forecast["horizon"] == horizon]
        for role in ROLES:
            figure.add_trace(
                _forecast_line(
                    points, horizon, role, models[role], when, legend=legend
                ),
                row=row,
                col=col,
            )
    # `opacity` is explicit on both rules, since simple_white draws shapes at 0.3.
    for _, _, boundary in changes:
        figure.add_shape(
            type="line",
            x0=boundary,
            x1=boundary,
            y0=0,
            y1=_BOUNDARY_TICK,
            yref="y domain",
            line={"color": palette.AXIS_COLOR, "width": _RULE_WIDTH},
            opacity=1,
            row=row,
            col=col,
        )
    # Dash-dot, since solid and dashed carry model.
    figure.add_shape(
        type="line",
        x0=facet.origin,
        x1=facet.origin,
        y0=0,
        y1=1,
        yref="y domain",
        line={"color": palette.AXIS_COLOR, "width": _RULE_WIDTH, "dash": "dashdot"},
        opacity=1,
        layer="below",
        row=row,
        col=col,
    )
    # Notes sit on a header line of their own, under the title, so none covers data.
    header = {
        "yref": "y domain",
        "y": 1,
        "yanchor": "bottom",
        "yshift": _NOTE_LINE,
        "showarrow": False,
    }
    figure.add_annotation(
        text=facet.title,
        name=TITLE_ANNOTATION,
        xref="x domain",
        x=0,
        xanchor="left",
        row=row,
        col=col,
        **header,
    )
    figure.add_annotation(
        text=f"origin {facet.origin:%Y-%m-%d}",
        xref="x domain",
        x=1,
        xanchor="right",
        font={"size": 11},
        row=row,
        col=col,
        **header,
    )
    if facet.notes:
        figure.add_annotation(
            text="; ".join(facet.notes),
            name=NOTE_ANNOTATION,
            xref="x domain",
            yref="y domain",
            x=0,
            y=1,
            xanchor="left",
            yanchor="bottom",
            showarrow=False,
            font={"size": 10, "color": palette.AXIS_COLOR},
            row=row,
            col=col,
        )
    figure.update_xaxes(type="date", row=row, col=col)


def _weekly_sum(weekly: pd.DataFrame, members: Sequence) -> pd.DataFrame:
    """The members' weekly total, missing wherever any member's week is. A member
    adds nothing before its first week."""
    rows = weekly[weekly["unique_id"].isin(members)]
    grouped = rows.groupby(["ds", "fiscal_year_month"])["y"]
    total = grouped.sum().where(grouped.count() == grouped.size())
    return total.rename("y").reset_index()


def _weekly_forecast(
    data: GridData, origin: pd.Timestamp, members: Sequence
) -> pd.DataFrame:
    """Each role's weekly forecast from `origin`, summed over `members`, each week
    under the horizon of the month it falls in."""
    comparison = data.per_series_comparison
    at_origin = comparison[comparison["forecast_origin_date"] == origin]
    horizon = (
        at_origin.drop_duplicates("predicted_fiscal_year_month")
        .set_index("predicted_fiscal_year_month")["horizon"]
        .astype(str)
    )
    sums = {}
    for role in ROLES:
        raw = data.raw_cv_forecasts[role]
        rows = raw[
            (raw["forecast_origin_date"] == origin) & raw["unique_id"].isin(members)
        ]
        sums[role] = rows.groupby("ds")["ypred"].sum()
    frame = pd.DataFrame(sums).rename_axis("ds").reset_index()
    month = frame["ds"].map(data.calendar.set_index("ds")["fiscal_year_month"])
    return frame.assign(fiscal_year_month=month, horizon=month.map(horizon))


def _notes(
    months: pd.DataFrame, forecast: pd.DataFrame, *, pooled: bool
) -> tuple[str, ...]:
    """What the facet cannot show: a forecast it has none of, and gapped months,
    split by whether a forecast was scored against one."""
    window = set(forecast["fiscal_year_month"])
    gapped = sorted(months.loc[months["status"] == GAPPED, "fiscal_year_month"])
    history = [str(month) for month in gapped if month not in window]
    scored = [str(month) for month in gapped if month in window]
    parts = [", ".join(history)] if history else []
    if scored:
        parts.append(f"{', '.join(scored)} scored against a short total, so not drawn")
    missing = "a member is missing a week" if pooled else "missing a week"
    notes = [] if len(forecast) else [NO_FORECAST]
    if parts:
        notes.append(f"{missing}: {'; '.join(parts)}")
    return tuple(notes)


def _month_changes(
    forecast: pd.DataFrame,
) -> list[tuple[pd.Series, pd.Series, pd.Timestamp]]:
    """Each pair of consecutive weekly forecast rows in different months, with the
    boundary midway between them."""
    rows = [row for _, row in forecast.iterrows()]
    return [
        (before, after, before["ds"] + (after["ds"] - before["ds"]) / 2)
        for before, after in pairwise(rows)
        if before["fiscal_year_month"] != after["fiscal_year_month"]
    ]


def _bridges(
    before: pd.Series, after: pd.Series, boundary: pd.Timestamp
) -> list[go.Scatter]:
    """Each role's line across a month boundary, each half in its own month's
    horizon color, so the color changes at the boundary itself."""
    bridges = []
    for role in ROLES:
        middle = (before[role] + after[role]) / 2
        for side in (before, after):
            color = palette.rgba(palette.HORIZON_COLOR[side["horizon"]])
            bridges.append(
                go.Scatter(
                    x=[side["ds"], boundary],
                    y=[side[role], middle],
                    mode="lines",
                    line={"color": color, "width": _LINE_WIDTH, "dash": _DASH[role]},
                    legendgroup=f"{role} {side['horizon']}",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
    return bridges


def _actual_line(actual: pd.DataFrame, when: str, *, legend: bool) -> go.Scatter:
    """The actual, its marker small enough that dense weeks merge into the line."""
    color = palette.rgba(palette.ACTUAL_INK)
    return go.Scatter(
        x=actual["ds"],
        y=actual["y"],
        mode="lines+markers",
        name=ACTUAL,
        legendgroup=ACTUAL,
        showlegend=legend,
        line={"color": color, "width": _LINE_WIDTH},
        marker={"size": _ACTUAL_MARKER, "color": color},
        customdata=actual[["fiscal_year_month"]],
        hovertemplate=f"actual<br>{when}<br>%{{y:,.0f}}<extra></extra>",
    )


def _forecast_line(
    points: pd.DataFrame,
    horizon: str,
    role: str,
    model: str,
    when: str,
    *,
    legend: bool,
) -> go.Scatter:
    """One role's forecast for one horizon: color and shape carry the horizon,
    dash and fill the role."""
    color = palette.rgba(palette.HORIZON_COLOR[horizon])
    return go.Scatter(
        x=points["ds"],
        y=points[role],
        mode="lines+markers",
        name=f"{model} {horizon}",
        legendgroup=f"{role} {horizon}",
        showlegend=legend,
        line={"color": color, "width": _LINE_WIDTH, "dash": _DASH[role]},
        marker={
            "symbol": f"{palette.HORIZON_MARKER[horizon]}{_FILL[role]}",
            "size": _MARKER_SIZE[role],
            "color": color,
            "line": {"color": color, "width": _MARKER_STROKE[role]},
        },
        customdata=points[["fiscal_year_month"]],
        hovertemplate=f"{model} {horizon}<br>{when}<br>%{{y:,.0f}}<extra></extra>",
    )
