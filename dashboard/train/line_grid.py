from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard.shared import axes, palette
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

_KEY = "line_grid"
GRIDS = 2
ROWS, COLS = 3, 2
GRAINS = (MONTHLY, WEEKLY)
Y_SCALES = ("independent", "shared")
WINDOWS = ("full history", "backtest window")
EXPANDER = "Time series line grids"
_HISTORY = pd.Timedelta(weeks=52)
"""How far before the run's first origin the backtest window starts."""
_FLAT_HEADROOM = 1.1
"""A facet of one repeated value runs from 0 to this multiple of it."""
_FACET_HEIGHT = 200
"""Plot area per facet, in px."""
_ROW_GAP = 64
"""Room between rows for the next row's two header lines; ticks are last-row only."""
_COLUMN_GAP = 0.06
_PAD = 0.04
"""Fraction of a span added at each end of a computed range."""


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
) -> None:
    """Draw one facet: the actual, then each horizon's challenger and benchmark,
    the open benchmark last so a coincidence reads as a ring around a dot.

    Args:
        figure: A subplot grid holding the facet's cell.
        facet: From `build_facet`.
        models: Each role's model name, for the legend and tooltips.
        row: The facet's row in `figure`.
        col: Its column.
    """
    when = _WHEN[facet.grain]
    changes = _month_changes(facet.forecast) if facet.grain == WEEKLY else []
    # One model in both roles would name both lines alike, and the legend keeps one.
    shared = models["challenger"] == models["benchmark"]
    labels = {
        role: f"{models[role]} ({role})" if shared else models[role] for role in ROLES
    }
    figure.add_trace(_actual_line(facet.actual, when), row=row, col=col)
    for change in changes:
        for trace in _bridges(*change):
            figure.add_trace(trace, row=row, col=col)
    for horizon in sorted(facet.forecast["horizon"].unique()):
        points = facet.forecast[facet.forecast["horizon"] == horizon]
        for role in ROLES:
            figure.add_trace(
                _forecast_line(points, horizon, role, labels[role], when),
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
        parts.append(f"{', '.join(scored)} scored against a short total")
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


def _actual_line(actual: pd.DataFrame, when: str) -> go.Scatter:
    """The actual, its marker small enough that dense weeks merge into the line."""
    color = palette.rgba(palette.ACTUAL_INK)
    return go.Scatter(
        x=actual["ds"],
        y=actual["y"],
        mode="lines+markers",
        name=ACTUAL,
        legendgroup=ACTUAL,
        showlegend=True,
        line={"color": color, "width": _LINE_WIDTH},
        marker={"size": _ACTUAL_MARKER, "color": color},
        customdata=actual[["fiscal_year_month"]],
        hovertemplate=f"actual<br>{when}<br>%{{y:,.0f}}<extra></extra>",
    )


def _forecast_line(
    points: pd.DataFrame,
    horizon: str,
    role: str,
    label: str,
    when: str,
) -> go.Scatter:
    """One role's forecast for one horizon, named `label`: color and shape carry
    the horizon, dash and fill the role."""
    color = palette.rgba(palette.HORIZON_COLOR[horizon])
    return go.Scatter(
        x=points["ds"],
        y=points[role],
        mode="lines+markers",
        name=f"{label} {horizon}",
        legendgroup=f"{role} {horizon}",
        showlegend=True,
        line={"color": color, "width": _LINE_WIDTH, "dash": _DASH[role]},
        marker={
            "symbol": f"{palette.HORIZON_MARKER[horizon]}{_FILL[role]}",
            "size": _MARKER_SIZE[role],
            "color": color,
            "line": {"color": color, "width": _MARKER_STROKE[role]},
        },
        customdata=points[["fiscal_year_month"]],
        hovertemplate=f"{label} {horizon}<br>{when}<br>%{{y:,.0f}}<extra></extra>",
    )


def grid_data(
    calendar: pd.DataFrame,
    snapshot: pd.DataFrame,
    per_series_comparison: pd.DataFrame,
    raw_cv_forecasts: dict[str, pd.DataFrame],
    models: dict[str, str],
) -> GridData:
    """The frames both grids read, with the reindex and the month rule run once."""
    weekly = reindex_weeks(snapshot, calendar)
    return GridData(
        calendar=calendar,
        weekly=weekly,
        monthly=monthly_totals(weekly, calendar),
        per_series_comparison=per_series_comparison,
        raw_cv_forecasts=raw_cv_forecasts,
        models=models,
    )


def default_selection(
    per_series_comparison: pd.DataFrame,
) -> tuple[object, pd.Timestamp]:
    """The first series in sorted order, from the run's last origin."""
    return (
        sorted(per_series_comparison["unique_id"].unique())[0],
        per_series_comparison["forecast_origin_date"].max(),
    )


def grid_figure(
    data: GridData,
    selections: Sequence[tuple[object, pd.Timestamp]],
    grain: str,
    *,
    shared_y: bool,
    window: bool,
) -> go.Figure:
    """Six facets, three rows by two columns, on one x axis.

    At full history a short series draws only over the right of an axis a long one
    fills, which shows ragged starts. The backtest window runs from `_HISTORY`
    before the run's first origin to the panel's end, and fits y to it, which
    Plotly cannot: it fits a facet's y to all of its data, visible or not.

    Args:
        data: From `grid_data`.
        selections: Each facet's series and origin, in reading order.
        grain: `MONTHLY` or `WEEKLY`, for every facet.
        shared_y: One y range over every facet, rather than each its own.
        window: The backtest window rather than full history.

    Returns:
        go.Figure: The grid, its legend naming each trace once.
    """
    plot_height = ROWS * _FACET_HEIGHT + (ROWS - 1) * _ROW_GAP
    figure = make_subplots(
        rows=ROWS,
        cols=COLS,
        shared_xaxes="all",
        horizontal_spacing=_COLUMN_GAP,
        vertical_spacing=_ROW_GAP / plot_height,
    )
    facets = [build_facet(data, series, origin, grain) for series, origin in selections]
    for index, facet in enumerate(facets):
        row, col = divmod(index, COLS)
        add_facet(figure, facet, data.models, row=row + 1, col=col + 1)
    named = set()
    for trace in figure.data:
        trace.showlegend = trace.showlegend and trace.name not in named
        named.add(trace.name)

    end = data.weekly["ds"].max()
    start = (
        data.per_series_comparison["forecast_origin_date"].min() - _HISTORY
        if window
        else data.weekly["ds"].min()
    )
    figure.update_xaxes(range=_padded(start, end))
    visible = [_visible(facet, start) for facet in facets]
    if shared_y:
        figure.update_yaxes(range=_y_range(pd.concat(visible)))
    else:
        for index, values in enumerate(visible):
            # A lone repeated value would autoscale to a sliver around it.
            if window or values.nunique() == 1:
                row, col = divmod(index, COLS)
                figure.update_yaxes(range=_y_range(values), row=row + 1, col=col + 1)
    axes.style_axes(figure)
    figure.update_layout(
        **axes.base_layout(
            showlegend=True,
            hovermode="closest",
            height=plot_height + 120,
            legend={
                "orientation": "h",
                "x": 0,
                "y": 1,
                "yanchor": "top",
                "yref": "container",
            },
            margin={"l": 56, "r": 16, "t": 84, "b": 36},
        )
    )
    return figure


def render_line_grids(data: GridData) -> None:
    """Both grids in a collapsed expander. Each has its own grain, y-axis and
    window radios, a copy button, and a series and an origin selector per facet,
    laid out as the facets are."""
    comparison = data.per_series_comparison
    choices = {
        "series": series_options(comparison),
        "origin": origin_options(comparison),
    }
    defaults = dict(zip(choices, default_selection(comparison), strict=True))
    with st.expander(EXPANDER, expanded=False):
        for grid in range(1, GRIDS + 1):
            _keep_selections_in_this_run(grid, choices, defaults)
            st.markdown(f"**Grid {grid}**")
            grain_column, y_column, window_column, copy_column = st.columns(
                [1, 1, 1.2, 1.2]
            )
            grain = grain_column.radio(
                "grain", GRAINS, horizontal=True, key=grain_key(grid)
            )
            y_scale = y_column.radio(
                "y-axis", Y_SCALES, horizontal=True, key=y_key(grid)
            )
            extent = window_column.radio(
                "x-axis", WINDOWS, horizontal=True, key=window_key(grid)
            )
            copy_column.button(
                "Copy facet 1's selections to the rest",
                key=copy_key(grid),
                on_click=_copy_first_facet,
                args=(grid,),
            )
            selections = []
            for row in range(ROWS):
                for col, cell in enumerate(st.columns(COLS)):
                    facet = row * COLS + col + 1
                    series_column, origin_column = cell.columns(2)
                    series = series_column.selectbox(
                        f"facet {facet}: series",
                        choices["series"],
                        format_func=str,
                        key=selection_key(grid, facet, "series"),
                    )
                    origin = origin_column.selectbox(
                        f"facet {facet}: origin",
                        choices["origin"],
                        format_func=_origin_label,
                        key=selection_key(grid, facet, "origin"),
                    )
                    selections.append((series, origin))
            figure = grid_figure(
                data,
                selections,
                grain,
                shared_y=y_scale == "shared",
                window=extent == "backtest window",
            )
            st.plotly_chart(figure, width="stretch", key=f"{_KEY}_{grid}_chart")


def series_options(per_series_comparison: pd.DataFrame) -> list:
    """`global` first, where a long list cannot bury it, then each in sorted order."""
    return [GLOBAL, *sorted(per_series_comparison["unique_id"].unique())]


def origin_options(per_series_comparison: pd.DataFrame) -> list[pd.Timestamp]:
    """The run's origins, latest first, so the default leads."""
    origins = pd.to_datetime(per_series_comparison["forecast_origin_date"].unique())
    return sorted(origins, reverse=True)


def grain_key(grid: int) -> str:
    """The widget key of a grid's grain radio."""
    return f"{_KEY}_{grid}_grain"


def y_key(grid: int) -> str:
    """The widget key of a grid's y-axis radio."""
    return f"{_KEY}_{grid}_y"


def window_key(grid: int) -> str:
    """The widget key of a grid's x-axis radio."""
    return f"{_KEY}_{grid}_window"


def copy_key(grid: int) -> str:
    """The widget key of a grid's copy button."""
    return f"{_KEY}_{grid}_copy"


def selection_key(grid: int, facet: int, kind: str) -> str:
    """The widget key of one facet's `series` or `origin` selector."""
    return f"{_KEY}_{grid}_{facet}_{kind}"


def _keep_selections_in_this_run(
    grid: int, choices: dict[str, list], defaults: dict[str, object]
) -> None:
    """Default each selection the run's choices lack, as on first draw or after a
    run switch. Set here, not as an index, so the copy button can write the keys."""
    for facet in range(1, ROWS * COLS + 1):
        for kind, options in choices.items():
            key = selection_key(grid, facet, kind)
            if st.session_state.get(key) not in options:
                st.session_state[key] = defaults[kind]


def _copy_first_facet(grid: int) -> None:
    """Facet 1's series and origin into the grid's other facets, run before the
    widgets draw again."""
    for facet in range(2, ROWS * COLS + 1):
        for kind in ("series", "origin"):
            st.session_state[selection_key(grid, facet, kind)] = st.session_state[
                selection_key(grid, 1, kind)
            ]


def _origin_label(origin: pd.Timestamp) -> str:
    """An origin as its date."""
    return f"{origin:%Y-%m-%d}"


def _visible(facet: Facet, start: pd.Timestamp) -> pd.Series:
    """Every value the facet plots from `start` on."""
    actual = facet.actual.loc[facet.actual["ds"] >= start, "y"]
    return pd.concat([actual, *(facet.forecast[role] for role in ROLES)]).dropna()


def _y_range(values: pd.Series) -> list | None:
    """A padded range over `values`, or 0 to `_FLAT_HEADROOM` of a lone repeated
    one. Measured: Plotly widens the zero case's `[0, 0]` to -1 to 1."""
    if values.empty:
        return None
    low, high = values.min(), values.max()
    if low == high:
        return sorted([0.0, _FLAT_HEADROOM * high])
    return _padded(low, high)


def _padded(low, high) -> list:
    """`low` to `high`, widened by `_PAD` of the span at each end."""
    margin = (high - low) * _PAD
    return [low - margin, high + margin]
