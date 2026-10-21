from collections.abc import Sequence

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard.shared import axes, palette
from fcstnyctaxi.core.train.evaluate_impl import GLOBAL_TIER

_KEY = "origin_skill"
_DEFAULT_METRIC = "wrmae_pooled"
COUNTS_ANNOTATION = "fold_counts"
"""Names each facet's fold-count annotation apart from its tier label."""
LATEST_RING = "latest origin"
"""Names the traces that ring the run's last origin."""

_FACET_HEIGHT = 110
"""Plot area per facet, in px."""
_FACET_GAP = 12
_GUTTER = 150
_LABEL_SHIFT = -52
"""Where the right-aligned tier names end, in px left of the plot, clear of ticks."""
_TICK_EVERY = 4

_MARKER_SIZE = 9
_LATEST_SIZE = 13
_HALO_WIDTH = 2
_RING_WIDTH = 2


def origin_figure(
    fold_metrics: pd.DataFrame,
    metric: str,
    tier_labels: Sequence[str],
    challenger_model: str,
    benchmark_model: str,
) -> go.Figure:
    """A column of facets, `global` then each configured tier, with the
    challenger's value at each origin on one shared date axis.

    A date scale rather than an index, since a gap between origins is real
    information. The grid comes from the configured `tier_labels`, so an absent
    tier draws an empty facet rather than shifting the column.
    """
    facets = [GLOBAL_TIER, *tier_labels]
    rows = fold_metrics[fold_metrics["metric"] == metric]
    rows = rows.assign(
        tier=rows["tier"].astype(str), horizon=rows["horizon"].astype(str)
    )
    challenger = rows[rows["model"] == challenger_model]
    benchmark = rows[rows["model"] == benchmark_model]
    horizons = sorted(challenger["horizon"].unique())
    origins = sorted(challenger["forecast_origin_date"].unique())

    show_benchmark = palette.SHOW_BENCHMARK[metric]
    scaled = list(challenger["value"])
    if show_benchmark:
        scaled += list(benchmark["value"])
    low, high = axes.metric_range(scaled, metric)

    plot_height = len(facets) * _FACET_HEIGHT + (len(facets) - 1) * _FACET_GAP
    figure = make_subplots(
        rows=len(facets),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=_FACET_GAP / plot_height,
    )
    for row, tier in enumerate(facets, start=1):
        facet = challenger[challenger["tier"] == tier]
        if facet.empty:
            figure.add_trace(axes.empty_facet(), row=row, col=1)
        for horizon in horizons:
            points = facet[facet["horizon"] == horizon]
            if points.empty:
                continue
            latest = points[points["forecast_origin_date"] == origins[-1]]
            if not latest.empty:
                figure.add_trace(_latest_ring(latest, horizon), row=row, col=1)
            figure.add_trace(
                _marks(points, horizon, origins[-1], legend=row == 1), row=row, col=1
            )
            if show_benchmark:
                ticks = benchmark[
                    (benchmark["tier"] == tier) & (benchmark["horizon"] == horizon)
                ]
                figure.add_trace(_benchmark_ticks(ticks, horizon), row=row, col=1)
        _label_facet(figure, row, tier, _fold_counts(facet, horizons))

    axes.add_reference(figure, metric)
    axes.style_axes(figure)
    # Explicit, since Plotly types an axis by its data and an empty facet has none:
    # as the bottom facet, a linear axis would drop every date label.
    figure.update_xaxes(
        type="date", tickvals=origins[::_TICK_EVERY], tickformat="%Y-%m-%d"
    )
    figure.update_yaxes(range=[low, high])
    figure.update_layout(
        **axes.base_layout(
            showlegend=True,
            height=plot_height + 64,
            margin={"l": _GUTTER, "r": 16, "t": 32, "b": 32},
        )
    )
    return figure


def _fold_counts(facet: pd.DataFrame, horizons: Sequence[str]) -> list[int]:
    """Finite fold values per horizon, in `horizons` order: what each facet averages."""
    finite = facet[facet["value"].notna()]
    return [int((finite["horizon"] == horizon).sum()) for horizon in horizons]


def render_origin_skill(
    fold_metrics: pd.DataFrame,
    tier_labels: Sequence[str],
    challenger_model: str,
    benchmark_model: str,
) -> None:
    """The panel under its own metric selector, independent of the scorecard rows'."""
    st.subheader("By forecast origin")
    metric = st.selectbox(
        "metric",
        palette.METRICS,
        index=palette.METRICS.index(_DEFAULT_METRIC),
        key=f"{_KEY}_metric",
    )
    figure = origin_figure(
        fold_metrics, metric, tier_labels, challenger_model, benchmark_model
    )
    st.plotly_chart(figure, width="stretch", key=f"{_KEY}_chart")
    st.caption(
        "Under each tier, its folds per horizon, horizon_1 / horizon_2. The final "
        "month's origins carry no horizon_2 mark: their evaluation period ends at "
        "the last complete month, so only horizon_1 exists."
    )


def _marks(
    points: pd.DataFrame, horizon: str, latest_origin: pd.Timestamp, *, legend: bool
) -> go.Scatter:
    """One horizon's values in one facet, the latest origin's marker larger."""
    points = points.sort_values("forecast_origin_date")
    latest = points["forecast_origin_date"] == latest_origin
    return go.Scatter(
        x=points["forecast_origin_date"],
        y=points["value"],
        mode="markers",
        name=horizon,
        legendgroup=horizon,
        showlegend=legend,
        marker={
            "color": palette.rgba(palette.HORIZON_COLOR[horizon]),
            "symbol": palette.HORIZON_MARKER[horizon],
            "size": [
                _LATEST_SIZE if is_latest else _MARKER_SIZE for is_latest in latest
            ],
            **axes.horizon_offset(horizon),
        },
        customdata=points[["n_obs"]],
        hovertemplate=(
            f"{horizon}<br>%{{x|%Y-%m-%d}}<br>%{{y:.4f}}"
            "<br>n_obs %{customdata[0]:,}<extra></extra>"
        ),
    )


def _latest_ring(point: pd.DataFrame, horizon: str) -> go.Scatter:
    """A gray ring with a white halo inside it, drawn beneath the latest origin's
    marker. Not a white ring alone, which on the white plot area shows only where
    it overlaps a line or a mark."""
    return go.Scatter(
        x=point["forecast_origin_date"],
        y=point["value"],
        mode="markers",
        name=LATEST_RING,
        legendgroup=horizon,
        showlegend=False,
        hoverinfo="skip",
        marker={
            "symbol": palette.HORIZON_MARKER[horizon],
            # A stroke straddles the edge, so this leaves _HALO_WIDTH px of the
            # white fill between the marker and the ring.
            "size": _LATEST_SIZE + 2 * _HALO_WIDTH + _RING_WIDTH,
            "color": palette.PLOT_BGCOLOR,
            "line": {"color": palette.AXIS_COLOR, "width": _RING_WIDTH},
            **axes.horizon_offset(horizon),
        },
    )


def _benchmark_ticks(points: pd.DataFrame, horizon: str) -> go.Scatter:
    """The benchmark's values at its horizon's offset."""
    return go.Scatter(
        x=points["forecast_origin_date"],
        y=points["value"],
        mode="markers",
        name="benchmark",
        legendgroup="benchmark",
        showlegend=False,
        marker=axes.offset_benchmark_mark(horizon),
        hovertemplate=(
            f"benchmark {horizon}<br>%{{x|%Y-%m-%d}}<br>%{{y:.4f}}<extra></extra>"
        ),
    )


def _label_facet(figure: go.Figure, row: int, tier: str, counts: list[int]) -> None:
    """The tier's name and, under it, its fold counts, right-aligned in the left
    gutter so no label shifts a plot area."""
    gutter = {
        "xref": "x domain",
        "yref": "y domain",
        "x": 0,
        "y": 0.5,
        "xanchor": "right",
        "xshift": _LABEL_SHIFT,
        "showarrow": False,
    }
    figure.add_annotation(text=tier, yanchor="bottom", row=row, col=1, **gutter)
    figure.add_annotation(
        text=" / ".join(str(count) for count in counts),
        yanchor="top",
        font={"size": 10},
        name=COUNTS_ANNOTATION,
        row=row,
        col=1,
        **gutter,
    )
