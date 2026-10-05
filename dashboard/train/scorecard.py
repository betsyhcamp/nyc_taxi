from collections.abc import Sequence

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard.shared import axes, palette

# GLOBAL_TIER is imported rather than redeclared: it is the producer's own public
# name for the aggregate column beside the tier partition, never a total.
from fcstnyctaxi.core.train.evaluate_impl import GLOBAL_TIER

_BENCHMARK_MARK = {
    "symbol": "line-ew",
    "size": 18,
    "line": {"color": palette.AXIS_COLOR, "width": 2},
}


# The two scorecard bar rows: `global` then the configured tiers, challenger only.
def bar_row(
    summary_metrics: pd.DataFrame,
    metric: str,
    tier_labels: Sequence[str],
    challenger_model: str,
    benchmark_model: str,
) -> go.Figure:
    """One row of grouped bars, two per facet, horizon_1 and horizon_2.

    The grid comes from the configured `tier_labels`, not the labels present, so
    an absent tier draws an empty facet rather than shifting the grid.
    """
    columns = [GLOBAL_TIER, *tier_labels]
    rows = summary_metrics[summary_metrics["metric"] == metric]
    challenger = _cells(rows, challenger_model)
    benchmark = _cells(rows, benchmark_model)
    horizons = sorted({horizon for _, horizon in challenger})

    show_benchmark = palette.SHOW_BENCHMARK[metric]
    scaled = [value for value, _ in challenger.values()]
    if show_benchmark:
        scaled += [value for value, _ in benchmark.values()]
    low, high = axes.metric_range(scaled, metric)

    reference = palette.REFERENCE_LINE[metric]
    baseline = 0.0 if reference is None else reference

    figure = make_subplots(
        rows=1,
        cols=len(columns),
        shared_yaxes=True,
        subplot_titles=columns,
        horizontal_spacing=0.012,
    )
    for column, tier in enumerate(columns, start=1):
        drawn = 0
        for horizon in horizons:
            cell = challenger.get((tier, horizon))
            if cell is None:
                continue
            value, n_obs = cell
            drawn += 1
            figure.add_trace(
                _bar(horizon, value, n_obs, baseline, legend=column == 1),
                row=1,
                col=column,
            )
            if show_benchmark and (tier, horizon) in benchmark:
                figure.add_trace(
                    _benchmark_tick(horizon, benchmark[(tier, horizon)][0]),
                    row=1,
                    col=column,
                )
        if not drawn:
            figure.add_trace(_empty_facet(), row=1, col=column)

    axes.add_reference(figure, metric)
    axes.style_axes(figure)
    # The legend names the two horizons once; repeating them under all six facets
    # is the same fact six times and costs the row a third of its height.
    figure.update_xaxes(showticklabels=False)
    figure.update_yaxes(range=[low, high])
    figure.update_layout(**axes.base_layout(showlegend=True))
    figure.update_annotations(font_size=12)
    return figure


def render_bar_row(
    summary_metrics: pd.DataFrame,
    tier_labels: Sequence[str],
    challenger_model: str,
    benchmark_model: str,
    *,
    key: str,
    default_metric: str,
) -> None:
    """One row with its own metric selector. Never a shared one: the two rows
    exist precisely so skill and bias are readable at the same time."""
    metric = st.selectbox(
        "metric",
        palette.METRICS,
        index=palette.METRICS.index(default_metric),
        key=key,
    )
    figure = bar_row(
        summary_metrics, metric, tier_labels, challenger_model, benchmark_model
    )
    st.plotly_chart(figure, use_container_width=True)


def _cells(rows: pd.DataFrame, model: str) -> dict[tuple[str, str], tuple[float, int]]:
    """(tier, horizon) to (value, n_obs) for one model.

    A dict rather than a filter per facet, and keyed on `str(tier)`: the column
    is an ordered categorical, and a configured label absent from its categories
    compares False rather than raising, which would read as an empty facet
    whether the tier is absent or the comparison is wrong.
    """
    model_rows = rows[rows["model"] == model]
    return {
        (str(tier), str(horizon)): (float(value), int(n_obs))
        for tier, horizon, value, n_obs in zip(
            model_rows["tier"],
            model_rows["horizon"],
            model_rows["value"],
            model_rows["n_obs"],
            strict=True,
        )
    }


def _bar(
    horizon: str, value: float, n_obs: int, baseline: float, *, legend: bool
) -> go.Bar:
    """One challenger bar, from the baseline to its value.

    Plotly reads `y` as the bar's length from `base`, so the real value rides in
    `customdata` or hover shows the delta. Color is the horizon's and never the
    magnitude's: a breach is direction from the baseline.
    """
    return go.Bar(
        x=[horizon],
        y=[value - baseline],
        base=[baseline],
        name=horizon,
        legendgroup=horizon,
        showlegend=legend,
        marker_color=palette.rgba(palette.HORIZON_COLOR[horizon]),
        text=[f"{n_obs:,}"],
        textposition="outside",
        textfont={"size": 9},
        customdata=[[value, n_obs]],
        hovertemplate=(
            f"{horizon}<br>%{{customdata[0]:.4f}}<br>n_obs %{{customdata[1]:,}}"
            "<extra></extra>"
        ),
    )


def _benchmark_tick(horizon: str, value: float) -> go.Scatter:
    """A muted tick at the benchmark's value, for an absolute metric only.

    A relative metric needs no tick, its benchmark reading ~1.0 by construction, and
    the axis color keeps the mark out of the horizon palette. Absolute metrics need the
    benchmark forecast as a refereence.
    """
    return go.Scatter(
        x=[horizon],
        y=[value],
        mode="markers",
        marker=_BENCHMARK_MARK,
        name="benchmark",
        legendgroup="benchmark",
        showlegend=False,
        hovertemplate=f"benchmark {horizon}<br>%{{y:.4f}}<extra></extra>",
    )


def _empty_facet() -> go.Bar:
    """A placeholder for a configured tier with no rows.

    Measured: Plotly materializes no axis for a facet holding no trace, and
    `add_hline` reaches only facets that hold one, with or without `row="all"`.
    So without this the tier's panel loses its frame and the reference rule
    breaks mid-row, instead of rendering as the empty panel the grid promises.
    """
    return go.Bar(x=[], y=[], showlegend=False, hoverinfo="skip")
