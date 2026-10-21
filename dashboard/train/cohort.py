from collections.abc import Sequence

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard.shared import axes, palette
from fcstnyctaxi.core.train.evaluate_impl import GLOBAL_TIER
from fcstnyctaxi.lib.origin_modeling_table.builders import enumerate_origins

_KEY = "cohort"
_DEFAULT_METRIC = "wrmae_pooled"
METRIC_KEY = f"{_KEY}_metric"
COHORT_KEYS = ["weeks_in_month", "weeks_actualized"]
ROWS_ANNOTATION = "cohort_rows"
"""Names each row's target-month label apart from the column titles."""
BENCHMARK_NAME = "benchmark mean"

_CELL_KEYS = ["model", "horizon", "tier", *COHORT_KEYS]

_ROW_HEIGHT = 160
"""Plot area per row, in px."""
_ROW_GAP = 40
_COLUMN_GAP = 0.012
_GUTTER = 150
_LABEL_SHIFT = -40
"""Where the right-aligned row labels end, in px left of the plot, clear of ticks."""

_RING_SIZE = 8
_RING_STROKE = 2
_MEAN_LENGTH = 10
_MEAN_WIDTH = 3
# Not BENCHMARK_MARK's 18px: every position here holds both horizons' ticks, and
# at the 5px offsets two 18px ticks overlap by 8px and both look centered. At 14px
# they overlap by 4px, each sits on its horizon's side, and 2px of gray still
# shows past each end of a coincident mean.
_BENCHMARK_LENGTH = 14

_CAPTION = (
    "Each ring is one fold. The short colored line is the folds' mean, and the "
    "gray tick the benchmark's; hover a mean for how many folds it averages."
)
_WAPE_CAPTION = (
    "Expect horizon_1 to fall toward zero as the month fills in, because the weeks "
    "already observed carry no error. horizon_2 has no observed weeks and falls "
    "only as its lead time shortens. Read each mark against the benchmark tick, "
    "not against its neighbors."
)


def attach_cohorts(fold_metrics: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Each fold row's position cohort: the weeks its origin's horizon_1 target
    month has, and how many of them were actualized at the origin.

    Joined on the origin alone, not the fold key. A horizon_2 row predicts the
    month after its origin's spine entry, so a fold-key join would leave it
    without a cohort; on the origin, both horizons share one, as the strip needs.
    """
    spine = enumerate_origins(calendar)[["forecast_origin_date", *COHORT_KEYS]]
    return fold_metrics.merge(spine, on="forecast_origin_date", how="left")


def cohort_figure(
    cohorts: pd.DataFrame,
    metric: str,
    tier_labels: Sequence[str],
    challenger_model: str,
    benchmark_model: str,
) -> go.Figure:
    """A row per horizon_1 target-month length in the run and a column for `global`
    and each configured tier, all on one y.

    Each fold is an open ring at its horizon's offset, so rings that overlap stay
    countable. Its cell's mean is a short line in the horizon's color, drawn over
    the benchmark's longer gray tick so that equal means still show both.
    """
    facets = [GLOBAL_TIER, *tier_labels]
    rows = cohorts[cohorts["metric"] == metric]
    rows = rows.assign(
        tier=rows["tier"].astype(str), horizon=rows["horizon"].astype(str)
    )
    means = _cell_means(rows)
    challenger = rows[rows["model"] == challenger_model]
    challenger_means = means[means["model"] == challenger_model]
    benchmark_means = means[means["model"] == benchmark_model]
    lengths = sorted(challenger["weeks_in_month"].unique())
    horizons = sorted(challenger["horizon"].unique())

    show_benchmark = palette.SHOW_BENCHMARK[metric]
    scaled = list(challenger["value"])
    if show_benchmark:
        scaled += list(benchmark_means["mean"])
    low, high = axes.metric_range(scaled, metric)

    plot_height = len(lengths) * _ROW_HEIGHT + (len(lengths) - 1) * _ROW_GAP
    figure = make_subplots(
        rows=len(lengths),
        cols=len(facets),
        shared_yaxes="all",
        horizontal_spacing=_COLUMN_GAP,
        vertical_spacing=_ROW_GAP / plot_height,
        subplot_titles=[*facets, *[""] * (len(facets) * (len(lengths) - 1))],
    )
    for row, length in enumerate(lengths, start=1):
        for column, tier in enumerate(facets, start=1):
            first = row == 1 and column == 1
            # First, so the gray ticks sit beneath the rings and the means.
            if show_benchmark:
                for horizon in horizons:
                    ticks = _cell(benchmark_means, tier, horizon, length)
                    figure.add_trace(
                        _benchmark_ticks(
                            ticks, horizon, legend=first and horizon == horizons[0]
                        ),
                        row=row,
                        col=column,
                    )
            for horizon in horizons:
                folds = _cell(challenger, tier, horizon, length)
                figure.add_trace(
                    _rings(folds, horizon, legend=first), row=row, col=column
                )
            for horizon in horizons:
                cell_means = _cell(challenger_means, tier, horizon, length)
                figure.add_trace(_mean_lines(cell_means, horizon), row=row, col=column)
        figure.add_annotation(
            text=f"horizon_1<br>target month:<br>{length} weeks",
            name=ROWS_ANNOTATION,
            xref="x domain",
            yref="y domain",
            x=0,
            y=0.5,
            xanchor="right",
            xshift=_LABEL_SHIFT,
            align="right",
            showarrow=False,
            row=row,
            col=1,
        )
        figure.update_xaxes(
            range=[-0.6, length - 0.4], tickvals=list(range(length)), row=row
        )

    axes.add_reference(figure, metric)
    axes.style_axes(figure)
    figure.update_yaxes(range=[low, high])
    figure.add_annotation(
        text="weeks actualized",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0,
        yshift=-36,
        yanchor="top",
        showarrow=False,
    )
    figure.update_layout(
        **axes.base_layout(
            showlegend=True,
            hovermode="closest",
            height=plot_height + 140,
            legend={"orientation": "h", "x": 1, "xanchor": "right", "y": 1.16},
            margin={"l": _GUTTER, "r": 16, "t": 84, "b": 56},
        )
    )
    return figure


def render_cohort(
    fold_metrics: pd.DataFrame,
    calendar: pd.DataFrame,
    tier_labels: Sequence[str],
    challenger_model: str,
    benchmark_model: str,
) -> None:
    """The strip under its own metric selector, with what to expect under wape."""
    st.subheader("By position in the target month")
    metric = st.selectbox(
        "metric",
        palette.METRICS,
        index=palette.METRICS.index(_DEFAULT_METRIC),
        key=METRIC_KEY,
    )
    figure = cohort_figure(
        attach_cohorts(fold_metrics, calendar),
        metric,
        tier_labels,
        challenger_model,
        benchmark_model,
    )
    st.plotly_chart(figure, width="stretch", key=f"{_KEY}_chart")
    st.caption(_CAPTION)
    if metric == "wape":
        st.caption(_WAPE_CAPTION)


def _cell_means(rows: pd.DataFrame) -> pd.DataFrame:
    """Each cell's nanmean, as evaluate aggregates upward, with its finite and
    total fold counts."""
    return (
        rows.groupby(_CELL_KEYS, observed=True)["value"]
        .agg(mean="mean", finite="count", folds="size")
        .reset_index()
    )


def _cell(frame: pd.DataFrame, tier: str, horizon: str, length: int) -> pd.DataFrame:
    """One facet's rows for one horizon."""
    return frame[
        (frame["tier"] == tier)
        & (frame["horizon"] == horizon)
        & (frame["weeks_in_month"] == length)
    ]


def _rings(folds: pd.DataFrame, horizon: str, *, legend: bool) -> go.Scatter:
    """One horizon's fold values in one facet, an open ring each."""
    color = palette.rgba(palette.HORIZON_COLOR[horizon])
    return go.Scatter(
        x=folds["weeks_actualized"],
        y=folds["value"],
        mode="markers",
        name=horizon,
        legendgroup=horizon,
        showlegend=legend,
        marker={
            "symbol": f"{palette.HORIZON_MARKER[horizon]}-open",
            "size": _RING_SIZE,
            "color": color,
            "line": {"color": color, "width": _RING_STROKE},
            **axes.horizon_offset(horizon),
        },
        customdata=pd.DataFrame(
            {
                "origin": folds["forecast_origin_date"].dt.strftime("%Y-%m-%d"),
                "month": folds["predicted_fiscal_year_month"],
                "n_obs": folds["n_obs"],
            }
        ),
        hovertemplate=(
            f"{horizon} fold<br>origin %{{customdata[0]}}"
            "<br>predicts %{customdata[1]}<br>%{y:.4f}"
            "<br>n_obs %{customdata[2]:,}<extra></extra>"
        ),
    )


def _mean_lines(cell_means: pd.DataFrame, horizon: str) -> go.Scatter:
    """One horizon's cell means in one facet, each a short line in its color."""
    line = {"color": palette.rgba(palette.HORIZON_COLOR[horizon]), "width": _MEAN_WIDTH}
    return go.Scatter(
        x=cell_means["weeks_actualized"],
        y=cell_means["mean"],
        mode="markers",
        name=f"{horizon} mean",
        legendgroup=horizon,
        showlegend=False,
        marker=axes.offset_tick(horizon, line, _MEAN_LENGTH),
        customdata=cell_means[["finite", "folds"]],
        hovertemplate=(
            f"{horizon} mean of %{{customdata[0]}} of %{{customdata[1]}} folds"
            "<br>%{y:.4f}<extra></extra>"
        ),
    )


def _benchmark_ticks(
    cell_means: pd.DataFrame, horizon: str, *, legend: bool
) -> go.Scatter:
    """The benchmark's cell means at a horizon's offset. Means only, no folds,
    which would double cells of 2 to 5 and blur the count."""
    return go.Scatter(
        x=cell_means["weeks_actualized"],
        y=cell_means["mean"],
        mode="markers",
        name=BENCHMARK_NAME,
        legendgroup="benchmark",
        showlegend=legend,
        marker=axes.offset_tick(
            horizon, axes.BENCHMARK_MARK["line"], _BENCHMARK_LENGTH
        ),
        hovertemplate=f"benchmark {horizon} mean<br>%{{y:.4f}}<extra></extra>",
    )
