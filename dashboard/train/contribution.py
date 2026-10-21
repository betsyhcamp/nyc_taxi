from collections.abc import Sequence

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard.shared import axes, palette
from fcstnyctaxi.lib.fold_metrics import compute_wrmae_pooled_shares

_KEY = "contribution"
SUMS_ANNOTATION = "share_sums"
"""Names each facet's Σx and Σy annotation."""
TOP_N = 20

_MARKER_SIZE = 8
_STROKE = 2
_FACET_HEIGHT = 520

# The decomposition's inputs, each side renamed to the frame compute_wrmae_pooled
# takes. The actual and the weight are the challenger's; evaluate checked they agree.
_JOIN_KEYS = ["forecast_origin_date", "predicted_fiscal_year_month", "unique_id"]
_SHARED = ["actual_monthly_total", "series_weight"]


def horizon_shares(
    per_series_comparison: pd.DataFrame, tier_labels: Sequence[str]
) -> dict[str, pd.DataFrame]:
    """Per horizon, each series' benchmark share `x`, challenger share `y`, net
    effect `y - x`, tier at its last origin, and a label naming any other tier.

    One tier per series, read across both horizons, so a series keeps one symbol
    in both facets. The label counts that horizon's folds, since the last-origin
    tier says where a series ended, not where it earned its share.
    """
    rows = per_series_comparison.assign(
        tier=per_series_comparison["tier"].astype(str),
        horizon=per_series_comparison["horizon"].astype(str),
    )
    last_tier = (
        rows.sort_values("forecast_origin_date")
        .drop_duplicates("unique_id", keep="last")
        .set_index("unique_id")["tier"]
    )

    shares = {}
    for horizon, folds in rows.groupby("horizon"):
        challenger = folds[_JOIN_KEYS + _SHARED + ["monthly_forecast_ch"]].rename(
            columns={"monthly_forecast_ch": "monthly_forecast"}
        )
        benchmark = folds[_JOIN_KEYS + ["monthly_forecast_bm"]].rename(
            columns={"monthly_forecast_bm": "monthly_forecast"}
        )
        frame = compute_wrmae_pooled_shares(challenger, benchmark)
        shares[horizon] = frame.assign(
            net=frame["y"] - frame["x"],
            tier=frame["unique_id"].map(last_tier),
            label=frame["unique_id"].map(
                _tier_labels_held(folds, last_tier, tier_labels)
            ),
        )
    return shares


def contribution_figure(
    shares: dict[str, pd.DataFrame], tier_labels: Sequence[str]
) -> go.Figure:
    """One facet per horizon: the benchmark's share against the challenger's, one
    trace per tier, `y = x` and the two sums.

    The first facet's y-axis is anchored 1:1 to its x, which is what makes `y = x`
    a 45 degree line. Plotly forbids `scaleanchor` beside `matches` on one axis,
    so the other facet takes the aspect by matching both of the first's axes.

    Raises:
        ValueError: If no tier ramp exists for that many configured tiers.
    """
    encoding = palette.tier_encoding(tier_labels)
    horizons = sorted(shares)
    # Across both facets, since their ranges match: y = x reaches the shared extent.
    top = max(float(frame[["x", "y"]].max().max()) for frame in shares.values())
    figure = make_subplots(
        rows=1, cols=len(horizons), subplot_titles=horizons, horizontal_spacing=0.08
    )
    for column, horizon in enumerate(horizons, start=1):
        frame = shares[horizon]
        for tier in tier_labels:
            points = frame[frame["tier"] == tier]
            if not points.empty:
                figure.add_trace(
                    _tier_marks(points, tier, encoding[tier], legend=column == 1),
                    row=1,
                    col=column,
                )
        figure.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=top,
            y1=top,
            line={"color": palette.AXIS_COLOR, "width": 2},
            layer="below",
            row=1,
            col=column,
        )
        figure.add_annotation(
            text=f"Σx = {frame['x'].sum():.3f}   Σy = {frame['y'].sum():.3f}",
            name=SUMS_ANNOTATION,
            xref="x domain",
            yref="y domain",
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            showarrow=False,
            row=1,
            col=column,
        )

    axes.style_axes(figure)
    figure.update_xaxes(rangemode="tozero", title_text="benchmark's share, x")
    figure.update_yaxes(rangemode="tozero")
    figure.update_yaxes(title_text="challenger's share, y", row=1, col=1)
    figure.update_layout(yaxis={"scaleanchor": "x", "scaleratio": 1})
    for column in range(2, len(horizons) + 1):
        figure.update_xaxes(matches="x", row=1, col=column)
        figure.update_yaxes(matches="y", row=1, col=column)
    figure.update_layout(
        **axes.base_layout(showlegend=True, hovermode="closest", height=_FACET_HEIGHT)
    )
    return figure


def top_table(frame: pd.DataFrame) -> pd.DataFrame:
    """The horizon's `TOP_N` largest benchmark shares, largest first."""
    top = frame.nlargest(TOP_N, "x")
    return top[["unique_id", "label", "x", "y", "net"]].rename(
        columns={"unique_id": "series", "label": "tier", "net": "y - x"}
    )


def render_contribution(
    per_series_comparison: pd.DataFrame, tier_labels: Sequence[str]
) -> None:
    """The scatter, or a message in its place, then a top-share table per facet."""
    st.subheader("Contribution by series")
    shares = horizon_shares(per_series_comparison, tier_labels)
    if len(tier_labels) in palette.TIER_RAMPS:
        figure = contribution_figure(shares, tier_labels)
        st.plotly_chart(figure, width="stretch", key=f"{_KEY}_chart")
    else:
        st.warning(
            f"The run configures {len(tier_labels)} tiers, and the tier encoding is "
            f"validated for {min(palette.TIER_RAMPS)} to {max(palette.TIER_RAMPS)} "
            "only, so the scatter is not drawn. The tables below are unaffected."
        )
    st.caption(
        "Each series' share of the benchmark's error (x) and the challenger's (y), "
        "each fold divided by its own benchmark total. Σx is 1 and Σy is the "
        "scorecard's wrmae_pooled; a series above y = x lost ground."
    )
    for column, horizon in zip(st.columns(len(shares)), sorted(shares), strict=True):
        column.caption(f"{horizon}: the {TOP_N} largest benchmark shares")
        column.dataframe(
            top_table(shares[horizon]),
            hide_index=True,
            column_config={
                name: st.column_config.NumberColumn(format="%.4f")
                for name in ("x", "y", "y - x")
            },
        )


def _tier_labels_held(
    folds: pd.DataFrame, last_tier: pd.Series, tier_labels: Sequence[str]
) -> pd.Series:
    """Per series, its last-origin tier plus any other tier it held in these folds,
    as "middle (also high, 21 of 30 folds)"."""
    held = (
        folds.groupby(["unique_id", "tier"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=list(tier_labels), fill_value=0)
    )
    labels = {}
    for unique_id, counts in held.iterrows():
        tier = last_tier[unique_id]
        total = int(counts.sum())
        others = [
            f"{other}, {int(count)} of {total} folds"
            for other, count in counts.items()
            if other != tier and count
        ]
        labels[unique_id] = f"{tier} (also {'; '.join(others)})" if others else tier
    return pd.Series(labels)


def _tier_marks(
    points: pd.DataFrame,
    tier: str,
    color_symbol: tuple[str, str],
    *,
    legend: bool,
) -> go.Scatter:
    """One tier's series in one facet. Open symbols, so overlapping series stay
    countable; `Scatter`, since SVG copes at 800 points and `Scattergl` has fewer
    symbols."""
    color, symbol = color_symbol
    return go.Scatter(
        x=points["x"],
        y=points["y"],
        mode="markers",
        name=tier,
        legendgroup=tier,
        showlegend=legend,
        marker={
            "symbol": symbol,
            "size": _MARKER_SIZE,
            "color": palette.rgba(color),
            "line": {"color": palette.rgba(color), "width": _STROKE},
        },
        customdata=points[["unique_id", "label", "net"]],
        hovertemplate=(
            "%{customdata[0]}<br>%{customdata[1]}"
            "<br>x %{x:.4f}   y %{y:.4f}"
            "<br>y - x %{customdata[2]:+.4f}<extra></extra>"
        ),
    )
