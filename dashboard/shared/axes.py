from collections.abc import Iterable
from typing import Any

import numpy as np
import plotly.graph_objects as go

from dashboard.shared import palette

_PAD = 0.06
"""Fraction of the span added at each end of axes."""

BENCHMARK_MARK = {
    "symbol": "line-ew",
    "size": 18,
    "line": {"color": palette.AXIS_COLOR, "width": 2},
}
"""The benchmark's muted tick, in the axis color to stay out of the horizon palette."""

# Plotly places a scatter point in data units only. Measured: `standoff` moves the
# drawn marker in px opposite to `angle`, whatever its docstring says, and a circle
# or a diamond turned a quarter is unchanged. So horizon_1 sits 5px left of its x
# and horizon_2 5px right.
_OFFSET_ANGLE = {"horizon_1": 90, "horizon_2": 270}
_OFFSET_PX = 5


def metric_range(values: Iterable[float], metric: str) -> tuple[float, float]:
    """The y-range one metric's row shares across its facets.

    The reference is forced in, which is not cosmetic: in the expected steady
    state, where the challenger wins in every tier, plain min-max padding drops
    1.0 off the chart entirely.

    Symmetric where the reference is zero, which puts it at the vertical center
    of every facet so it reads as one continuous mid-line. Skill is one-sided, so
    symmetry there would spend most of the axis unvisited.
    """
    finite = np.asarray(list(values), dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        raise ValueError(
            f"no finite {metric} value to scale an axis by, so every facet would "
            "draw empty with nothing to say why."
        )

    low, high = float(finite.min()), float(finite.max())

    if palette.SYMMETRIC_RANGE[metric]:
        bound = max(abs(low), abs(high))
        margin = (bound if bound > 0 else 1.0) * _PAD
        return -(bound + margin), bound + margin

    reference = palette.REFERENCE_LINE[metric]
    if reference is not None:
        low, high = min(low, reference), max(high, reference)

    span = high - low
    margin = (span if span > 0 else max(abs(low), 1.0)) * _PAD
    low, high = low - margin, high + margin
    # A metric with no negative value keeps its floor at zero rather than padding
    # into a region it cannot occupy. Never an expansion, so skill is unaffected.
    if finite.min() >= 0:
        low = max(low, 0.0)
    return low, high


def add_reference(figure: go.Figure, metric: str) -> None:
    """Draw `REFERENCE_LINE` as a constant across every facet.

    Never read off a benchmark row: that row reads near 1.0 rather than 1.0,
    because `wrmae_per_series` renormalizes.
    """
    reference = palette.REFERENCE_LINE[metric]
    if reference is None:
        return
    figure.add_hline(y=reference, line_color=palette.AXIS_COLOR, line_width=2)


def horizon_offset(horizon: str) -> dict[str, int]:
    """Marker properties that set a horizon's mark beside its x, never on it."""
    return {"angle": _OFFSET_ANGLE[horizon], "standoff": _OFFSET_PX}


def offset_benchmark_mark(horizon: str) -> dict[str, Any]:
    """`BENCHMARK_MARK` at a horizon's offset. `line-ns`, because the offset's
    quarter turn would stand `line-ew` upright."""
    return {**BENCHMARK_MARK, "symbol": "line-ns", **horizon_offset(horizon)}


def empty_facet() -> go.Scatter:
    """A placeholder for a configured tier with no rows.

    Measured: Plotly materializes no axis for a facet holding no trace, and
    `add_hline` reaches only facets that hold one, with or without `row="all"`.
    So without this the tier's panel loses its frame and the reference rule
    breaks at it, instead of rendering as the empty panel the grid promises.
    """
    return go.Scatter(x=[], y=[], mode="markers", showlegend=False, hoverinfo="skip")


def base_layout(**overrides: Any) -> dict[str, Any]:
    """The layout Streamlit's theme does not reach, since it does not reach Plotly."""
    layout: dict[str, Any] = {
        "template": "simple_white",
        "paper_bgcolor": palette.PLOT_BGCOLOR,
        "plot_bgcolor": palette.PLOT_BGCOLOR,
        "font": {"color": palette.FONT_COLOR, "size": 12},
        "margin": {"l": 48, "r": 16, "t": 32, "b": 32},
    }
    return layout | overrides


def style_axes(figure: go.Figure) -> None:
    """Grid on the value axis only, and below the traces.

    `layer` is explicit because Plotly defaults the grid above them, where a
    translucent line tints every mark it crosses. `zeroline` is off because a
    metric whose reference is zero already draws it through `add_reference`, and
    two lines at one value read as a rendering fault.
    """
    axis = {"linecolor": palette.AXIS_COLOR, "tickcolor": palette.AXIS_COLOR}
    figure.update_xaxes(showgrid=False, zeroline=False, **axis)
    figure.update_yaxes(
        showgrid=True,
        gridcolor=palette.rgba(palette.GRID_COLOR, palette.GRID_ALPHA),
        layer="below traces",
        zeroline=False,
        **axis,
    )
