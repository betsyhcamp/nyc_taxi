"""The per-metric y-range rule.

Values are the real run's, so the steady-state case is the one the data actually
produces once its single breaching tier is dropped rather than an invented one.
"""

import numpy as np
import plotly.graph_objects as go
import pytest
from plotly.subplots import make_subplots

from dashboard.shared import axes, palette

# The challenger's skill across the six tiers of run 20260929t045735959418z,
# horizon_1, with the one value above 1.0 excluded: the expected steady state.
WINNING_SKILL = (0.8329, 0.9847, 0.9044, 0.8275, 0.8042, 0.8808)
BIAS = (-0.0134, -0.0125, 0.0241, -0.0088, 0.0163, -0.0042)
WAPE = (0.0570, 0.1294, 0.0689, 0.0455, 0.0398, 0.0612)

PAD = 0.06


def _plain_range(values: tuple[float, ...]) -> tuple[float, float]:
    """Min to max with the same padding and no reference forced in."""
    margin = (max(values) - min(values)) * PAD
    return min(values) - margin, max(values) + margin


# ================================================
# the reference, forced in
# ================================================


def test_the_reference_stays_on_the_axis_when_every_tier_wins() -> None:
    """The steady state. Dropped off the chart, the guardrail the row exists to
    show is invisible exactly when the news is good."""
    low, high = axes.metric_range(WINNING_SKILL, "wrmae_pooled")
    assert low <= palette.REFERENCE_LINE["wrmae_pooled"] <= high


def test_plain_padding_would_have_dropped_the_reference() -> None:
    """Self-check on the test above: without forcing, 1.0 really does fall off,
    so that assertion is not passing for free."""
    low, high = _plain_range(WINNING_SKILL)
    assert not low <= palette.REFERENCE_LINE["wrmae_pooled"] <= high


def test_skill_is_not_symmetric() -> None:
    """A working model's skill is one-sided, so symmetry would spend most of the
    axis on a region you hope never to visit."""
    low, high = axes.metric_range(WINNING_SKILL, "wrmae_pooled")
    assert abs(low) != pytest.approx(abs(high))


# ================================================
# the symmetric metric
# ================================================


def test_bias_is_symmetric_about_zero() -> None:
    """Zero at the vertical center of every facet, so the reference reads as one
    continuous mid-line down the row."""
    low, high = axes.metric_range(BIAS, "signed_bias_pooled")
    assert low == pytest.approx(-high)


def test_bias_stays_symmetric_when_every_value_is_one_sided() -> None:
    """One-sided data is where a min-max rule would push zero to an edge."""
    low, high = axes.metric_range([0.01, 0.02, 0.03], "signed_bias_pooled")
    assert low == pytest.approx(-high)
    assert low < 0.0 < high


def test_the_symmetric_bound_follows_whichever_direction_runs_further() -> None:
    """Read off the max alone, an over-forecasting row would clip its own worst
    under-forecast off the bottom of the axis."""
    low, high = axes.metric_range([-0.05, 0.01], "signed_bias_pooled")
    assert high > 0.05
    assert low == pytest.approx(-high)


def test_the_symmetric_bound_covers_the_largest_excursion() -> None:
    """Padded outward from the furthest value, in either direction."""
    low, high = axes.metric_range(BIAS, "signed_bias_pooled")
    assert high > max(abs(value) for value in BIAS)
    assert low < -max(abs(value) for value in BIAS)


# ================================================
# the floor at zero
# ================================================


def test_a_metric_with_no_negative_value_keeps_its_floor_at_zero() -> None:
    """WAPE cannot be negative, so padding into that region invents an area the
    metric can never occupy."""
    low, _ = axes.metric_range([0.004, 0.02], "wape")
    assert low >= 0.0


def test_the_floor_follows_the_data_and_not_the_metric_name() -> None:
    """Clamping on the name would hide a negative value the data really carries."""
    low, _ = axes.metric_range([-0.05, 0.02], "wape")
    assert low < -0.05


def test_the_floor_never_expands_a_range() -> None:
    """A floor, not a bound: skill padded to 0.77 must stay at 0.77."""
    low, _ = axes.metric_range(WINNING_SKILL, "wrmae_pooled")
    assert low > 0.5


# ================================================
# degenerate input
# ================================================


def test_an_all_equal_row_still_gets_a_drawable_range() -> None:
    """Every tier reading the same value is legitimate and must not collapse the
    axis to a single point."""
    low, high = axes.metric_range([1.0, 1.0, 1.0], "wrmae_pooled")
    assert low < 1.0 < high


def test_a_nan_among_finite_values_is_ignored() -> None:
    """A cell the nanmean could not compute must not scale the whole row."""
    assert axes.metric_range([*WAPE, np.nan], "wape") == axes.metric_range(WAPE, "wape")


def test_an_all_nan_row_raises() -> None:
    """Every facet would draw empty with nothing to say why."""
    with pytest.raises(ValueError, match="wape"):
        axes.metric_range([np.nan, np.nan], "wape")


# ================================================
# the reference mark
# ================================================


def _faceted(cols: int) -> go.Figure:
    """A grid with one trace per facet, as a panel builds it.

    Traces first, because `add_hline` reaches only facets that hold one: that is
    why a panel places a placeholder in a facet with no data.
    """
    figure = make_subplots(rows=1, cols=cols, shared_yaxes=True)
    for col in range(1, cols + 1):
        figure.add_trace(go.Bar(x=["h"], y=[0.2], base=[1.0]), row=1, col=col)
    return figure


def test_the_reference_is_drawn_in_every_facet() -> None:
    """Shared y puts it at the same height in each, so it reads as one rule."""
    figure = _faceted(3)
    axes.add_reference(figure, "wrmae_pooled")
    assert len(figure.full_figure_for_development(warn=False).layout.shapes) == 3


def test_a_metric_with_no_reference_draws_none() -> None:
    """WAPE has no meaningful constant, so a line would invent one."""
    figure = _faceted(3)
    axes.add_reference(figure, "wape")
    assert not figure.full_figure_for_development(warn=False).layout.shapes


def test_the_reference_is_not_a_horizon_color() -> None:
    """Color means horizon, so chrome drawn in one would claim to be data."""
    figure = go.Figure()
    axes.add_reference(figure, "wrmae_pooled")
    assert figure.layout.shapes[0].line.color not in set(palette.HORIZON_COLOR.values())


# ================================================
# shared chrome
# ================================================


def test_the_plot_area_is_the_one_the_palette_is_validated_against() -> None:
    """Plotly's default is grey; every contrast figure is measured on white."""
    layout = axes.base_layout()
    assert layout["plot_bgcolor"] == palette.PLOT_BGCOLOR
    assert layout["paper_bgcolor"] == palette.PLOT_BGCOLOR


def test_the_grid_is_drawn_below_the_traces() -> None:
    """Plotly defaults it above, where a translucent line tints every mark."""
    figure = go.Figure(go.Bar(x=["a"], y=[1.0]))
    axes.style_axes(figure)
    assert figure.layout.yaxis.layer == "below traces"


def test_the_zero_line_is_off() -> None:
    """A metric whose reference is zero draws it through `add_reference`, and two
    lines at one value read as a rendering fault."""
    figure = go.Figure(go.Bar(x=["a"], y=[1.0]))
    axes.style_axes(figure)
    assert figure.layout.yaxis.zeroline is False
