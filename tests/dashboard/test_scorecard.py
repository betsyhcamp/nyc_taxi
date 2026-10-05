"""The two bar rows.

`summary_metrics` is derived from `fold_metrics` by the producer's own `_derive`,
with one fold per cell so the mean equals the planted value exactly. That gives
per-cell control over a frame the producer really emits.
"""

from collections.abc import Sequence

import pandas as pd
import pytest

from dashboard.shared import palette
from dashboard.train import scorecard
from fcstnyctaxi.core.train.evaluate_impl import _SCORE_KEYS, _derive

CHALLENGER = "model_a"
BENCHMARK = "model_b"
HORIZONS = ("horizon_1", "horizon_2")

# A tier that wins and a tier that breaches in BOTH horizons, which the
# no-recoloring invariant needs: a breach in one horizon only cannot catch a
# recolor that reaches for the other horizon's hue. The real run carries the
# shape, very_low at 1.0362.
WINNING_TIER = "low"
BREACHING_TIER = "very_low"
# Configured but given no rows, so its facet must render empty.
ABSENT_TIER = "very_high"
TIER_LABELS = (BREACHING_TIER, WINNING_TIER, ABSENT_TIER)

_PLANTED = {
    (scorecard.GLOBAL_TIER, "horizon_1"): 0.83,
    (scorecard.GLOBAL_TIER, "horizon_2"): 0.81,
    (BREACHING_TIER, "horizon_1"): 1.02,
    (BREACHING_TIER, "horizon_2"): 1.04,
    (WINNING_TIER, "horizon_1"): 0.90,
    (WINNING_TIER, "horizon_2"): 0.84,
}


def _summary(metric: str) -> pd.DataFrame:
    """One cell per (model, horizon, tier) for `metric`, through `_derive`."""
    rows = []
    for model in (CHALLENGER, BENCHMARK):
        for (tier, horizon), value in _PLANTED.items():
            rows.append(
                {
                    "model": model,
                    "horizon": horizon,
                    "tier": tier,
                    "forecast_origin_date": pd.Timestamp("2025-03-23"),
                    "predicted_fiscal_year_month": 202504,
                    "metric": metric,
                    # The benchmark scores itself, so its relative metric reads
                    # 1.0 while its absolute metric reads a level of its own.
                    "value": value if model == CHALLENGER else value + 0.1,
                    "n_obs": 2040,
                }
            )
    derived = _derive(pd.DataFrame(rows), _SCORE_KEYS)
    # Self-check: one fold per cell, so the mean is the planted value and a
    # value asserted below is the one planted rather than an average of two.
    assert set(derived["n_folds_used"]) == {1}
    assert derived["tier"].nunique() == len({tier for tier, _ in _PLANTED})
    return derived


def _bars(figure: object) -> list:
    """The challenger bars, which carry a `base`."""
    return [trace for trace in figure.data if trace.type == "bar" and len(trace.x)]


def _facet_count(figure: object) -> int:
    """How many facets materialized, which an absent tier must not reduce."""
    full = figure.full_figure_for_development(warn=False)
    return len([key for key in full.layout if str(key).startswith("xaxis")])


def _row(metric: str, tier_labels: Sequence[str] = TIER_LABELS):
    """One bar row over the planted frame."""
    return scorecard.bar_row(
        _summary(metric), metric, tier_labels, CHALLENGER, BENCHMARK
    )


# ================================================
# the grid
# ================================================


def test_the_grid_comes_from_the_configured_tiers() -> None:
    """`global` plus every configured label, whether or not it has rows."""
    assert _facet_count(_row("wrmae_pooled")) == 1 + len(TIER_LABELS)


def test_a_configured_tier_with_no_rows_still_renders_a_panel() -> None:
    """Plotly materializes no axis for a facet holding no trace, so without a
    placeholder the tier's panel loses its frame and the grid reads as shifted."""
    figure = _row("wrmae_pooled")
    placeholders = [
        trace for trace in figure.data if trace.type == "bar" and not len(trace.x)
    ]
    assert len(placeholders) == 1


def test_the_reference_reaches_the_empty_facet_too() -> None:
    """Shared y puts 1.0 at one height, so a gap in the rule reads as a defect."""
    figure = _row("wrmae_pooled")
    shapes = figure.full_figure_for_development(warn=False).layout.shapes
    assert len(shapes) == 1 + len(TIER_LABELS)


def test_the_absent_tier_draws_no_bar() -> None:
    """An empty panel, not a zero: a zero would read as a measured value."""
    drawn = {trace.x[0] for trace in _bars(_row("wrmae_pooled"))}
    assert drawn == set(HORIZONS)


# ================================================
# what the bars encode
# ================================================


def test_only_the_challenger_is_drawn_as_a_bar() -> None:
    """The benchmark scores itself at 1.0; a bar for it would halve the row."""
    bars = _bars(_row("wrmae_pooled"))
    assert len(bars) == 2 * len({tier for tier, _ in _PLANTED})


def test_bars_grow_from_the_guardrail_not_from_zero() -> None:
    """From a zero baseline 0.97 and 1.03 look 5.8% apart while sitting on
    opposite sides of the guardrail."""
    assert {trace.base[0] for trace in _bars(_row("wrmae_pooled"))} == {1.0}


def test_an_absolute_metric_grows_from_zero() -> None:
    """WAPE has no guardrail to grow from, so its baseline is the axis floor."""
    assert {trace.base[0] for trace in _bars(_row("wape"))} == {0.0}


def test_the_real_value_survives_the_delta_encoding() -> None:
    """Plotly reads `y` as length from `base`, so hover would show the delta."""
    for trace in _bars(_row("wrmae_pooled")):
        assert trace.customdata[0][0] == pytest.approx(trace.base[0] + trace.y[0])


def test_a_breach_is_not_recolored() -> None:
    """Color means horizon everywhere. Recoloring would destroy horizon identity
    exactly where the guardrail failed, which is where it is needed most."""
    by_horizon: dict[str, set[str]] = {}
    for trace in _bars(_row("wrmae_pooled")):
        by_horizon.setdefault(trace.x[0], set()).add(trace.marker.color)
    assert all(len(colors) == 1 for colors in by_horizon.values())


def test_the_bars_carry_the_challengers_values_not_the_benchmarks() -> None:
    """The two models differ by a constant here, so reading the wrong one shows a
    complete and plausible row of the wrong model's numbers."""
    drawn = {
        (trace.x[0], round(trace.customdata[0][0], 6)) for trace in _bars(_row("wape"))
    }
    planted = {(horizon, round(value, 6)) for (_, horizon), value in _PLANTED.items()}
    assert {value for _, value in drawn} <= {value for _, value in planted}


def test_each_horizon_keeps_its_own_color() -> None:
    """The one thing color does carry, so the two must not collapse."""
    colors = {trace.x[0]: trace.marker.color for trace in _bars(_row("wrmae_pooled"))}
    assert colors == {
        horizon: palette.rgba(palette.HORIZON_COLOR[horizon]) for horizon in HORIZONS
    }


def test_the_legend_names_each_horizon_exactly_once() -> None:
    """The tick labels are hidden, so the legend is the only thing identifying
    which bar is which horizon, and six facets must not repeat it six times."""
    named = [trace.name for trace in _bars(_row("wrmae_pooled")) if trace.showlegend]
    assert sorted(named) == sorted(HORIZONS)


def test_every_bar_is_annotated_with_its_n_obs() -> None:
    """Accurate for all three metrics, each guarding at fold level."""
    assert all(trace.text[0] == "2,040" for trace in _bars(_row("wrmae_pooled")))


# ================================================
# the benchmark mark
# ================================================


def test_an_absolute_metric_carries_a_benchmark_tick() -> None:
    """A 12% WAPE is uninterpretable without one."""
    ticks = [trace for trace in _row("wape").data if trace.type == "scatter"]
    assert len(ticks) == len(_bars(_row("wape")))


def test_a_relative_metric_carries_none() -> None:
    """Its benchmark reads ~1.0 by construction, which the reference already is."""
    assert not [trace for trace in _row("wrmae_pooled").data if trace.type == "scatter"]


def test_the_benchmark_tick_is_not_a_horizon_color() -> None:
    """Chrome drawn in a horizon color would claim to be that horizon's data."""
    ticks = [trace for trace in _row("wape").data if trace.type == "scatter"]
    colors = {trace.marker.line.color for trace in ticks}
    assert not colors & set(palette.HORIZON_COLOR.values())


def test_every_benchmark_tick_lands_inside_the_axis() -> None:
    """Scaled to the challenger alone, a benchmark above it is clipped away and
    the absolute metric loses the reference that makes it readable."""
    figure = _row("wape")
    full = figure.full_figure_for_development(warn=False)
    low, high = full.layout.yaxis.range
    ticks = [trace for trace in figure.data if trace.type == "scatter"]
    assert ticks
    assert all(low <= trace.y[0] <= high for trace in ticks)


# ================================================
# the y-axis
# ================================================


def test_the_y_range_is_shared_across_the_facets_of_a_row() -> None:
    """So the reference lands at one height and bar heights are comparable."""
    full = _row("wrmae_pooled").full_figure_for_development(warn=False)
    ranges = {
        tuple(full.layout[key].range)
        for key in full.layout
        if str(key).startswith("yaxis") and full.layout[key].range is not None
    }
    assert len(ranges) == 1


def test_the_two_rows_do_not_share_a_range() -> None:
    """Never across rows: skill near 1.0 and bias near 0 on one axis would flatten
    whichever row is narrower."""
    skill = _row("wrmae_pooled").full_figure_for_development(warn=False)
    bias = _row("signed_bias_pooled").full_figure_for_development(warn=False)
    assert tuple(skill.layout.yaxis.range) != tuple(bias.layout.yaxis.range)
