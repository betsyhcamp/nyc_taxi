"""Skill by forecast origin.

`fold_metrics` comes from `compute_evaluate_outputs` over two synthetic sidecars, so
every row is one the producer emits, nan folds included.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from test_palette import MARK_CONTRAST_GATE, contrast_ratio

from dashboard.shared import axes, palette
from dashboard.train import origin_skill
from fcstnyctaxi.core.train.evaluate_impl import GLOBAL_TIER, compute_evaluate_outputs
from fcstnyctaxi.schemas.run_identity import TrainRunIdentity

CHALLENGER = "model_a"
BENCHMARK = "model_b"
# The data bins the first three, so `high` is configured with no rows.
TIER_LABELS = ("very_low", "low", "middle", "high")

WEEKS = pd.date_range("2025-01-05", periods=17, freq="W-SUN")
WEEKS_IN_MONTH = {202501: 4, 202502: 4, 202503: 5, 202504: 4}
# Two origins in 202502 a fortnight apart, forecasting it and 202503. The month-end
# origin and one inside 202503 forecast 202503 alone, the last complete month.
ORIGIN_TARGETS = (
    (WEEKS[4], (202502, 202503)),
    (WEEKS[6], (202502, 202503)),
    (WEEKS[7], (202503,)),
    (WEEKS[8], (202503,)),
)
LATEST = WEEKS[8]
SERIES = ("time_series_a", "time_series_b", "time_series_c")
# time_series_c is dormant at the first origin, so two series bin into two tiers and
# it is forced into the lowest, leaving `middle` empty there: a nan fold.
TIER_INDEX = {
    "time_series_a": (1, 2, 2, 2),
    "time_series_b": (0, 1, 1, 1),
    "time_series_c": (0, 0, 0, 0),
}
WEIGHT = {
    "time_series_a": (30.0, 30.0, 30.0, 30.0),
    "time_series_b": (17.0, 17.0, 17.0, 17.0),
    "time_series_c": (0.0, 6.0, 6.0, 6.0),
}


def _calendar() -> pd.DataFrame:
    """The calendar columns evaluate reads, each week's fraction of its month."""
    months = [month for month, weeks in WEEKS_IN_MONTH.items() for _ in range(weeks)]
    elapsed = [
        week / weeks
        for weeks in WEEKS_IN_MONTH.values()
        for week in range(1, weeks + 1)
    ]
    return pd.DataFrame(
        {
            "ds": WEEKS,
            "fiscal_year_month": months,
            "origin_month_fraction_elapsed": elapsed,
        }
    )


def _monthly_series(model: str) -> pd.DataFrame:
    """One sidecar's monthly_series. The benchmark errs by 20% and up, the challenger
    by 8% and up, varying by origin and series so no two folds score alike."""
    base_error = 0.20 if model == BENCHMARK else 0.08
    rows = []
    for origin_index, (origin, months) in enumerate(ORIGIN_TARGETS):
        for month in months:
            for index, series in enumerate(SERIES):
                actual = 100.0 * (3 - index) ** 2 * (1 + 0.05 * (month - 202502))
                error = base_error + 0.03 * origin_index + 0.02 * index
                rows.append(
                    {
                        "forecast_origin_date": origin,
                        "predicted_fiscal_year_month": month,
                        "unique_id": series,
                        "tier": TIER_LABELS[TIER_INDEX[series][origin_index]],
                        "series_weight": WEIGHT[series][origin_index],
                        "monthly_forecast": actual * (1 + (-1) ** index * error),
                        "actual_monthly_total": actual,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def fold_metrics() -> pd.DataFrame:
    """Both models' fold table, through the producer."""
    identity = TrainRunIdentity(
        git_hash="0" * 40,
        feature_run_id="feature_run",
        train_run_id="train-20250101t000000000000z",
        panel_uri="gs://bucket/panel.parquet",
        calendar_uri="gs://bucket/calendar.parquet",
        additional_exog_uri="gs://bucket/exog.parquet",
    )
    frame = compute_evaluate_outputs(
        challenger_ms=_monthly_series(CHALLENGER),
        benchmark_ms=_monthly_series(BENCHMARK),
        calendar_df=_calendar(),
        challenger_model=CHALLENGER,
        benchmark_model=BENCHMARK,
        tier_labels=TIER_LABELS,
        identity=identity,
    ).fold_metrics

    # Self-check: each hazard a test below relies on is in the data.
    assert (_gaps(frame["forecast_origin_date"]) == pd.Timedelta(days=14)).any()
    challenger = frame[frame["model"] == CHALLENGER]
    assert challenger.loc[challenger["tier"] != GLOBAL_TIER, "value"].isna().any()
    folds = challenger.drop_duplicates(["forecast_origin_date", "horizon"])
    assert (folds["horizon"] == "horizon_2").sum() < (
        folds["horizon"] == "horizon_1"
    ).sum()
    assert set(TIER_LABELS) - set(frame["tier"].astype(str))
    return frame


def _gaps(origins: object) -> pd.Series:
    """The spacing between consecutive distinct origins."""
    return pd.Series(sorted(set(pd.to_datetime(list(origins))))).diff().dropna()


def _figure(fold_metrics: pd.DataFrame, metric: str) -> go.Figure:
    """The panel over the producer's fold table."""
    return origin_skill.origin_figure(
        fold_metrics, metric, TIER_LABELS, CHALLENGER, BENCHMARK
    )


def _marks(figure: go.Figure) -> list:
    """The challenger's traces, one per horizon per facet."""
    return [trace for trace in figure.data if trace.name in palette.HORIZON_COLOR]


def _ticks(figure: go.Figure) -> list:
    """The benchmark's traces, drawn only for a metric that shows it."""
    return [trace for trace in figure.data if trace.name == "benchmark"]


def _rings(figure: go.Figure) -> list:
    """The traces ringing the latest origin."""
    return [trace for trace in figure.data if trace.name == origin_skill.LATEST_RING]


def _y_ranges(figure: go.Figure) -> set[tuple[float, float]]:
    """Every facet's resolved y range."""
    full = figure.full_figure_for_development(warn=False)
    return {
        tuple(full.layout[key].range)
        for key in full.layout
        if str(key).startswith("yaxis")
    }


def _gutter(figure: go.Figure, *, counts: bool) -> list[str]:
    """The left gutter's text in row order: tier names, or their fold counts."""
    labels = [
        annotation
        for annotation in figure.layout.annotations
        if (annotation.name == origin_skill.COUNTS_ANNOTATION) == counts
    ]
    return [annotation.text for annotation in labels]


# ================================================
# the axes
# ================================================


def test_a_gap_between_origins_stays_on_the_axis(fold_metrics: pd.DataFrame) -> None:
    """A date scale keeps a missing origin's 14 days, which an index would hide."""
    trace = _marks(_figure(fold_metrics, "wrmae_pooled"))[0]
    assert (_gaps(trace.x) == pd.Timedelta(days=14)).any()


def test_the_rows_are_global_then_each_configured_tier(
    fold_metrics: pd.DataFrame,
) -> None:
    """From the configured labels, not the ones present, so no row shifts."""
    assert _gutter(_figure(fold_metrics, "wrmae_pooled"), counts=False) == [
        GLOBAL_TIER,
        *TIER_LABELS,
    ]


def test_an_unpopulated_tier_draws_a_facet_the_reference_reaches(
    fold_metrics: pd.DataFrame,
) -> None:
    """Without a placeholder the facet loses its frame and the rule its segment."""
    full = _figure(fold_metrics, "wrmae_pooled").full_figure_for_development(warn=False)
    assert len(full.layout.shapes) == 1 + len(TIER_LABELS)


def test_one_y_range_spans_every_facet(fold_metrics: pd.DataFrame) -> None:
    """So the reference sits at one height down the whole column."""
    assert len(_y_ranges(_figure(fold_metrics, "wrmae_pooled"))) == 1


def test_the_range_covers_the_benchmark_where_it_is_drawn(
    fold_metrics: pd.DataFrame,
) -> None:
    """Scaled to the challenger alone, the benchmark ticks above it are clipped."""
    benchmark = fold_metrics[
        (fold_metrics["model"] == BENCHMARK) & (fold_metrics["metric"] == "wape")
    ]
    challenger = fold_metrics[
        (fold_metrics["model"] == CHALLENGER) & (fold_metrics["metric"] == "wape")
    ]
    # Self-check: the benchmark runs above the challenger, so the range must reach it.
    assert benchmark["value"].max() > challenger["value"].max()

    ((_, high),) = _y_ranges(_figure(fold_metrics, "wape"))
    assert high >= benchmark["value"].max()


def test_the_range_ignores_the_benchmark_where_it_is_hidden(
    fold_metrics: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drawn or not is the one rule, so a hidden benchmark must not widen the axis."""
    monkeypatch.setitem(palette.SHOW_BENCHMARK, "wape", False)
    challenger = fold_metrics[
        (fold_metrics["model"] == CHALLENGER) & (fold_metrics["metric"] == "wape")
    ]
    figure = _figure(fold_metrics, "wape")
    assert not _ticks(figure)
    assert _y_ranges(figure) == {axes.metric_range(challenger["value"], "wape")}


# ================================================
# the marks
# ================================================


def test_only_the_latest_origin_is_ringed(fold_metrics: pd.DataFrame) -> None:
    """The ring marks where the run ends; on every origin it marks nothing."""
    rings = _rings(_figure(fold_metrics, "wrmae_pooled"))
    assert rings
    assert {pd.Timestamp(x) for ring in rings for x in ring.x} == {LATEST}


def test_each_ring_is_drawn_beneath_its_marker(fold_metrics: pd.DataFrame) -> None:
    """Drawn above, the ring's white fill would hide the marker it rings."""
    data = _figure(fold_metrics, "wrmae_pooled").data
    marks_at = {
        (trace.legendgroup, trace.yaxis): position
        for position, trace in enumerate(data)
        if trace.name in palette.HORIZON_COLOR
    }
    rings = [
        (position, trace)
        for position, trace in enumerate(data)
        if trace.name == origin_skill.LATEST_RING
    ]
    assert rings
    for position, ring in rings:
        assert marks_at[(ring.legendgroup, ring.yaxis)] > position


def test_the_ring_is_visible_against_the_plot_area(fold_metrics: pd.DataFrame) -> None:
    """A ring in the plot area's own white marks the latest origin only by size."""
    rings = _rings(_figure(fold_metrics, "wrmae_pooled"))
    assert rings
    assert all(
        contrast_ratio(ring.marker.line.color, palette.PLOT_BGCOLOR)
        >= MARK_CONTRAST_GATE
        for ring in rings
    )


def test_a_halo_of_plot_area_white_parts_the_ring_from_its_marker(
    fold_metrics: pd.DataFrame,
) -> None:
    """With no gap the ring reads as a darker marker, not as a mark around one."""
    figure = _figure(fold_metrics, "wrmae_pooled")
    largest_marker = max(max(trace.marker.size) for trace in _marks(figure))
    rings = _rings(figure)
    assert rings
    for ring in rings:
        inner_edge = (ring.marker.size - ring.marker.line.width) / 2
        assert ring.marker.color == palette.PLOT_BGCOLOR
        assert inner_edge - largest_marker / 2 >= 1


def test_every_x_axis_is_a_date_axis(fold_metrics: pd.DataFrame) -> None:
    """Plotly types an axis by its data, and an empty facet's axis has none."""
    full = _figure(fold_metrics, "wrmae_pooled").full_figure_for_development(warn=False)
    types = {
        full.layout[key].type for key in full.layout if str(key).startswith("xaxis")
    }
    assert types == {"date"}


def test_the_two_horizons_sit_on_opposite_sides_of_the_origin(
    fold_metrics: pd.DataFrame,
) -> None:
    """Each horizon 5px to its own side, its tick and ring with it; a shared side
    stacks the marks, and a vertical offset moves them off their values."""
    figure = _figure(fold_metrics, "wape")
    offsets = {}
    for horizon in palette.HORIZON_COLOR:
        marks = [trace for trace in _marks(figure) if trace.name == horizon]
        ticks = [
            trace
            for trace in _ticks(figure)
            if trace.hovertemplate.startswith(f"benchmark {horizon}<")
        ]
        rings = [trace for trace in _rings(figure) if trace.legendgroup == horizon]
        assert marks and ticks
        offsets[horizon] = {
            (trace.marker.angle, trace.marker.standoff)
            for trace in marks + ticks + rings
        }
    assert _rings(figure)
    assert all(len(offset) == 1 for offset in offsets.values())
    ((angle_1, standoff_1),) = offsets["horizon_1"]
    ((angle_2, standoff_2),) = offsets["horizon_2"]
    # A literal, not _OFFSET_PX: the spec fixes 5px, and a test reading the
    # constant would carry a mistyped one along.
    assert standoff_1 == standoff_2 == 5
    assert angle_1 % 180 == angle_2 % 180 == 90
    assert (angle_1 - angle_2) % 360 == 180


def test_a_metric_without_a_drawn_benchmark_has_no_ticks(
    fold_metrics: pd.DataFrame,
) -> None:
    """Skill's benchmark reads 1.0 by construction, which the reference already is."""
    assert not _ticks(_figure(fold_metrics, "wrmae_pooled"))
    assert _ticks(_figure(fold_metrics, "wape"))


# ================================================
# the fold counts
# ================================================


def test_each_facets_counts_are_its_finite_folds_per_horizon(
    fold_metrics: pd.DataFrame,
) -> None:
    """What each facet's marks average over, so a nan fold lowers its count."""
    challenger = fold_metrics[
        (fold_metrics["model"] == CHALLENGER)
        & (fold_metrics["metric"] == "wrmae_pooled")
    ]
    finite = challenger[np.isfinite(challenger["value"])]
    horizons = sorted(challenger["horizon"].unique())
    expected = [
        " / ".join(
            str(((finite["tier"] == tier) & (finite["horizon"] == horizon)).sum())
            for horizon in horizons
        )
        for tier in (GLOBAL_TIER, *TIER_LABELS)
    ]
    assert _gutter(_figure(fold_metrics, "wrmae_pooled"), counts=True) == expected
