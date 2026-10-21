"""The contribution scatter and its top-share tables.

`per_series_comparison` and the hero come from `compute_evaluate_outputs`, so every
row is one the producer emits and the sums are checked against its own table.
"""

import pandas as pd
import plotly.graph_objects as go
import pytest
from streamlit.testing.v1 import AppTest

from dashboard.shared import palette
from dashboard.train import contribution
from fcstnyctaxi.core.train.evaluate_impl import (
    _hero_metric_values,
    compute_evaluate_outputs,
)
from fcstnyctaxi.lib.fold_metrics import compute_wrmae_pooled_shares
from fcstnyctaxi.schemas.run_identity import TrainRunIdentity

CHALLENGER = "model_a"
BENCHMARK = "model_b"
TIER_LABELS = ("low", "middle", "high")

WEEKS = pd.date_range("2025-01-05", periods=17, freq="W-SUN")
WEEKS_IN_MONTH = {202501: 4, 202502: 4, 202503: 5, 202504: 4}
# Two origins in 202502 forecasting it and 202503; the month-end origin forecasts
# 202503 alone, the last complete month, so horizon_2 has no last origin.
ORIGIN_TARGETS = (
    (WEEKS[4], (202502, 202503)),
    (WEEKS[5], (202502, 202503)),
    (WEEKS[7], (202503,)),
)
SERIES = tuple(f"time_series_{letter}" for letter in "abcdefghijklmnopqrstuvwx")
# Revenue rises with the index, so the tiers bin it in thirds. At the last origin
# p overtakes q, and the two trade tiers.
MIGRANT = "time_series_p"
DISPLACED = "time_series_q"
# The largest series, which the challenger misses by exactly the benchmark's error.
EVEN = "time_series_x"


def _tier(series: str, origin_index: int) -> str:
    """The tier `assign_tiers` would give at this origin: thirds by revenue."""
    if origin_index == len(ORIGIN_TARGETS) - 1 and series in (MIGRANT, DISPLACED):
        return "high" if series == MIGRANT else "middle"
    return TIER_LABELS[SERIES.index(series) // 8]


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
    """One sidecar's monthly_series. Errors vary by series and origin, and EVEN's
    challenger error matches the benchmark's in size with the opposite sign."""
    rows = []
    for origin_index, (origin, months) in enumerate(ORIGIN_TARGETS):
        for month in months:
            for index, series in enumerate(SERIES):
                actual = 100.0 * (index + 1) ** 1.5 * (1 + 0.05 * (month - 202502))
                benchmark_error = 0.20 + 0.01 * (index % 4) + 0.02 * origin_index
                if model == BENCHMARK:
                    error = benchmark_error
                elif series == EVEN:
                    error = -benchmark_error
                else:
                    error = 0.10 + 0.02 * (index % 5) + 0.01 * origin_index
                rows.append(
                    {
                        "forecast_origin_date": origin,
                        "predicted_fiscal_year_month": month,
                        "unique_id": series,
                        "tier": _tier(series, origin_index),
                        "series_weight": (index + 1) ** (1 / 3),
                        "monthly_forecast": actual * (1 + error),
                        "actual_monthly_total": actual,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def outputs():
    """The producer's four tables over the two sidecars."""
    identity = TrainRunIdentity(
        git_hash="0" * 40,
        feature_run_id="feature_run",
        train_run_id="train-20250101t000000000000z",
        panel_uri="gs://bucket/panel.parquet",
        calendar_uri="gs://bucket/calendar.parquet",
        additional_exog_uri="gs://bucket/exog.parquet",
    )
    return compute_evaluate_outputs(
        challenger_ms=_monthly_series(CHALLENGER),
        benchmark_ms=_monthly_series(BENCHMARK),
        calendar_df=_calendar(),
        challenger_model=CHALLENGER,
        benchmark_model=BENCHMARK,
        tier_labels=TIER_LABELS,
        identity=identity,
    )


@pytest.fixture(scope="module")
def per_series(outputs) -> pd.DataFrame:
    """The table the panel reads, with its hazards checked."""
    frame = outputs.per_series_comparison
    # Self-check: the migrant starts in another tier than the one it ends in.
    migrant = frame[frame["unique_id"] == MIGRANT].sort_values("forecast_origin_date")
    assert migrant["tier"].iloc[0] != migrant["tier"].iloc[-1]
    # Self-check: by x and by absolute net effect the top tables differ, EVEN
    # being in the first and not the second.
    for horizon in sorted(frame["horizon"].unique()):
        shares = _shares(frame, horizon).set_index("unique_id")
        by_x = set(shares.nlargest(contribution.TOP_N, "x").index)
        by_net = set(
            (shares["y"] - shares["x"]).abs().nlargest(contribution.TOP_N).index
        )
        assert EVEN in by_x - by_net
    return frame


def _shares(frame: pd.DataFrame, horizon: str) -> pd.DataFrame:
    """The decomposition straight off the producer's table, for one horizon."""
    folds = frame[frame["horizon"] == horizon]
    keys = ["forecast_origin_date", "predicted_fiscal_year_month", "unique_id"]
    challenger = folds[keys + ["actual_monthly_total", "series_weight"]].assign(
        monthly_forecast=folds["monthly_forecast_ch"]
    )
    benchmark = folds[keys].assign(monthly_forecast=folds["monthly_forecast_bm"])
    return compute_wrmae_pooled_shares(challenger, benchmark)


def _figure(per_series: pd.DataFrame) -> go.Figure:
    """The scatter over the producer's table."""
    return contribution.contribution_figure(
        contribution.horizon_shares(per_series, TIER_LABELS), TIER_LABELS
    )


def _facets(figure: go.Figure) -> dict[str, list]:
    """Each horizon's marker traces, keyed by the facet's title."""
    titles = [annotation.text for annotation in figure.layout.annotations]
    horizons = [title for title in titles if title.startswith("horizon_")]
    axis_of = {
        "x" if index == 0 else f"x{index + 1}": horizon
        for index, horizon in enumerate(horizons)
    }
    facets: dict[str, list] = {horizon: [] for horizon in horizons}
    for trace in figure.data:
        facets[axis_of[trace.xaxis]].append(trace)
    return facets


# ================================================
# the sums
# ================================================


def test_the_sums_read_one_and_the_hero_of_each_horizon(outputs, per_series) -> None:
    """Σy from the decomposition and the hero from the summary table must agree."""
    hero = _hero_metric_values(outputs.summary_metrics, CHALLENGER)
    # Self-check: the horizons' heroes differ, so a swapped facet cannot pass.
    assert f"{hero['horizon_1']:.3f}" != f"{hero['horizon_2']:.3f}"

    sums = [
        annotation.text
        for annotation in _figure(per_series).layout.annotations
        if annotation.name == contribution.SUMS_ANNOTATION
    ]
    assert sums == [
        f"Σx = 1.000   Σy = {hero[horizon]:.3f}" for horizon in sorted(hero)
    ]


# ================================================
# the axes and the traces
# ================================================


def test_the_facets_are_square_and_zoom_together(per_series) -> None:
    """Without 1:1, y = x is not 45 degrees and distance from it is not magnitude."""
    layout = _figure(per_series).layout
    assert layout.yaxis.scaleanchor == "x"
    assert layout.yaxis.scaleratio == 1
    assert layout.xaxis2.matches == "x"
    assert layout.yaxis2.matches == "y"


def test_each_facet_draws_one_scatter_trace_per_tier(per_series) -> None:
    """Per tier, not per series, and `Scatter`, whose symbols carry the tier."""
    last_tiers = set(
        per_series.sort_values("forecast_origin_date")
        .drop_duplicates("unique_id", keep="last")["tier"]
        .astype(str)
    )
    for traces in _facets(_figure(per_series)).values():
        assert all(trace.type == "scatter" for trace in traces)
        assert sorted(trace.name for trace in traces) == sorted(last_tiers)


def test_a_migrating_series_keeps_its_last_tier_in_both_facets(per_series) -> None:
    """One symbol per series; its label names the other tier and its fold count."""
    for horizon, traces in _facets(_figure(per_series)).items():
        (trace,) = [trace for trace in traces if MIGRANT in trace.customdata[:, 0]]
        row = list(trace.customdata[:, 0]).index(MIGRANT)

        folds = per_series[
            (per_series["unique_id"] == MIGRANT) & (per_series["horizon"] == horizon)
        ]
        held_middle = int((folds["tier"].astype(str) == "middle").sum())
        assert trace.name == "high"
        assert trace.customdata[row, 1] == (
            f"high (also middle, {held_middle} of {len(folds)} folds)"
        )


# ================================================
# the tables
# ================================================


def test_each_table_holds_its_horizons_largest_benchmark_shares(per_series) -> None:
    """By x, largest first: a large share with y near x is in, not ranked out."""
    shares = contribution.horizon_shares(per_series, TIER_LABELS)
    for horizon, frame in shares.items():
        table = contribution.top_table(frame)
        expected = _shares(per_series, horizon).nlargest(contribution.TOP_N, "x")
        assert list(table["series"]) == list(expected["unique_id"])
        assert table["x"].is_monotonic_decreasing
        assert EVEN in set(table["series"])


# Unannotated: `AppTest.from_function` runs this function's source as a script of
# its own, where the annotations are evaluated and `pd` is undefined.
def _contribution_page(per_series, tier_labels):
    """The panel alone, as a script `AppTest` can run."""
    from dashboard.train import contribution

    contribution.render_contribution(per_series, tier_labels)


def test_an_unsupported_tier_count_shows_a_message_and_the_tables(per_series) -> None:
    """No validated encoding for six tiers, which costs the scatter, not the tables."""
    six = (*TIER_LABELS, "tier_d", "tier_e", "tier_f")
    app = AppTest.from_function(_contribution_page, args=(per_series, six)).run()
    assert not app.exception, app.exception[0].message if app.exception else ""
    assert not app.get("plotly_chart")
    (warning,) = app.warning
    assert "6 tiers" in warning.value
    assert f"{min(palette.TIER_RAMPS)} to {max(palette.TIER_RAMPS)}" in warning.value
    assert len(app.dataframe) == len(per_series["horizon"].unique())
