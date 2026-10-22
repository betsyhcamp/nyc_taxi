"""The contribution scatter and its top-share tables.

`per_series_comparison` and the hero come from `compute_evaluate_outputs`, so every
row is one the producer emits and the sums are checked against its own table.
"""

import json

import numpy as np
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
# With no trailing revenue, weighted 0 at every origin, so both its shares are 0.
DORMANT = "time_series_a"


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


def _evaluate(challenger_ms: pd.DataFrame, benchmark_ms: pd.DataFrame):
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
        challenger_ms=challenger_ms,
        benchmark_ms=benchmark_ms,
        calendar_df=_calendar(),
        challenger_model=CHALLENGER,
        benchmark_model=BENCHMARK,
        tier_labels=TIER_LABELS,
        identity=identity,
    )


@pytest.fixture(scope="module")
def outputs():
    """The producer's four tables over the two sidecars."""
    return _evaluate(_monthly_series(CHALLENGER), _monthly_series(BENCHMARK))


@pytest.fixture(scope="module")
def perfect_horizon_1(outputs) -> pd.DataFrame:
    """The table again, the benchmark exact on every horizon_1 fold, so every one
    is excluded and horizon_1 has no fold to average."""
    keys = ["forecast_origin_date", "predicted_fiscal_year_month", "unique_id"]
    table = outputs.per_series_comparison
    horizon_1 = table.loc[table["horizon"] == "horizon_1", keys]
    benchmark = _monthly_series(BENCHMARK)
    exact = (
        benchmark.merge(horizon_1, on=keys, how="left", indicator=True)["_merge"]
        .eq("both")
        .to_numpy()
    )
    benchmark.loc[exact, "monthly_forecast"] = benchmark.loc[
        exact, "actual_monthly_total"
    ]
    rerun = _evaluate(_monthly_series(CHALLENGER), benchmark)
    # Self-check: horizon_1's hero is nan and horizon_2's is not.
    hero = _hero_metric_values(rerun.summary_metrics, CHALLENGER)
    assert np.isnan(hero["horizon_1"])
    assert np.isfinite(hero["horizon_2"])
    return rerun.per_series_comparison


@pytest.fixture(scope="module")
def dormant() -> pd.DataFrame:
    """The table again, DORMANT shaped like a real dormant series: no revenue, weight
    0, a naive forecast of 0, and a challenger forecast that is not."""
    sidecars = []
    for model in (CHALLENGER, BENCHMARK):
        frame = _monthly_series(model)
        rows = frame["unique_id"] == DORMANT
        frame.loc[rows, ["series_weight", "actual_monthly_total"]] = 0.0
        if model == BENCHMARK:
            frame.loc[rows, "monthly_forecast"] = 0.0
        sidecars.append(frame)
    table = _evaluate(*sidecars).per_series_comparison
    # Self-check: in each horizon DORMANT alone has a zero share.
    for horizon in sorted(table["horizon"].unique()):
        shares = _shares(table, horizon)
        zero = shares[(shares["x"] == 0) | (shares["y"] == 0)]
        assert list(zero["unique_id"]) == [DORMANT]
    return table


@pytest.fixture(scope="module")
def exact_low(per_series) -> pd.DataFrame:
    """The table again, the challenger exact on every horizon_1 row of the lowest
    tier, so on log-log that tier draws in the horizon_2 facet alone."""
    keys = ["forecast_origin_date", "predicted_fiscal_year_month", "unique_id"]
    horizon_1 = per_series.loc[per_series["horizon"] == "horizon_1", keys]
    challenger = _monthly_series(CHALLENGER)
    in_horizon_1 = challenger.merge(horizon_1, on=keys, how="left", indicator=True)
    exact = (in_horizon_1["_merge"] == "both").to_numpy() & (
        challenger["tier"] == TIER_LABELS[0]
    ).to_numpy()
    challenger.loc[exact, "monthly_forecast"] = challenger.loc[
        exact, "actual_monthly_total"
    ]
    table = _evaluate(challenger, _monthly_series(BENCHMARK)).per_series_comparison
    # Self-check: the lowest tier's horizon_1 shares are all zero, and some of its
    # horizon_2 shares are drawable on log axes.
    shares = contribution.horizon_shares(table, TIER_LABELS)
    first, second = (
        shares[horizon][shares[horizon]["tier"] == TIER_LABELS[0]]
        for horizon in ("horizon_1", "horizon_2")
    )
    assert (first["y"] == 0).all()
    assert ((second["x"] > 0) & (second["y"] > 0)).any()
    return table


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


def _figure(per_series: pd.DataFrame, *, log_log: bool = False) -> go.Figure:
    """The scatter over the producer's table."""
    return contribution.contribution_figure(
        contribution.horizon_shares(per_series, TIER_LABELS),
        TIER_LABELS,
        log_log=log_log,
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


def test_a_horizon_with_no_included_fold_reads_nan(perfect_horizon_1) -> None:
    """Not 0.000, a reconciliation that is not there; the other facet's y = x stays."""
    figure = contribution.contribution_figure(
        contribution.horizon_shares(perfect_horizon_1, TIER_LABELS), TIER_LABELS
    )
    sums = [
        annotation.text
        for annotation in figure.layout.annotations
        if annotation.name == contribution.SUMS_ANNOTATION
    ]
    assert sums[0] == "Σx = nan   Σy = nan"
    assert all(
        np.isfinite([shape.x1, shape.y1]).all() for shape in figure.layout.shapes
    )


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
            f"high at last origin (middle in {held_middle} of {len(folds)} folds)"
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


# ================================================
# the scale toggle
# ================================================


def test_the_axes_open_on_linear(per_series) -> None:
    """Linear answers the panel's question: which series moved the headline."""
    app = AppTest.from_function(_contribution_page, args=(per_series, TIER_LABELS))
    app.run()
    assert not app.exception, app.exception[0].message if app.exception else ""
    assert app.radio(key=contribution.SCALE_KEY).value == "linear"
    (chart,) = app.get("plotly_chart")
    layout = json.loads(chart.proto.spec)["layout"]
    axis_types = [
        axis.get("type")
        for name, axis in layout.items()
        if name.startswith(("xaxis", "yaxis"))
    ]
    assert axis_types
    assert "log" not in axis_types


def test_log_log_puts_every_axis_on_log_and_keeps_the_aspect(per_series) -> None:
    """Both facets, both axes; y = x stays square and starts above zero."""
    figure = _figure(per_series, log_log=True)
    axis_types = [
        axis.type for axis in (*figure.select_xaxes(), *figure.select_yaxes())
    ]
    assert axis_types
    assert set(axis_types) == {"log"}
    layout = figure.layout
    assert layout.yaxis.scaleanchor == "x"
    assert layout.yaxis.scaleratio == 1
    assert layout.xaxis2.matches == "x"
    assert layout.yaxis2.matches == "y"
    for shape in layout.shapes:
        ends = np.array([shape.x0, shape.y0, shape.x1, shape.y1], dtype=float)
        assert np.isfinite(ends).all() and (ends > 0).all()


def test_log_log_leaves_out_zero_shares_and_counts_them(dormant) -> None:
    """A log axis cannot draw a zero, so each facet's note counts what it drops."""
    figure = _figure(dormant, log_log=True)
    for traces in _facets(figure).values():
        drawn = {series for trace in traces for series in trace.customdata[:, 0]}
        assert DORMANT not in drawn

    notes = [
        annotation.text
        for annotation in figure.layout.annotations
        if annotation.name == contribution.ZEROS_ANNOTATION
    ]
    zero_shares = []
    for horizon in sorted(dormant["horizon"].unique()):
        shares = _shares(dormant, horizon)
        zero_shares.append(int(((shares["x"] == 0) | (shares["y"] == 0)).sum()))
    assert [int(note.split()[0]) for note in notes] == zero_shares


def test_a_tier_drawn_in_horizon_2_alone_enters_the_legend_in_order(
    exact_low,
) -> None:
    """Its legend entry moves to the first trace drawn, and keeps its tier's place."""
    figure = _figure(exact_low, log_log=True)
    shown = [trace for trace in figure.data if trace.showlegend]
    assert sorted(trace.name for trace in shown) == sorted(TIER_LABELS)
    rank = {trace.name: trace.legendrank for trace in shown}
    ranks = [rank[tier] for tier in TIER_LABELS]
    assert None not in ranks
    assert ranks == sorted(set(ranks))


def test_a_nan_share_is_not_noted_as_a_zero(perfect_horizon_1) -> None:
    """A horizon with no included fold has nan shares, not zeros, so it gets no note."""
    figure = _figure(perfect_horizon_1, log_log=True)
    names = {annotation.name for annotation in figure.layout.annotations}
    assert contribution.ZEROS_ANNOTATION not in names


@pytest.mark.parametrize(
    ("scale", "reads", "never_reads"),
    [("linear", "net effect", "ratio"), ("log-log", "ratio", "net effect")],
)
def test_the_caption_reads_the_diagonal_by_scale(
    per_series, scale: str, reads: str, never_reads: str
) -> None:
    """Distance from y = x is the net effect on linear axes, the ratio on log-log."""
    app = AppTest.from_function(_contribution_page, args=(per_series, TIER_LABELS))
    app.run()
    app.radio(key=contribution.SCALE_KEY).set_value(scale).run()
    assert not app.exception, app.exception[0].message if app.exception else ""
    (caption,) = [caption.value for caption in app.caption if "y = x" in caption.value]
    assert reads in caption
    assert never_reads not in caption
