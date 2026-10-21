"""Position cohorts and their strip, over the fold metrics evaluate emits.

The calendar is hand-built, since the Feature pipeline owns the real one; its
self-check is that `enumerate_origins` accepts it.
"""

import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from streamlit.testing.v1 import AppTest

from dashboard.shared import palette
from dashboard.train import cohort
from fcstnyctaxi.core.train.evaluate_impl import GLOBAL_TIER, compute_evaluate_outputs
from fcstnyctaxi.lib.origin_modeling_table.builders import enumerate_origins
from fcstnyctaxi.schemas.run_identity import TrainRunIdentity

CHALLENGER = "model_a"
BENCHMARK = "model_b"
TIER_LABELS = ("low", "high")
SERIES = ("time_series_a", "time_series_b")

# A 4/4/5 quarter closing a 53-week year, so its last month holds 6 weeks.
WEEKS_IN_MONTH = {202010: 4, 202011: 4, 202012: 6, 202101: 4, 202102: 4, 202103: 5}
WEEKS = pd.date_range("2020-09-27", periods=sum(WEEKS_IN_MONTH.values()), freq="W-SUN")
# 202011's last week, which has completed it, so it forecasts the two after.
MONTH_END = WEEKS[7]
# 202012's second week, which forecasts it and the next.
MID_MONTH = WEEKS[9]
# The last four pair up into 4-week cells at 0 and 2 weeks actualized, two folds
# each, so a cell can lose one to nan.
ORIGIN_TARGETS = (
    (MONTH_END, (202012, 202101)),
    (MID_MONTH, (202012, 202101)),
    (WEEKS[13], (202101, 202102)),
    (WEEKS[15], (202101, 202102)),
    (WEEKS[17], (202102, 202103)),
    (WEEKS[19], (202102, 202103)),
)
# Where the benchmark is exact for the tier's one series, so that fold is excluded.
NAN_ORIGIN = WEEKS[15]
NAN_TIER = TIER_LABELS[0]


def _calendar() -> pd.DataFrame:
    """The calendar columns `enumerate_origins` and evaluate read."""
    rows = [
        {
            "fiscal_year_month": month,
            "fiscal_week_of_month": week,
            "weeks_in_month": weeks,
            "origin_month_fraction_elapsed": week / weeks,
        }
        for month, weeks in WEEKS_IN_MONTH.items()
        for week in range(1, weeks + 1)
    ]
    return pd.DataFrame(rows).assign(ds=WEEKS)


def _monthly_series(model: str) -> pd.DataFrame:
    """One sidecar's monthly_series; the benchmark errs more than the challenger."""
    error = 0.20 if model == BENCHMARK else 0.10
    rows = [
        {
            "forecast_origin_date": origin,
            "predicted_fiscal_year_month": month,
            "unique_id": series,
            "tier": tier,
            "series_weight": float(index + 1),
            "monthly_forecast": 100.0 * (index + 1) * (1 + error + 0.01 * step),
            "actual_monthly_total": 100.0 * (index + 1),
        }
        for origin, months in ORIGIN_TARGETS
        for step, month in enumerate(months)
        for index, (series, tier) in enumerate(zip(SERIES, TIER_LABELS, strict=True))
    ]
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def calendar() -> pd.DataFrame:
    """The fixture calendar, which `enumerate_origins` must accept."""
    frame = _calendar()
    # Self-check: every fixture origin is on the spine.
    spine = enumerate_origins(frame)
    assert {origin for origin, _ in ORIGIN_TARGETS} <= set(
        spine["forecast_origin_date"]
    )
    return frame


def _fold_metrics(benchmark_ms: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """The producer's fold metrics over the challenger's sidecar and this one."""
    identity = TrainRunIdentity(
        git_hash="0" * 40,
        feature_run_id="feature_run",
        train_run_id="train-20250101t000000000000z",
        panel_uri="gs://bucket/panel.parquet",
        calendar_uri="gs://bucket/calendar.parquet",
        additional_exog_uri="gs://bucket/exog.parquet",
    )
    outputs = compute_evaluate_outputs(
        challenger_ms=_monthly_series(CHALLENGER),
        benchmark_ms=benchmark_ms,
        calendar_df=calendar,
        challenger_model=CHALLENGER,
        benchmark_model=BENCHMARK,
        tier_labels=TIER_LABELS,
        identity=identity,
    )
    frame = outputs.fold_metrics
    return frame.assign(horizon=frame["horizon"].astype(str))


@pytest.fixture(scope="module")
def fold_metrics(calendar: pd.DataFrame) -> pd.DataFrame:
    """The producer's fold metrics over the two sidecars."""
    frame = _fold_metrics(_monthly_series(BENCHMARK), calendar)
    # Self-check: both horizons exist, so the horizon_2 test has rows to read.
    assert set(frame["horizon"]) == {"horizon_1", "horizon_2"}
    return frame


@pytest.fixture(scope="module")
def cohorts(fold_metrics: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """The fold metrics with each row's cohort attached."""
    return cohort.attach_cohorts(fold_metrics, calendar)


@pytest.fixture(scope="module")
def nan_cohorts(calendar: pd.DataFrame) -> pd.DataFrame:
    """The cohorts again, the benchmark exact for NAN_TIER's one series at
    NAN_ORIGIN's horizon_1 fold, which excludes that fold of its wrmae_pooled."""
    benchmark = _monthly_series(BENCHMARK)
    exact = (
        (benchmark["forecast_origin_date"] == NAN_ORIGIN)
        & (benchmark["tier"] == NAN_TIER)
        & (
            benchmark["predicted_fiscal_year_month"]
            == dict(ORIGIN_TARGETS)[NAN_ORIGIN][0]
        )
    )
    benchmark.loc[exact, "monthly_forecast"] = benchmark.loc[
        exact, "actual_monthly_total"
    ]
    frame = cohort.attach_cohorts(_fold_metrics(benchmark, calendar), calendar)
    # Self-check: that cell holds one nan fold and one finite one.
    values = _nan_cell(frame)["value"]
    assert values.isna().sum() == 1
    assert values.notna().sum() == 1
    return frame


def _nan_cell(cohorts: pd.DataFrame) -> pd.DataFrame:
    """The challenger's folds in the cell NAN_ORIGIN sits in."""
    origin = cohorts[cohorts["forecast_origin_date"] == NAN_ORIGIN].iloc[0]
    rows = _challenger(cohorts, "wrmae_pooled")
    return rows[
        (rows["tier"].astype(str) == NAN_TIER)
        & (rows["horizon"] == "horizon_1")
        & (rows["weeks_in_month"] == origin["weeks_in_month"])
        & (rows["weeks_actualized"] == origin["weeks_actualized"])
    ]


def _challenger(cohorts: pd.DataFrame, metric: str) -> pd.DataFrame:
    """The challenger's rows of one metric."""
    return cohorts[(cohorts["model"] == CHALLENGER) & (cohorts["metric"] == metric)]


def _at(cohorts: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    """One origin's fold rows, both horizons."""
    return cohorts[cohorts["forecast_origin_date"] == origin]


def test_a_month_end_origin_has_no_weeks_and_the_next_months_length(cohorts) -> None:
    """It has completed its own month, so it counts weeks of the one it targets."""
    own_month = _calendar().set_index("ds").loc[MONTH_END]
    # Self-check: the two months differ in length, so taking the own month shows.
    assert own_month["weeks_in_month"] != WEEKS_IN_MONTH[202012]

    rows = _at(cohorts, MONTH_END)
    assert set(rows["weeks_actualized"]) == {0}
    assert set(rows["weeks_in_month"]) == {WEEKS_IN_MONTH[202012]}


def test_a_mid_month_origin_counts_its_week_of_month(cohorts) -> None:
    """The weeks of its own month already actualized."""
    week = _calendar().set_index("ds").loc[MID_MONTH, "fiscal_week_of_month"]
    rows = _at(cohorts, MID_MONTH)
    assert set(rows["weeks_actualized"]) == {week}


def test_a_horizon_2_row_takes_its_origins_cohort(cohorts) -> None:
    """Joined on the fold key it would find no spine entry, since it predicts M+1."""
    for origin, _ in ORIGIN_TARGETS:
        rows = _at(cohorts, origin)
        by_horizon = rows.groupby("horizon")[cohort.COHORT_KEYS].agg(set)
        assert by_horizon.loc["horizon_2"].equals(by_horizon.loc["horizon_1"])
        assert rows[cohort.COHORT_KEYS].notna().all().all()


def test_a_six_week_month_yields_six(cohorts) -> None:
    """A 53-week year's closing month gets a cohort of its own, not a 4 or a 5."""
    assert 6 in set(cohorts["weeks_in_month"])


def test_every_fold_row_receives_a_cohort(fold_metrics, cohorts) -> None:
    """No row lost and none left without a cohort, whatever its horizon or tier."""
    assert len(cohorts) == len(fold_metrics)
    assert cohorts[cohort.COHORT_KEYS].notna().all().all()


# ================================================
# the strip
# ================================================

FACETS = (GLOBAL_TIER, *TIER_LABELS)
HORIZONS = ("horizon_1", "horizon_2")


def _figure(
    cohorts: pd.DataFrame, metric: str = "wrmae_pooled", tier_labels=TIER_LABELS
) -> go.Figure:
    """The strip over the producer's fold metrics."""
    return cohort.cohort_figure(cohorts, metric, tier_labels, CHALLENGER, BENCHMARK)


def _lengths(cohorts: pd.DataFrame, metric: str) -> list[int]:
    """The horizon_1 target-month lengths the challenger's folds hold."""
    return sorted(_challenger(cohorts, metric)["weeks_in_month"].unique())


def _facets(cohorts: pd.DataFrame, metric: str):
    """Each facet's grid position, with the month length and tier it shows."""
    for row, length in enumerate(_lengths(cohorts, metric), start=1):
        for col, tier in enumerate(FACETS, start=1):
            yield row, col, length, tier


def _in_facet(rows: pd.DataFrame, tier: str, length: int) -> pd.DataFrame:
    """One facet's rows."""
    return rows[(rows["tier"].astype(str) == tier) & (rows["weeks_in_month"] == length)]


def test_each_cell_draws_one_ring_per_finite_fold(nan_cohorts) -> None:
    """Each at its own value, so the count reads off the cell; a nan draws none."""
    figure = _figure(nan_cohorts)
    rows = _challenger(nan_cohorts, "wrmae_pooled").dropna(subset=["value"])
    for row, col, length, tier in _facets(nan_cohorts, "wrmae_pooled"):
        for horizon in HORIZONS:
            (rings,) = figure.select_traces(
                selector={"name": horizon}, row=row, col=col
            )
            drawn = sorted(
                (x, y) for x, y in zip(rings.x, rings.y, strict=True) if np.isfinite(y)
            )
            cell = _in_facet(rows, tier, length)
            cell = cell[cell["horizon"] == horizon]
            assert drawn == sorted(
                zip(cell["weeks_actualized"], cell["value"], strict=True)
            )


def test_a_nan_fold_draws_one_ring_fewer_and_the_mean_skips_it(nan_cohorts) -> None:
    """The mean is the nanmean, and its tooltip counts the finite folds."""
    cell = _nan_cell(nan_cohorts)
    position = cell["weeks_actualized"].iloc[0]
    row = _lengths(nan_cohorts, "wrmae_pooled").index(cell["weeks_in_month"].iloc[0])
    figure = _figure(nan_cohorts)
    grid = {"row": row + 1, "col": FACETS.index(NAN_TIER) + 1}
    (rings,) = figure.select_traces(selector={"name": "horizon_1"}, **grid)
    (means,) = figure.select_traces(selector={"name": "horizon_1 mean"}, **grid)

    drawn = [
        y
        for x, y in zip(rings.x, rings.y, strict=True)
        if x == position and np.isfinite(y)
    ]
    assert len(drawn) == len(cell) - 1
    at = list(means.x).index(position)
    assert means.y[at] == pytest.approx(np.nanmean(cell["value"]))
    assert list(means.customdata[at]) == [len(cell) - 1, len(cell)]


@pytest.mark.parametrize("metric", palette.METRICS)
def test_the_benchmark_tick_is_its_cell_mean_under_show_benchmark_only(
    cohorts, metric: str
) -> None:
    """One tick per cell at the benchmark's mean, never its folds."""
    figure = _figure(cohorts, metric)
    if not palette.SHOW_BENCHMARK[metric]:
        assert not list(figure.select_traces(selector={"name": cohort.BENCHMARK_NAME}))
        return

    rows = cohorts[(cohorts["model"] == BENCHMARK) & (cohorts["metric"] == metric)]
    for row, col, length, tier in _facets(cohorts, metric):
        ticks = figure.select_traces(
            selector={"name": cohort.BENCHMARK_NAME}, row=row, col=col
        )
        drawn = sorted(
            (x, y) for trace in ticks for x, y in zip(trace.x, trace.y, strict=True)
        )
        expected = sorted(
            (position, float(np.nanmean(values)))
            for (_, position), values in _in_facet(rows, tier, length).groupby(
                ["horizon", "weeks_actualized"]
            )["value"]
        )
        assert [x for x, _ in drawn] == [x for x, _ in expected]
        assert [y for _, y in drawn] == pytest.approx([y for _, y in expected])


def test_equal_means_still_show_both(cohorts) -> None:
    """The gray tick is drawn beneath the colored mean and is longer than it."""
    figure = _figure(cohorts, "wape")
    mean_names = {f"{horizon} mean" for horizon in HORIZONS}
    for row, col, _, _ in _facets(cohorts, "wape"):
        traces = list(figure.select_traces(row=row, col=col))
        ticks = [i for i, t in enumerate(traces) if t.name == cohort.BENCHMARK_NAME]
        means = [i for i, t in enumerate(traces) if t.name in mean_names]
        assert ticks and means
        assert max(ticks) < min(means)
        assert all(
            traces[i].marker.size > traces[j].marker.size for i in ticks for j in means
        )


def test_the_rings_are_open_so_overlapping_folds_stay_countable(cohorts) -> None:
    """A filled mark hides any fold within its radius; a ring lets it show."""
    figure = _figure(cohorts)
    rings = [
        t
        for horizon in HORIZONS
        for t in figure.select_traces(selector={"name": horizon})
    ]
    assert rings
    assert all(trace.marker.symbol.endswith("-open") for trace in rings)


def test_the_rows_are_the_target_month_lengths_in_the_run(cohorts) -> None:
    """Read from the data, so a 53-week year's 6-week month adds a row of its own."""
    labels = [
        annotation.text
        for annotation in _figure(cohorts).layout.annotations
        if annotation.name == cohort.ROWS_ANNOTATION
    ]
    lengths = [int(re.search(r"(\d+) weeks", label).group(1)) for label in labels]
    # Self-check: the fixture holds a length besides 4 and 5.
    assert 6 in _lengths(cohorts, "wrmae_pooled")
    assert lengths == _lengths(cohorts, "wrmae_pooled")


@pytest.mark.parametrize("metric", palette.METRICS)
def test_every_facet_shares_one_y_range_that_holds_every_mark(
    cohorts, metric: str
) -> None:
    """Every row is the same metric, so one scale, the benchmark ticks inside it."""
    figure = _figure(cohorts, metric)
    ranges = {tuple(axis.range) for axis in figure.select_yaxes()}
    assert len(ranges) == 1
    ((low, high),) = ranges
    plotted = np.array([y for trace in figure.data for y in trace.y], dtype=float)
    plotted = plotted[np.isfinite(plotted)]
    assert low <= plotted.min()
    assert plotted.max() <= high


def test_a_configured_tier_with_no_rows_still_draws_its_facets(cohorts) -> None:
    """Plotly draws no axis for a facet without a trace, so it would lose its frame."""
    tiers = (*TIER_LABELS, "tier_c")
    figure = _figure(cohorts, tier_labels=tiers)
    for row in range(1, len(_lengths(cohorts, "wrmae_pooled")) + 1):
        assert list(figure.select_traces(row=row, col=len(tiers) + 1))


# Unannotated: `AppTest.from_function` runs this function's source as a script of
# its own, where the annotations are evaluated and `pd` is undefined.
def _cohort_page(fold_metrics, calendar, tier_labels, challenger, benchmark):
    """The panel alone, as a script `AppTest` can run."""
    from dashboard.train import cohort

    cohort.render_cohort(fold_metrics, calendar, tier_labels, challenger, benchmark)


@pytest.mark.parametrize("metric", palette.METRICS)
def test_the_wape_caption_shows_for_wape_alone(fold_metrics, calendar, metric) -> None:
    """horizon_1 falling toward zero is wape's pattern; signed bias sits flat."""
    args = (fold_metrics, calendar, TIER_LABELS, CHALLENGER, BENCHMARK)
    app = AppTest.from_function(_cohort_page, args=args).run()
    app.selectbox(key=cohort.METRIC_KEY).set_value(metric).run()
    assert not app.exception, app.exception[0].message if app.exception else ""
    shown = any("fall toward zero" in caption.value for caption in app.caption)
    assert shown == (metric == "wape")
