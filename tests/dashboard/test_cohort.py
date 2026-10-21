"""Position cohorts, attached to the fold metrics evaluate emits.

The calendar is hand-built, since the Feature pipeline owns the real one; its
self-check is that `enumerate_origins` accepts it.
"""

import pandas as pd
import pytest

from dashboard.train import cohort
from fcstnyctaxi.core.train.evaluate_impl import compute_evaluate_outputs
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
ORIGIN_TARGETS = (
    (MONTH_END, (202012, 202101)),
    (MID_MONTH, (202012, 202101)),
    (WEEKS[13], (202101, 202102)),
    (WEEKS[15], (202101, 202102)),
)


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


@pytest.fixture(scope="module")
def fold_metrics(calendar: pd.DataFrame) -> pd.DataFrame:
    """The producer's fold metrics over the two sidecars."""
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
        benchmark_ms=_monthly_series(BENCHMARK),
        calendar_df=calendar,
        challenger_model=CHALLENGER,
        benchmark_model=BENCHMARK,
        tier_labels=TIER_LABELS,
        identity=identity,
    )
    frame = outputs.fold_metrics
    frame = frame.assign(horizon=frame["horizon"].astype(str))
    # Self-check: both horizons exist, so the horizon_2 test has rows to read.
    assert set(frame["horizon"]) == {"horizon_1", "horizon_2"}
    return frame


@pytest.fixture(scope="module")
def cohorts(fold_metrics: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """The fold metrics with each row's cohort attached."""
    return cohort.attach_cohorts(fold_metrics, calendar)


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
