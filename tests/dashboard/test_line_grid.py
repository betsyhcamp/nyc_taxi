"""The line grids' data: the weekly reindex, the month rule, and `global`.

The weekly panel is hand-built, since the Feature pipeline owns the real one; its
self-check is that every row falls on a calendar week. `per_series_comparison`
comes from `compute_evaluate_outputs`, its actuals from the panel through
`compute_actual_monthly_totals`, as backtest derives them.
"""

import numpy as np
import pandas as pd
import pytest

from dashboard.train import line_grid
from fcstnyctaxi.core.train.evaluate_impl import compute_evaluate_outputs
from fcstnyctaxi.lib.monthly_aggregation import compute_actual_monthly_totals
from fcstnyctaxi.schemas.run_identity import TrainRunIdentity

CHALLENGER = "model_a"
BENCHMARK = "model_b"
TIER_LABELS = ("low", "high")

# Half a 4/4/5 fiscal year of Sunday weeks.
WEEKS_IN_MONTH = {202401: 4, 202402: 4, 202403: 5, 202404: 4, 202405: 4, 202406: 5}
WEEKS = pd.date_range("2023-12-31", periods=sum(WEEKS_IN_MONTH.values()), freq="W-SUN")
FULL = range(len(WEEKS))
GAP_WEEK = 9
"""The second week of 202403."""
MID_MONTH_START = 6
"""The third week of 202402."""

# The series without a fold at the first origin starts after it, in 202404's
# second week; the second origin, in 202404's third, has folds for all three.
LATE = "time_series_c"
SPANS = {"time_series_a": FULL, "time_series_b": FULL, LATE: range(14, len(WEEKS))}
FIRST_ORIGIN = WEEKS[GAP_WEEK]
ORIGIN_TARGETS = (
    (FIRST_ORIGIN, (202403, 202404)),
    (WEEKS[15], (202404, 202405)),
)


def _calendar() -> pd.DataFrame:
    """The calendar columns the reindex, the month rule and evaluate read."""
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


def _snapshot(
    spans: dict[str, range],
    *,
    nan: frozenset = frozenset(),
    absent: frozenset = frozenset(),
) -> pd.DataFrame:
    """A weekly panel as Feature ships it. `nan` keeps a (series, week) row with a
    null `y`; `absent` drops the row."""
    rows = [
        {
            "unique_id": series,
            "ds": WEEKS[week],
            "y": np.nan if (series, week) in nan else 100.0 * (index + 1) + week,
        }
        for index, (series, weeks) in enumerate(spans.items())
        for week in weeks
        if (series, week) not in absent
    ]
    frame = pd.DataFrame(rows)
    # Self-check: every row is a calendar week, as in Feature's panel.
    assert set(frame["ds"]) <= set(WEEKS)
    return frame


def _months(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Each series' months under the month rule."""
    calendar = _calendar()
    return line_grid.monthly_totals(
        line_grid.reindex_weeks(snapshot, calendar), calendar
    )


def _by_month(months: pd.DataFrame, series: str) -> pd.DataFrame:
    """One series' months, indexed by month."""
    return months[months["unique_id"] == series].set_index("fiscal_year_month")


def _monthly_series(model: str, snapshot: pd.DataFrame) -> pd.DataFrame:
    """One sidecar's monthly_series: a fold per series with training rows at each
    origin, scored against the panel's own monthly totals."""
    actuals = compute_actual_monthly_totals(snapshot, _calendar()).set_index(
        ["unique_id", "fiscal_year_month"]
    )["actual_monthly_total"]
    error = 0.20 if model == BENCHMARK else 0.10
    rows = []
    for origin, months in ORIGIN_TARGETS:
        started = snapshot.loc[snapshot["ds"] <= origin, "unique_id"].unique()
        for step, month in enumerate(months):
            for index, series in enumerate(sorted(started)):
                actual = actuals[(series, month)]
                rows.append(
                    {
                        "forecast_origin_date": origin,
                        "predicted_fiscal_year_month": month,
                        "unique_id": series,
                        "tier": TIER_LABELS[index % 2],
                        "series_weight": float(index + 1),
                        "monthly_forecast": actual * (1 + error + 0.01 * step),
                        "actual_monthly_total": actual,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def per_series() -> pd.DataFrame:
    """The producer's per_series_comparison over the panel with a late series."""
    snapshot = _snapshot(SPANS)
    identity = TrainRunIdentity(
        git_hash="0" * 40,
        feature_run_id="feature_run",
        train_run_id="train-20250101t000000000000z",
        panel_uri="gs://bucket/panel.parquet",
        calendar_uri="gs://bucket/calendar.parquet",
        additional_exog_uri="gs://bucket/exog.parquet",
    )
    table = compute_evaluate_outputs(
        challenger_ms=_monthly_series(CHALLENGER, snapshot),
        benchmark_ms=_monthly_series(BENCHMARK, snapshot),
        calendar_df=_calendar(),
        challenger_model=CHALLENGER,
        benchmark_model=BENCHMARK,
        tier_labels=TIER_LABELS,
        identity=identity,
    ).per_series_comparison
    # Self-check: the late series has folds, none at the first origin, and one
    # predicting a month the first origin also predicts.
    late = table[table["unique_id"] == LATE]
    assert not late.empty
    assert FIRST_ORIGIN not in set(late["forecast_origin_date"])
    first = table[table["forecast_origin_date"] == FIRST_ORIGIN]
    assert set(late["predicted_fiscal_year_month"]) & set(
        first["predicted_fiscal_year_month"]
    )
    return table


# ================================================
# the month rule
# ================================================


def test_a_nan_week_and_an_absent_row_gap_their_month_alike() -> None:
    """Each leaves a week inside the span without a value, so neither reads whole."""
    months = _months(
        _snapshot(
            {"time_series_a": FULL, "time_series_b": FULL},
            nan=frozenset({("time_series_a", GAP_WEEK)}),
            absent=frozenset({("time_series_b", GAP_WEEK)}),
        )
    )
    nan_week = _by_month(months, "time_series_a")
    absent_row = _by_month(months, "time_series_b")
    assert nan_week["status"].equals(absent_row["status"])
    assert nan_week.loc[202403, "status"] == line_grid.GAPPED
    assert np.isnan(nan_week.loc[202403, "y"])


def test_a_series_starting_mid_month_drops_its_first_month_without_a_gap() -> None:
    """Nothing is missing there, the series had not begun, so no note is owed."""
    # Self-check: the series' first week is not its month's first.
    assert _calendar()["fiscal_week_of_month"][MID_MONTH_START] != 1
    months = _by_month(
        _months(_snapshot({"time_series_a": range(MID_MONTH_START, len(WEEKS))})),
        "time_series_a",
    )
    assert months.loc[202402, "status"] == line_grid.INCOMPLETE
    assert np.isnan(months.loc[202402, "y"])
    assert months.loc[202403, "status"] == line_grid.COMPLETE


def test_a_panel_ending_mid_month_drops_its_last_month() -> None:
    """The weeks after the panel's end are not yet known, so the sum is partial."""
    end = len(WEEKS) - 2
    # Self-check: the panel ends before its last month does.
    assert _calendar()["origin_month_fraction_elapsed"][end - 1] < 1
    months = _months(
        _snapshot({"time_series_a": range(end), "time_series_b": range(end)})
    )
    for series in ("time_series_a", "time_series_b"):
        assert _by_month(months, series).loc[202406, "status"] == line_grid.INCOMPLETE
        assert _by_month(months, series).loc[202405, "status"] == line_grid.COMPLETE


def test_monthly_x_is_the_months_last_fiscal_week(per_series) -> None:
    """Read from the calendar, so the gap from the origin rule to it is the MTD."""
    calendar = _calendar()
    last_week = calendar.groupby("fiscal_year_month")["ds"].max()
    months = _months(_snapshot(SPANS))
    points = line_grid.forecast_points(
        per_series, FIRST_ORIGIN, line_grid.fold_set(per_series, FIRST_ORIGIN), calendar
    )
    assert months["ds"].equals(months["fiscal_year_month"].map(last_week))
    assert points["ds"].equals(points["predicted_fiscal_year_month"].map(last_week))


# ================================================
# global
# ================================================


def test_global_is_gapped_when_one_member_is() -> None:
    """Its true total is unknown, whatever the other members hold."""
    members = ("time_series_a", "time_series_b")
    months = _months(
        _snapshot(dict.fromkeys(members, FULL), nan=frozenset({(members[0], GAP_WEEK)}))
    )
    # Self-check: the other member's month is whole, so only the gap marks it.
    assert _by_month(months, members[1]).loc[202403, "status"] == line_grid.COMPLETE
    totals = line_grid.global_monthly(months, members).set_index("fiscal_year_month")
    assert totals.loc[202403, "status"] == line_grid.GAPPED
    assert np.isnan(totals.loc[202403, "y"])


def test_global_leaves_out_a_members_incomplete_month() -> None:
    """As that member's own facet does: global sums the members whose month is whole."""
    months = _months(
        _snapshot(
            {
                "time_series_a": FULL,
                "time_series_b": range(MID_MONTH_START, len(WEEKS)),
            }
        )
    )
    totals = line_grid.global_monthly(
        months, ("time_series_a", "time_series_b")
    ).set_index("fiscal_year_month")
    whole = _by_month(months, "time_series_a")
    assert totals.loc[202402, "status"] == line_grid.COMPLETE
    assert totals.loc[202402, "y"] == whole.loc[202402, "y"]


def test_a_series_without_a_fold_at_the_origin_is_left_out_of_global(
    per_series,
) -> None:
    """From both sides: global's actual and forecast cover the same series."""
    members = line_grid.fold_set(per_series, FIRST_ORIGIN)
    assert LATE not in members

    months = _months(_snapshot(SPANS))
    # Self-check: the late series has whole months global would otherwise sum.
    assert (_by_month(months, LATE)["status"] == line_grid.COMPLETE).any()
    member_months = months[months["unique_id"].isin(members)]
    whole = member_months[member_months["status"] == line_grid.COMPLETE]
    expected = whole.groupby("fiscal_year_month")["y"].sum()
    totals = line_grid.global_monthly(months, members).set_index("fiscal_year_month")
    assert totals.loc[expected.index, "y"].tolist() == pytest.approx(expected.tolist())

    points = line_grid.forecast_points(per_series, FIRST_ORIGIN, members, _calendar())
    at_origin = per_series[per_series["forecast_origin_date"] == FIRST_ORIGIN]
    expected = at_origin.groupby("predicted_fiscal_year_month")["monthly_forecast_ch"]
    assert points.set_index("predicted_fiscal_year_month")[
        "monthly_forecast_ch"
    ].tolist() == pytest.approx(expected.sum().tolist())
