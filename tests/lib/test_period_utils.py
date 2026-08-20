from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from fcstnyctaxi.lib.period_utils import (
    _get_trailing_dates,
    assign_tiers,
    compute_series_weights,
    derive_horizon_label,
    generate_origins_for_periods,
    label_horizon,
)

# ================================================
# Fixtures
# ================================================


@pytest.fixture
def calendar_df() -> pd.DataFrame:
    """4 fiscal months × 4 Sundays = 16 weeks (202501–202504)."""
    dates = pd.date_range("2025-01-05", periods=16, freq="W-SUN")
    months = [202501] * 4 + [202502] * 4 + [202503] * 4 + [202504] * 4
    return pd.DataFrame({"ds": dates, "fiscal_year_month": months})


@pytest.fixture
def train_df() -> pd.DataFrame:
    """3 series × 8 weeks with distinct revenue levels."""
    dates = pd.date_range("2025-01-05", periods=8, freq="W-SUN")
    rows = [
        {"unique_id": uid, "ds": d, "y": rev}
        for uid, rev in [("high", 10.0), ("mid", 5.0), ("low", 1.0)]
        for d in dates
    ]
    return pd.DataFrame(rows)


# ================================================
# _get_trailing_dates
# ================================================


def test_get_trailing_dates_returns_exact_count(calendar_df: pd.DataFrame) -> None:
    """Returns exactly trailing_weeks dates when the calendar is large enough."""
    result = _get_trailing_dates(calendar_df, date(2025, 2, 23), trailing_weeks=4)
    assert len(result) == 4


def test_get_trailing_dates_all_on_or_before_origin(calendar_df: pd.DataFrame) -> None:
    """No returned date exceeds origin_date."""
    origin = date(2025, 2, 16)
    result = _get_trailing_dates(calendar_df, origin, trailing_weeks=6)
    assert (result <= pd.Timestamp(origin)).all()


def test_get_trailing_dates_truncates_when_calendar_is_short(
    calendar_df: pd.DataFrame,
) -> None:
    """Returns fewer dates when fewer calendar rows precede origin_date."""
    # Only 2025-01-05 and 2025-01-12 fall on or before the origin
    result = _get_trailing_dates(calendar_df, date(2025, 1, 12), trailing_weeks=10)
    assert len(result) == 2


# ================================================
# generate_origins_for_periods
# ================================================


def test_generate_origins_single_start_month(calendar_df: pd.DataFrame) -> None:
    """Happy path: one start_month, horizon=1 → correct origin window and horizons."""
    # month_before=202501 → first_origin=2025-01-26 (last of 202501)
    # month_end=202502   → last_origin=2025-02-16 (second-to-last of 202502)
    # last_week=2025-02-23 → horizons: 4, 3, 2, 1
    result = generate_origins_for_periods(
        start_months=[202502],
        forecast_horizon_months=1,
        calendar_df=calendar_df,
    )
    assert result == [
        {"origin": "2025-01-26", "horizon": 4},
        {"origin": "2025-02-02", "horizon": 3},
        {"origin": "2025-02-09", "horizon": 2},
        {"origin": "2025-02-16", "horizon": 1},
    ]


def test_generate_origins_result_is_sorted_with_no_duplicates(
    calendar_df: pd.DataFrame,
) -> None:
    """Two overlapping periods produce a sorted, deduplicated origin list."""
    result = generate_origins_for_periods(
        start_months=[202502, 202503],
        forecast_horizon_months=2,
        calendar_df=calendar_df,
    )
    origins = [r["origin"] for r in result]
    assert origins == sorted(set(origins))


def test_generate_origins_raises_when_start_month_is_first_in_calendar(
    calendar_df: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="no preceding month"):
        generate_origins_for_periods(
            start_months=[202501],
            forecast_horizon_months=1,
            calendar_df=calendar_df,
        )


def test_generate_origins_raises_when_horizon_exceeds_calendar(
    calendar_df: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="extends beyond"):
        generate_origins_for_periods(
            start_months=[202504],
            forecast_horizon_months=2,
            calendar_df=calendar_df,
        )


def test_generate_origins_raises_when_start_month_not_in_calendar(
    calendar_df: pd.DataFrame,
) -> None:
    """start_month absent from calendar raises ValueError with helpful message."""
    with pytest.raises(ValueError, match="not found in calendar"):
        generate_origins_for_periods(
            start_months=[202512],
            forecast_horizon_months=1,
            calendar_df=calendar_df,
        )


def test_generate_origins_string_period_column_coerced_to_int(
    calendar_df: pd.DataFrame,
) -> None:
    """Period column stored as strings coerced to int and produces correct results."""
    str_calendar = calendar_df.assign(
        fiscal_year_month=calendar_df["fiscal_year_month"].astype(str)
    )
    result = generate_origins_for_periods(
        start_months=[202502],
        forecast_horizon_months=1,
        calendar_df=str_calendar,
    )
    assert result == [
        {"origin": "2025-01-26", "horizon": 4},
        {"origin": "2025-02-02", "horizon": 3},
        {"origin": "2025-02-09", "horizon": 2},
        {"origin": "2025-02-16", "horizon": 1},
    ]


def test_generate_origins_string_period_column_does_not_mutate_calendar_df(
    calendar_df: pd.DataFrame,
) -> None:
    """calendar_df is not modified when period column requires int coercion."""
    str_calendar = calendar_df.assign(
        fiscal_year_month=calendar_df["fiscal_year_month"].astype(str)
    )
    original_dtype = str_calendar["fiscal_year_month"].dtype
    generate_origins_for_periods(
        start_months=[202502],
        forecast_horizon_months=1,
        calendar_df=str_calendar,
    )
    assert str_calendar["fiscal_year_month"].dtype == original_dtype


def test_generate_origins_custom_column_names(calendar_df: pd.DataFrame) -> None:
    """calendar_time_col and calendar_period_id are respected."""
    renamed = calendar_df.rename(
        columns={"ds": "week_start", "fiscal_year_month": "period_id"}
    )
    result = generate_origins_for_periods(
        start_months=[202502],
        forecast_horizon_months=1,
        calendar_df=renamed,
        calendar_time_col="week_start",
        calendar_period_id="period_id",
    )
    assert result == [
        {"origin": "2025-01-26", "horizon": 4},
        {"origin": "2025-02-02", "horizon": 3},
        {"origin": "2025-02-09", "horizon": 2},
        {"origin": "2025-02-16", "horizon": 1},
    ]


# ================================================
# assign_tiers
# ================================================


def test_assign_tiers_returns_correct_columns(
    train_df: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    result = assign_tiers(train_df, date(2025, 2, 23), calendar_df, trailing_weeks=8)
    assert list(result.columns) == ["unique_id", "tier"]


def test_assign_tiers_covers_all_series_exactly_once(
    train_df: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    result = assign_tiers(train_df, date(2025, 2, 23), calendar_df, trailing_weeks=8)
    assert set(result["unique_id"]) == {"high", "mid", "low"}
    assert len(result) == result["unique_id"].nunique()


def test_assign_tiers_higher_revenue_gets_higher_tier(
    train_df: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """Higher-revenue series receive a higher tier label."""
    result = assign_tiers(
        train_df,
        date(2025, 2, 23),
        calendar_df,
        trailing_weeks=8,
        num_tiers=3,
        tier_labels=("low", "middle", "high"),
    )
    tier_map = result.set_index("unique_id")["tier"].astype(str).to_dict()
    order = ["low", "middle", "high"]
    assert order.index(tier_map["high"]) > order.index(tier_map["mid"])
    assert order.index(tier_map["mid"]) > order.index(tier_map["low"])


def test_assign_tiers_no_revenue_series_gets_tier_labels_0_with_custom_labels(
    calendar_df: pd.DataFrame,
) -> None:
    """No-revenue series falls back to tier_labels[0], not hardcoded 'very_low'."""
    dates = pd.date_range("2025-01-05", periods=8, freq="W-SUN")
    df = pd.DataFrame(
        {
            "unique_id": ["zero"] * 8 + ["pos"] * 8,
            "ds": list(dates) * 2,
            "y": [0.0] * 8 + [5.0] * 8,
        }
    )
    result = assign_tiers(
        df,
        date(2025, 2, 23),
        calendar_df,
        trailing_weeks=8,
        tier_labels=("bottom", "mid", "top"),  # custom tiers different than default
        num_tiers=3,
    )
    tier_map = result.set_index("unique_id")["tier"].astype(str).to_dict()
    assert tier_map["zero"] == "bottom"
    assert tier_map["zero"] != "very_low"


def test_assign_tiers_series_with_no_positive_revenue_gets_very_low(
    calendar_df: pd.DataFrame,
) -> None:
    """Series with only zero revenue falls into very_low via the fillna path."""
    dates = pd.date_range("2025-01-05", periods=8, freq="W-SUN")
    df = pd.DataFrame(
        {
            "unique_id": ["zero"] * 8 + ["pos"] * 8,
            "ds": list(dates) * 2,
            "y": [0.0] * 8 + [5.0] * 8,
        }
    )
    result = assign_tiers(df, date(2025, 2, 23), calendar_df, trailing_weeks=8)
    tier_map = result.set_index("unique_id")["tier"].astype(str).to_dict()
    assert tier_map["zero"] == "very_low"


def test_assign_tiers_sparse_series_not_in_trailing_window_gets_very_low(
    calendar_df: pd.DataFrame,
) -> None:
    """Series with data only before the trailing window → very_low."""
    early_dates = pd.date_range("2025-01-05", periods=4, freq="W-SUN")  # 202501
    recent_dates = pd.date_range("2025-02-02", periods=4, freq="W-SUN")  # 202502
    df = pd.DataFrame(
        {
            "unique_id": ["sparse"] * 4 + ["active"] * 4,
            "ds": list(early_dates) + list(recent_dates),
            "y": [10.0] * 4 + [5.0] * 4,
        }
    )
    # trailing_weeks=4 -> window covers only 202502 (2025-02-02 to 2025-02-23)
    # "sparse" has no data in that window
    result = assign_tiers(df, date(2025, 2, 23), calendar_df, trailing_weeks=4)
    tier_map = result.set_index("unique_id")["tier"].astype(str).to_dict()
    assert tier_map["sparse"] == "very_low"


def test_assign_tiers_all_no_positive_revenue_returns_all_very_low(
    calendar_df: pd.DataFrame,
) -> None:
    """When no series has positive revenue, early return fires and all get very_low."""

    dates = pd.date_range("2025-01-05", periods=8, freq="W-SUN")
    df = pd.DataFrame(
        {
            "unique_id": ["a"] * 8 + ["b"] * 8 + ["c"] * 8,
            "ds": list(dates) * 3,
            "y": [0.0] * 8 + [-1.0] * 8 + [0.0] * 8,
        }
    )
    result = assign_tiers(df, date(2025, 2, 23), calendar_df, trailing_weeks=3)
    tier_map = result.set_index("unique_id")["tier"].astype(str).to_dict()
    assert tier_map["a"] == "very_low"
    assert tier_map["b"] == "very_low"
    assert tier_map["c"] == "very_low"


# ================================================
# compute_series_weights
# ================================================


def test_compute_series_weights_returns_correct_columns(
    train_df: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    result = compute_series_weights(
        train_df, date(2025, 2, 23), calendar_df, trailing_weeks=8
    )
    assert list(result.columns) == ["unique_id", "series_weight"]


def test_compute_series_weights_covers_all_series_exactly_once(
    train_df: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    result = compute_series_weights(
        train_df, date(2025, 2, 23), calendar_df, trailing_weeks=8
    )
    assert set(result["unique_id"]) == {"high", "mid", "low"}
    assert len(result) == result["unique_id"].nunique()


def test_compute_series_weights_applies_dampening_fn(
    train_df: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """Weight equals dampening_fn(sum of trailing revenue) for each series."""
    result = compute_series_weights(
        train_df, date(2025, 2, 23), calendar_df, trailing_weeks=8, dampening_fn=np.sqrt
    )
    weight_map = result.set_index("unique_id")["series_weight"].to_dict()
    # "high": 8 weeks × 10.0 = 80.0 -> sqrt(80)
    # "mid":  8 weeks × 5.0  = 40.0 -> sqrt(40)
    assert weight_map["high"] == pytest.approx(np.sqrt(80.0))
    assert weight_map["mid"] == pytest.approx(np.sqrt(40.0))


def test_compute_series_weights_clips_negative_revenue_to_zero(
    calendar_df: pd.DataFrame,
) -> None:
    """Series with only negative revenue gets dampening_fn(0) = 0."""
    dates = pd.date_range("2025-01-05", periods=8, freq="W-SUN")
    df = pd.DataFrame({"unique_id": ["neg"] * 8, "ds": list(dates), "y": [-5.0] * 8})
    result = compute_series_weights(
        df, date(2025, 2, 23), calendar_df, trailing_weeks=8
    )
    weight_map = result.set_index("unique_id")["series_weight"].to_dict()
    assert weight_map["neg"] == pytest.approx(0.0)


def test_compute_series_weights_sparse_series_gets_zero_weight(
    calendar_df: pd.DataFrame,
) -> None:
    """Series with no observations in the trailing window → weight_fn(0) = 0."""
    early_dates = pd.date_range("2025-01-05", periods=4, freq="W-SUN")  # 202501
    recent_dates = pd.date_range("2025-02-02", periods=4, freq="W-SUN")  # 202502
    df = pd.DataFrame(
        {
            "unique_id": ["sparse"] * 4 + ["active"] * 4,
            "ds": list(early_dates) + list(recent_dates),
            "y": [10.0] * 4 + [5.0] * 4,
        }
    )
    # trailing_weeks=4 → window covers only 202502; "sparse" has no rows there
    result = compute_series_weights(
        df, date(2025, 2, 23), calendar_df, trailing_weeks=4
    )
    weight_map = result.set_index("unique_id")["series_weight"].to_dict()
    assert weight_map["sparse"] == pytest.approx(0.0)


# ================================================
# derive_horizon_label
# ================================================


def test_derive_horizon_label_scalar_origin_mid_month() -> None:
    """Origin mid-February (frac<1.0): last_completed=January, so February is
    horizon_1 and March is horizon_2."""
    result = derive_horizon_label(
        predicted_fiscal_year_month=pd.Series([202502, 202503]),
        origin_fiscal_year_month=202502,
        origin_month_fraction_elapsed=0.4,
    )
    assert result.tolist() == ["horizon_1", "horizon_2"]


def test_derive_horizon_label_january_rollover() -> None:
    """Origin in January: prev_fiscal_year_month must roll back to December
    of the prior fiscal year (202412), not month 00."""
    result = derive_horizon_label(
        predicted_fiscal_year_month=pd.Series([202501]),
        origin_fiscal_year_month=202501,
        origin_month_fraction_elapsed=0.4,
    )
    assert result.tolist() == ["horizon_1"]


def test_derive_horizon_label_frac_equals_one_shifts_current_month() -> None:
    """Origin sits exactly at month-end (frac=1.0): February now counts as
    already completed, so February itself is horizon_0 and March becomes
    horizon_1 -- one month earlier than the mid-month case above."""
    result = derive_horizon_label(
        predicted_fiscal_year_month=pd.Series([202502, 202503]),
        origin_fiscal_year_month=202502,
        origin_month_fraction_elapsed=1.0,
    )
    assert result.tolist() == ["horizon_0", "horizon_1"]


def test_derive_horizon_label_series_inputs_multiple_origins() -> None:
    """Three rows, one vectorized call, both np.where branches firing at once:
    origin 202502 (frac=1.0, month-end) alongside origin 202503 (frac=0.25,
    midmonth) reused across two different targets which proves the per row
    branching in prev_fiscal_year_month/last_completed works across a real
    multi origin Series, not just when exercised one origin at a time."""
    predicted_fiscal_year_month = pd.Series([202503, 202503, 202504])
    origin_fiscal_year_month = pd.Series([202502, 202503, 202503])
    origin_month_fraction_elapsed = pd.Series([1.0, 0.25, 0.25])

    result = derive_horizon_label(
        predicted_fiscal_year_month=predicted_fiscal_year_month,
        origin_fiscal_year_month=origin_fiscal_year_month,
        origin_month_fraction_elapsed=origin_month_fraction_elapsed,
    )
    assert result.tolist() == ["horizon_1", "horizon_1", "horizon_2"]


# ================================================
# label_horizon
#
# calendar_df runs 2025-01-05 + 16 W-SUN weeks, four per fiscal month:
# 202501 ends 01-26, 202502 ends 02-23, 202503 ends 03-23, 202504 starts 03-30.
# ================================================


@pytest.fixture
def monthly_series() -> pd.DataFrame:
    """Three rows spanning both horizons and both month-end branches."""
    return pd.DataFrame(
        {
            "forecast_origin_date": pd.to_datetime(
                ["2025-01-26", "2025-01-26", "2025-02-23"]
            ),
            "predicted_fiscal_year_month": [202501, 202502, 202503],
            "origin_month_fraction_elapsed": [0.75, 0.75, 1.0],
        }
    )


def test_label_horizon_matches_the_inline_map_then_delegate_pattern(
    monthly_series: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """Reproduces the two call sites it replaces, value for value."""
    inline = derive_horizon_label(
        predicted_fiscal_year_month=monthly_series["predicted_fiscal_year_month"],
        origin_fiscal_year_month=monthly_series["forecast_origin_date"].map(
            calendar_df.set_index("ds")["fiscal_year_month"]
        ),
        origin_month_fraction_elapsed=monthly_series["origin_month_fraction_elapsed"],
    )
    assert label_horizon(monthly_series, calendar_df).equals(inline)


def test_label_horizon_labels_both_horizons(
    monthly_series: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """A horizon_2 row is labeled as such rather than collapsing to horizon_1."""
    result = label_horizon(monthly_series, calendar_df)
    assert result.tolist() == ["horizon_1", "horizon_2", "horizon_1"]


def test_label_horizon_returns_a_series_aligned_to_the_input_index(
    monthly_series: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """Callers assign the result back onto a frame, so a reset index would misalign."""
    reindexed = monthly_series.set_index(pd.Index([10, 20, 30], name="row"))
    assert label_horizon(reindexed, calendar_df).index.equals(reindexed.index)


def test_label_horizon_matches_an_origin_whose_unit_differs_from_the_calendar(
    monthly_series: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """Registered sidecars carry ms origins against an ns calendar; both must label."""
    ms_origins = monthly_series.assign(
        forecast_origin_date=monthly_series["forecast_origin_date"].astype(
            "datetime64[ms]"
        )
    )
    assert calendar_df["ds"].dt.unit == "ns"
    assert ms_origins["forecast_origin_date"].dt.unit == "ms"
    assert (
        label_horizon(ms_origins, calendar_df).tolist()
        == label_horizon(monthly_series, calendar_df).tolist()
    )


def test_label_horizon_ignores_calendar_columns_it_does_not_need(
    monthly_series: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """A full fiscal calendar carries many columns; only two are required."""
    wide = calendar_df.assign(weeks_in_month=4, count_workdays=5)
    assert (
        label_horizon(monthly_series, wide).tolist()
        == label_horizon(monthly_series, calendar_df).tolist()
    )


def test_label_horizon_collapses_repeated_identical_calendar_rows(
    monthly_series: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """A ds repeated with the same fiscal month is benign, not a mapping conflict."""
    doubled = pd.concat([calendar_df, calendar_df], ignore_index=True)
    assert (
        label_horizon(monthly_series, doubled).tolist()
        == label_horizon(monthly_series, calendar_df).tolist()
    )


def test_label_horizon_raises_when_one_ds_carries_two_fiscal_months(
    monthly_series: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """Conflicting labels are a real defect; Series.map would raise obscurely."""
    conflicting = pd.concat(
        [calendar_df, calendar_df.head(1).assign(fiscal_year_month=209912)],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match=r"more than one fiscal_year_month"):
        label_horizon(monthly_series, conflicting)


def test_label_horizon_raises_when_an_origin_is_absent_from_the_calendar(
    monthly_series: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """The load-bearing check: a truncated calendar is what corrupts a label."""
    orphaned = monthly_series.assign(
        forecast_origin_date=pd.to_datetime(["2025-01-26", "2025-01-26", "2099-12-27"])
    )
    with pytest.raises(ValueError, match=r"absent from calendar_df"):
        label_horizon(orphaned, calendar_df)


def test_label_horizon_raises_on_a_non_datetime_origin_column(
    monthly_series: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """Names the dtype rather than reporting an unexplained wall of nulls."""
    as_strings = monthly_series.assign(
        forecast_origin_date=monthly_series["forecast_origin_date"].astype(str)
    )
    with pytest.raises(ValueError, match=r"must be a datetime dtype"):
        label_horizon(as_strings, calendar_df)


@pytest.mark.parametrize(
    "missing",
    [
        "forecast_origin_date",
        "predicted_fiscal_year_month",
        "origin_month_fraction_elapsed",
    ],
)
def test_label_horizon_raises_when_monthly_series_lacks_a_column(
    monthly_series: pd.DataFrame, calendar_df: pd.DataFrame, missing: str
) -> None:
    """Each of the three is required; the message names the frame at fault."""
    with pytest.raises(ValueError, match=r"monthly_series is missing required"):
        label_horizon(monthly_series.drop(columns=[missing]), calendar_df)


@pytest.mark.parametrize("missing", ["ds", "fiscal_year_month"])
def test_label_horizon_raises_when_the_calendar_lacks_a_column(
    monthly_series: pd.DataFrame, calendar_df: pd.DataFrame, missing: str
) -> None:
    """Distinguishes a bad calendar from a bad monthly_series in the message."""
    with pytest.raises(ValueError, match=r"calendar_df is missing required"):
        label_horizon(monthly_series, calendar_df.drop(columns=[missing]))


def test_label_horizon_does_not_mutate_its_inputs(
    monthly_series: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """Neither frame gains an origin-fiscal-month column as a side effect."""
    before_series, before_calendar = monthly_series.copy(), calendar_df.copy()
    label_horizon(monthly_series, calendar_df)
    pd.testing.assert_frame_equal(monthly_series, before_series)
    pd.testing.assert_frame_equal(calendar_df, before_calendar)
