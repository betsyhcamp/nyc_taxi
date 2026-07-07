from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from fcstnyctaxi.lib.period_utils import (
    _get_trailing_dates,
    assign_tiers,
    compute_series_weights,
    generate_origins_for_periods,
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
    """Happy path: one start_month, horizon=1 → correct origin window."""
    # month_before=202501 → first_origin=2025-01-26 (last of 202501)
    # month_end=202502   → last_origin=2025-02-16 (second-to-last of 202502)
    result = generate_origins_for_periods(
        start_months=[202502],
        forecast_horizon_months=1,
        calendar_df=calendar_df,
    )
    assert result == ["2025-01-26", "2025-02-02", "2025-02-09", "2025-02-16"]


def test_generate_origins_result_is_sorted_with_no_duplicates(
    calendar_df: pd.DataFrame,
) -> None:
    """Two overlapping periods produce a sorted, deduplicated origin list."""
    result = generate_origins_for_periods(
        start_months=[202502, 202503],
        forecast_horizon_months=2,
        calendar_df=calendar_df,
    )
    assert result == sorted(set(result))


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
    # trailing_weeks=4 → window covers only 202502 (2025-02-02 to 2025-02-23)
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


def test_compute_series_weights_applies_weight_fn(
    train_df: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """Weight equals weight_fn(sum of trailing revenue) for each series."""
    result = compute_series_weights(
        train_df, date(2025, 2, 23), calendar_df, trailing_weeks=8, weight_fn=np.sqrt
    )
    weight_map = result.set_index("unique_id")["series_weight"].to_dict()
    # "high": 8 weeks × 10.0 = 80.0 → sqrt(80)
    # "mid":  8 weeks × 5.0  = 40.0 → sqrt(40)
    assert weight_map["high"] == pytest.approx(np.sqrt(80.0))
    assert weight_map["mid"] == pytest.approx(np.sqrt(40.0))


def test_compute_series_weights_clips_negative_revenue_to_zero(
    calendar_df: pd.DataFrame,
) -> None:
    """Series with only negative revenue gets weight_fn(0) = 0."""
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
