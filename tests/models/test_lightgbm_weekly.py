from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.lightgbm_weekly import _build_future_calendar_df

# ================================================
# Fixtures
# ================================================


@pytest.fixture
def calendar_df() -> pd.DataFrame:
    """8 weeks, W-SUN, with distinct per-week calendar values.

    last_ds (see below) sits at index 3, leaving exactly 4 future weeks
    (indices 4-7) which is enough to exercise both a horizon that fits and one
    that doesn't.
    """
    dates = pd.date_range("2025-01-05", periods=8, freq="W-SUN")
    return pd.DataFrame(
        {
            "ds": dates,
            "fiscal_week_of_month": [1, 2, 3, 4, 1, 2, 3, 4],
            "count_workdays": [5, 5, 4, 5, 5, 3, 5, 5],
        }
    )


@pytest.fixture
def last_ds(calendar_df: pd.DataFrame) -> pd.Timestamp:
    """The training panel's last observed date is index 3 of calendar_df."""
    return calendar_df["ds"].iloc[3]


@pytest.fixture
def unique_ids() -> np.ndarray:
    return np.array([10, 20])


# ================================================
# _build_future_calendar_df — happy path
# ================================================


def test_build_future_calendar_df_returns_exact_row_count(
    calendar_df: pd.DataFrame, last_ds: pd.Timestamp, unique_ids: np.ndarray
) -> None:
    """Row count is exactly len(unique_ids) * horizon, not "all remaining weeks"."""
    horizon = 3
    result = _build_future_calendar_df(
        unique_ids=unique_ids,
        last_ds=last_ds,
        calendar_df=calendar_df,
        horizon=horizon,
        cal_cols=["fiscal_week_of_month", "count_workdays"],
    )
    assert len(result) == len(unique_ids) * horizon


def test_build_future_calendar_df_selects_correct_future_dates(
    calendar_df: pd.DataFrame, last_ds: pd.Timestamp, unique_ids: np.ndarray
) -> None:
    """Selected dates are exactly the horizon weeks strictly after last_ds, in order."""
    horizon = 3
    result = _build_future_calendar_df(
        unique_ids=unique_ids,
        last_ds=last_ds,
        calendar_df=calendar_df,
        horizon=horizon,
        cal_cols=["fiscal_week_of_month", "count_workdays"],
    )
    expected_dates = calendar_df["ds"].iloc[4:7]
    assert set(result["ds"].unique()) == set(expected_dates)
    assert (result["ds"] > last_ds).all()


def test_build_future_calendar_df_attaches_correct_calendar_values(
    calendar_df: pd.DataFrame, last_ds: pd.Timestamp, unique_ids: np.ndarray
) -> None:
    """Each (unique_id, ds) row carries the calendar values for its own ds —
    not, e.g., values from last_ds or from a mismatched row."""
    horizon = 3
    result = _build_future_calendar_df(
        unique_ids=unique_ids,
        last_ds=last_ds,
        calendar_df=calendar_df,
        horizon=horizon,
        cal_cols=["fiscal_week_of_month", "count_workdays"],
    )
    merge_df = result.merge(
        calendar_df[["ds", "fiscal_week_of_month", "count_workdays"]],
        how="left",
        on="ds",
    )
    assert all(
        merge_df["fiscal_week_of_month_x"].eq(merge_df["fiscal_week_of_month_y"])
    )
    assert all(merge_df["count_workdays_x"].eq(merge_df["count_workdays_y"]))


# ================================================
# _build_future_calendar_df — guard clause
# ================================================


def test_build_future_calendar_df_raises_when_calendar_too_short(
    calendar_df: pd.DataFrame, last_ds: pd.Timestamp, unique_ids: np.ndarray
) -> None:
    """Raises ValueError when calendar_df doesn't extend horizon weeks past last_ds.

    Only 4 future weeks exist past last_ds (indices 4-7); requesting 5 must fail
    loudly rather than silently returning a short/NaN-padded frame.
    """
    with pytest.raises(ValueError, match="5"):
        _build_future_calendar_df(
            unique_ids=unique_ids,
            last_ds=last_ds,
            calendar_df=calendar_df,
            horizon=5,
            cal_cols=["fiscal_week_of_month", "count_workdays"],
        )
