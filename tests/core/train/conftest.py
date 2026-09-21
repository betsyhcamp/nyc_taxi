import pandas as pd
import pytest

# ================================================
# The shaped synthetic fixture
#
# Shared so backtest and final_fit are exercised against one set of inputs. 80 weeks,
# W-SUN, four-week fiscal months. Five series with distinct trailing means so every
# tier label is used, one at zero revenue for the lowest tier, 52 weeks of history
# before the first origin, and one series activating after the first origin so a
# ragged series set is exercised. Integer ids, as the panel contract declares.
# ================================================

_FULL_WEEKS_PER_MONTH = 4
_FULL_N_WEEKS = 80
_FULL_WEEKS = pd.date_range("2024-01-07", periods=_FULL_N_WEEKS, freq="W-SUN")
_FULL_MONTHS = [202401 + i for i in range(12)] + [202501 + i for i in range(8)]
# Weekly level, and the week the series becomes active.
_FULL_SERIES = {
    1: (1000.0, 0),  # high
    2: (300.0, 0),  # mid
    3: (80.0, 0),  # low
    4: (5.0, 0),  # tiny
    5: (0.0, 0),  # zero
    6: (200.0, 66),  # late
}


@pytest.fixture(scope="module")
def full_calendar() -> pd.DataFrame:
    """Every column the contract declares, so the impl's trim has something to keep."""
    month_index = [week // _FULL_WEEKS_PER_MONTH for week in range(_FULL_N_WEEKS)]
    week_of_month = [week % _FULL_WEEKS_PER_MONTH + 1 for week in range(_FULL_N_WEEKS)]
    return pd.DataFrame(
        {
            "ds": _FULL_WEEKS,
            "fiscal_year_month": [_FULL_MONTHS[m] for m in month_index],
            "fiscal_month": [m % 12 + 1 for m in month_index],
            "fiscal_week_of_month": week_of_month,
            "weeks_in_month": _FULL_WEEKS_PER_MONTH,
            "origin_month_fraction_elapsed": [
                week / _FULL_WEEKS_PER_MONTH for week in week_of_month
            ],
            "count_workdays": 5,
            "fiscal_year": [_FULL_MONTHS[m] // 100 for m in month_index],
            "fiscal_year_week": list(range(1, 49)) + list(range(1, 33)),
        }
    )


@pytest.fixture(scope="module")
def full_panel() -> pd.DataFrame:
    """Deterministic levels on a five-week cycle, so naive is not trivially exact."""
    return pd.DataFrame(
        [
            {
                "unique_id": uid,
                "ds": _FULL_WEEKS[week],
                "y": level * (1 + 0.1 * (week % 5)),
            }
            for uid, (level, first_week) in _FULL_SERIES.items()
            for week in range(first_week, _FULL_N_WEEKS)
        ]
    )
