import numpy as np
import pandas as pd
import pytest

from fcstnyctaxi.models.lightgbm_weekly import lightgbm_weekly

# ================================================
# The pipeline callable contract
#
# The fixture is a real LightGBM fit, not a fake: it is small enough to be fast and
# carries enough calendar-driven signal that the calendar features change the
# forecast, which is what makes an assertion about them meaningful.
#
# The truncation sweep that used to live here moved to test_lightgbm_weekly_dev.py
# with _set_lightgbm_iteration, whose only callers are the calibration config and
# that test.
# ================================================

FREQ = "W-SUN"
_N_TRAIN_WEEKS = 40
_HORIZON = 2


@pytest.fixture
def calendar_df() -> pd.DataFrame:
    """Weekly fiscal calendar spanning the training weeks plus _HORIZON future
    weeks, carrying the calendar features lightgbm_weekly consumes. Exactly
    _HORIZON weeks extend past the training window, as _build_future_calendar_df
    requires."""
    n_weeks = _N_TRAIN_WEEKS + _HORIZON
    week_of_month = (np.arange(n_weeks) % 4) + 1
    month = ((np.arange(n_weeks) // 4) % 12) + 1
    return pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-07", periods=n_weeks, freq=FREQ),
            "fiscal_week_of_month": week_of_month,
            "fiscal_month": month,
            "weeks_in_month": 4,
            "count_workdays": 20 + week_of_month,
        }
    )


@pytest.fixture
def train_df(calendar_df: pd.DataFrame) -> pd.DataFrame:
    """Two series over the first _N_TRAIN_WEEKS weeks. y is calendar-driven
    (plus mild noise) rather than a pure trend, so the signal lives in features
    that are known for the future weeks too, which lets more boosting rounds
    genuinely change the forecast rather than a tree model flat-lining on
    out-of-range extrapolation."""
    cal = calendar_df.iloc[:_N_TRAIN_WEEKS]
    dates = cal["ds"].to_numpy()
    rng = np.random.default_rng(0)
    frames = []
    for uid, base in [(10, 100.0), (20, 40.0)]:
        y = (
            base
            + cal["fiscal_week_of_month"].to_numpy() * 8.0
            + cal["fiscal_month"].to_numpy() * 4.0
            + rng.normal(0, 1.0, _N_TRAIN_WEEKS)
        )
        frames.append(pd.DataFrame({"unique_id": uid, "ds": dates, "y": y}))
    return pd.concat(frames, ignore_index=True)


def _fit_once(train_df: pd.DataFrame, calendar_df: pd.DataFrame):
    """Fit with settings that make the small fixture actually split
    (min_data_in_leaf small) and build enough rounds that truncation matters;
    lags=[1] keeps the history requirement low so 40 weeks suffice."""
    _, _, mlfcst = lightgbm_weekly(
        train_df=train_df,
        horizon=_HORIZON,
        freq=FREQ,
        future_x_df=calendar_df,
        lags=[1],
        rolling_mean_window=4,
        min_data_in_leaf=5,
        n_estimators=60,
    )
    return mlfcst
