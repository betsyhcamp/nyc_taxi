from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from models.lightgbm_weekly import (
    _set_lightgbm_iteration,
    lightgbm_weekly,
    lightgbm_weekly_predict,
)

# ================================================
# Repeated-predict statelessness
#
# The whole fit-once-then-truncate calibration sweep rests on one assumption:
# truncating a fitted model to k rounds and predicting does NOT mutate state
# that contaminates a later predict at a different k. If it did, candidate k's
# would interfere and the calibration curve would be garbage.
#
# The fixture below is a real LightGBM fit (not a fake), deliberately given
# enough calendar-driven signal — and small enough min_data_in_leaf — that the
# model builds trees whose predictions genuinely change as rounds accumulate.
# That makes the "k1 vs k2 differ" guard meaningful, so the "k1 == k1-again"
# equality can't pass vacuously.
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
    that are known for the future weeks too — which lets more boosting rounds
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


def test_repeated_predict_is_stateless_across_truncation(
    train_df: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """Fit once, then truncate-and-predict at k1, k2, and k1 again. The first
    and third forecasts (same k) must be identical — repeated predicts don't
    contaminate each other — and the k1/k2 forecasts must differ, proving
    truncation actually changes the output so the equality check isn't vacuous.
    """
    mlfcst = _fit_once(train_df, calendar_df)
    k1, k2 = 5, 50

    _set_lightgbm_iteration(mlfcst, k1)
    f1 = lightgbm_weekly_predict(mlfcst, _HORIZON, future_x_df=calendar_df)

    _set_lightgbm_iteration(mlfcst, k2)
    f2 = lightgbm_weekly_predict(mlfcst, _HORIZON, future_x_df=calendar_df)

    _set_lightgbm_iteration(mlfcst, k1)
    f3 = lightgbm_weekly_predict(mlfcst, _HORIZON, future_x_df=calendar_df)

    # Guard against a vacuous test: truncation must actually change the forecast.
    max_abs_diff = np.abs(f1["ypred"].to_numpy() - f2["ypred"].to_numpy()).max()
    assert max_abs_diff > 1e-6, (
        "k1 and k2 forecasts are identical — truncation had no effect, so the "
        "statelessness assertion below would pass erroneously. Strengthen the "
        "fixture's signal or widen k1/k2."
    )

    # The load-bearing property: predict at the same k is repeatable / stateless.
    assert_frame_equal(f1, f3)
