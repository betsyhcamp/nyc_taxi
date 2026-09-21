from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mlforecast import MLForecast
from pandas.testing import assert_frame_equal

from fcstnyctaxi.models.lightgbm_weekly import (
    _CALENDAR_FEATURES,
    lightgbm_weekly,
    lightgbm_weekly_fit,
    lightgbm_weekly_predict,
    lightgbm_weekly_save,
)

# ================================================
# The pipeline callable contract
#
# The fixture is a real LightGBM fit, not a fake, and carries a calendar column
# outside _CALENDAR_FEATURES so the selection lightgbm_weekly still performs has
# something to drop. A wrapper that stopped selecting would train on that column
# and forecast differently, which is what the composition test below detects.
#
# The truncation sweep that used to live here moved to test_lightgbm_weekly_dev.py
# with _set_lightgbm_iteration, whose only callers are the calibration config and
# that test.
# ================================================

FREQ = "W-SUN"
_N_TRAIN_WEEKS = 40
_HORIZON = 2

# Small enough that the fixture splits, and enough rounds to be a real model.
_FIXTURE_HYPERPARAMETERS = {
    "lags": [1],
    "rolling_mean_window": 4,
    "min_data_in_leaf": 5,
    "n_estimators": 60,
}


@pytest.fixture
def calendar_df() -> pd.DataFrame:
    """A ds-keyed calendar extending exactly _HORIZON weeks past the panel."""
    n_weeks = _N_TRAIN_WEEKS + _HORIZON
    week_of_month = (np.arange(n_weeks) % 4) + 1
    return pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-07", periods=n_weeks, freq=FREQ),
            "fiscal_week_of_month": week_of_month,
            "fiscal_month": ((np.arange(n_weeks) // 4) % 12) + 1,
            "weeks_in_month": 4,
            "count_workdays": 20 + week_of_month,
            "fiscal_year": 2024,
        }
    )


@pytest.fixture
def train_df(calendar_df: pd.DataFrame) -> pd.DataFrame:
    """Two series whose y is calendar-driven, so future-known features carry signal."""
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


@pytest.fixture
def fitted_model(train_df: pd.DataFrame, calendar_df: pd.DataFrame) -> MLForecast:
    """A model fitted through the fit half alone, on the columns the wrapper picks."""
    return lightgbm_weekly_fit(
        train_df,
        FREQ,
        exog_df=calendar_df[["ds"] + _CALENDAR_FEATURES],
        **_FIXTURE_HYPERPARAMETERS,
    )


def test_the_wrapper_is_exactly_its_fit_and_predict_halves(
    train_df: pd.DataFrame, calendar_df: pd.DataFrame, fitted_model: MLForecast
) -> None:
    """The fit-predict callable must add nothing to the halves it delegates to, or
    the model the backtest scores stops being the model final_fit would register."""
    wrapper_forecast, _fitted_values, _model = lightgbm_weekly(
        train_df,
        _HORIZON,
        FREQ,
        future_x_df=calendar_df,
        **_FIXTURE_HYPERPARAMETERS,
    )

    composed_forecast = lightgbm_weekly_predict(
        fitted_model, _HORIZON, future_x_df=calendar_df
    )

    assert_frame_equal(wrapper_forecast, composed_forecast)


def test_a_saved_model_reloads_and_predicts_identically(
    calendar_df: pd.DataFrame, fitted_model: MLForecast, tmp_path: Path
) -> None:
    """The data-independent half of the bundle's write check, which is why no
    load-back runs at runtime: proven once here rather than on every run."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    before = lightgbm_weekly_predict(fitted_model, _HORIZON, future_x_df=calendar_df)

    lightgbm_weekly_save(fitted_model, model_dir)
    after = lightgbm_weekly_predict(
        MLForecast.load(model_dir), _HORIZON, future_x_df=calendar_df
    )

    assert any(path.stat().st_size > 0 for path in model_dir.iterdir())
    assert_frame_equal(before, after)


def test_the_fit_half_trains_on_exactly_the_columns_it_is_handed(
    fitted_model: MLForecast,
) -> None:
    """MLForecast adopts extra columns as features at fit and ignores them at
    predict, so only the booster shows whether the merge happened at all."""
    booster_features = set(
        fitted_model.models_["LGBMRegressor"].booster_.feature_name()
    )

    assert set(_CALENDAR_FEATURES) <= booster_features
    # Handed to neither half, so its presence would mean the caller's selection
    # was bypassed and the model trained on a wider frame than it was given.
    assert "fiscal_year" not in booster_features


def test_the_fit_half_refuses_an_unknown_hyperparameter(
    train_df: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """Declaring no **kwargs is what makes a misspelled hyperparameter raise here;
    the fit-predict callable swallows the same name to stay tsbricks-compatible."""
    with pytest.raises(TypeError, match="num_leavez"):
        lightgbm_weekly_fit(
            train_df,
            FREQ,
            exog_df=calendar_df[["ds"] + _CALENDAR_FEATURES],
            num_leavez=31,
            **_FIXTURE_HYPERPARAMETERS,
        )
