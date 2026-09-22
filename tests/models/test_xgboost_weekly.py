from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mlforecast import MLForecast
from pandas.testing import assert_frame_equal

from fcstnyctaxi.lib.exog import build_exog_frame
from fcstnyctaxi.models.xgboost_weekly import (
    xgboost_weekly,
    xgboost_weekly_fit,
    xgboost_weekly_predict,
    xgboost_weekly_save,
)

# ================================================
# The pipeline callable contract
#
# These callables merge whatever frame they are handed, on ["unique_id", "ds"],
# and select nothing. The impl decides the feature set, so the fixture hands them
# an assembled frame and the calendar carries a column outside it.
# ================================================

FREQ = "W-SUN"
_N_TRAIN_WEEKS = 40
_HORIZON = 2

# Shallow and few rounds: enough to split on a 40-week fixture, and fast.
_FIXTURE_HYPERPARAMETERS = {
    "lags": [1],
    "rolling_mean_window": 4,
    "max_depth": 3,
    "n_estimators": 60,
}

# What the impl selects; `fiscal_year` is deliberately not among them.
_EXOG_FEATURES = (
    "fiscal_week_of_month",
    "fiscal_month",
    "weeks_in_month",
    "count_workdays",
)


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
    for uid, base in [("time_series_a", 100.0), ("time_series_b", 40.0)]:
        y = (
            base
            + cal["fiscal_week_of_month"].to_numpy() * 8.0
            + cal["fiscal_month"].to_numpy() * 4.0
            + rng.normal(0, 1.0, _N_TRAIN_WEEKS)
        )
        frames.append(pd.DataFrame({"unique_id": uid, "ds": dates, "y": y}))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def exog_df(train_df: pd.DataFrame, calendar_df: pd.DataFrame) -> pd.DataFrame:
    """The frame the impl assembles, which is what the callables receive."""
    return build_exog_frame(train_df, calendar_df, exog_features=_EXOG_FEATURES)


@pytest.fixture
def fitted_model(train_df: pd.DataFrame, exog_df: pd.DataFrame) -> MLForecast:
    """A model fitted through the fit half alone, on the assembled frame."""
    return xgboost_weekly_fit(
        train_df, FREQ, exog_df=exog_df, **_FIXTURE_HYPERPARAMETERS
    )


def test_the_wrapper_is_exactly_its_fit_and_predict_halves(
    train_df: pd.DataFrame, exog_df: pd.DataFrame, fitted_model: MLForecast
) -> None:
    """Else the model the backtest scores is not the model final_fit registers."""
    wrapper_forecast, _fitted_values, _model = xgboost_weekly(
        train_df, _HORIZON, FREQ, future_x_df=exog_df, **_FIXTURE_HYPERPARAMETERS
    )

    composed_forecast = xgboost_weekly_predict(
        fitted_model, _HORIZON, future_x_df=exog_df
    )

    assert_frame_equal(wrapper_forecast, composed_forecast)


def test_a_saved_model_reloads_and_predicts_identically(
    exog_df: pd.DataFrame, fitted_model: MLForecast, tmp_path: Path
) -> None:
    """The model's half of the bundle contract; final_fit only checks bytes landed."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    before = xgboost_weekly_predict(fitted_model, _HORIZON, future_x_df=exog_df)

    xgboost_weekly_save(fitted_model, model_dir)
    after = xgboost_weekly_predict(
        MLForecast.load(model_dir), _HORIZON, future_x_df=exog_df
    )

    assert any(path.stat().st_size > 0 for path in model_dir.iterdir())
    assert_frame_equal(before, after)


def test_the_fit_half_trains_on_exactly_the_columns_it_is_handed(
    calendar_df: pd.DataFrame, fitted_model: MLForecast
) -> None:
    """MLForecast adopts extra columns as features at fit, so only the booster shows
    whether the merge happened, and whether anything unselected got in."""
    # Self-check: the exclusion below is vacuous unless the calendar carries it.
    assert "fiscal_year" in calendar_df.columns
    booster_features = set(
        fitted_model.models_["XGBRegressor"].get_booster().feature_names or []
    )

    assert set(_EXOG_FEATURES) <= booster_features
    assert "fiscal_year" not in booster_features


def test_the_fit_half_refuses_an_unknown_hyperparameter(
    train_df: pd.DataFrame, exog_df: pd.DataFrame
) -> None:
    """The fit-predict callable swallows the same name to stay tsbricks-compatible."""
    with pytest.raises(TypeError, match="max_detph"):
        xgboost_weekly_fit(
            train_df,
            FREQ,
            exog_df=exog_df,
            max_detph=3,
            **_FIXTURE_HYPERPARAMETERS,
        )


def test_a_ds_that_does_not_match_freq_is_refused(train_df: pd.DataFrame) -> None:
    """A freq default once cast datetime ds to integers silently."""
    integer_ds = train_df.assign(ds=train_df.groupby("unique_id").cumcount())

    with pytest.raises(TypeError, match="requires datetime"):
        xgboost_weekly_fit(integer_ds, FREQ, **_FIXTURE_HYPERPARAMETERS)
