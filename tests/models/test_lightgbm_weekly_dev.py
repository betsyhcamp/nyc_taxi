import numpy as np
import pandas as pd
import pytest
import yaml
from pandas.testing import assert_frame_equal
from tsbricks.backtesting.schema import ModelConfig
from tsbricks.runner import dynamic_import, invoke_model, invoke_predict

from experimental_models.lightgbm_weekly_dev import _CALENDAR_FEATURES
from fcstnyctaxi.lib.utils import get_project_root_dir

# ================================================
# The development callable contract
#
# Every test here resolves its callable from the notebook config by dotted path and
# calls it through tsbricks, rather than importing it. That is the whole point: the
# notebooks resolve these the same way, and nothing else in the suite reaches this
# path. tests/config/test_config_tree.py validates config/ against schemas and never
# resolves a dotted path, and tests/lib/test_calibration.py's callables are fakes, so
# importing the module and asserting the configs parse would pass on a mangled
# callable.
#
# The fixture is a real LightGBM fit, deliberately given enough calendar-driven
# signal - and a small enough min_data_in_leaf - that the model builds trees whose
# predictions genuinely change as rounds accumulate. That makes the "k1 vs k2 differ"
# guard meaningful, so the "k1 == k1-again" equality cannot pass vacuously.
# ================================================

FREQ = "W-SUN"
_N_TRAIN_WEEKS = 40
_HORIZON = 2

_BACKTEST_CONFIG_DIR = get_project_root_dir() / "notebooks" / "backtest_configs"


def _notebook_model_config() -> ModelConfig:
    """The model config the notebooks load, read from the file they read."""
    block = yaml.safe_load(
        (_BACKTEST_CONFIG_DIR / "model_weekly_lightgbm.yaml").read_text()
    )["model"]
    return ModelConfig(**block)


def _notebook_truncation_adapter():
    """The truncation adapter calibrate_n_estimators.py resolves, resolved its way."""
    calibration = yaml.safe_load(
        (_BACKTEST_CONFIG_DIR / "calibration_config.yaml").read_text()
    )
    return dynamic_import(calibration["truncation_adapter"])


def _sized_for_the_fixture(model_config: ModelConfig) -> ModelConfig:
    """The configured callables, with hyperparameters a 40-week fixture can fit.

    The shipped lags reach 52 weeks and min_data_in_leaf is 125, so on this fixture
    the booster would build no splits and every forecast would be one constant.
    """
    return model_config.model_copy(
        update={
            "hyperparameters": {
                **(model_config.hyperparameters or {}),
                "lags": [1],
                "min_data_in_leaf": 5,
                "n_estimators": 60,
            }
        }
    )


@pytest.fixture
def calendar_df() -> pd.DataFrame:
    """A raw ds-keyed calendar extending exactly _HORIZON weeks past the panel."""
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


def test_the_notebook_configs_name_callables_from_experimental_models() -> None:
    """The two packages carry different contracts, so which one a notebook names
    is the contract it gets; at this commit the two are still copies."""
    model_config = _notebook_model_config()
    adapter_path = yaml.safe_load(
        (_BACKTEST_CONFIG_DIR / "calibration_config.yaml").read_text()
    )["truncation_adapter"]

    for dotted_path in (
        model_config.fit_predict_callable,
        model_config.predict_callable,
        adapter_path,
    ):
        assert dotted_path.startswith("experimental_models.")


def test_the_configured_callables_fit_and_predict_against_a_raw_calendar(
    train_df: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """Resolved through tsbricks and handed a ds-keyed calendar, the pair produces a
    forecast per series and horizon step, from a booster that saw the calendar."""
    model_config = _sized_for_the_fixture(_notebook_model_config())

    forecast_df, fitted_df, mlfcst = invoke_model(
        train_df, model_config, _HORIZON, future_x_df=calendar_df
    )
    predicted_df = invoke_predict(
        mlfcst, model_config, _HORIZON, future_x_df=calendar_df
    )

    expected_rows = train_df["unique_id"].nunique() * _HORIZON
    assert len(forecast_df) == expected_rows
    assert not forecast_df["ypred"].isna().any()
    assert len(fitted_df) > 0
    # A callable that dropped the merge would forecast, so the forecast alone does
    # not prove the calendar was consumed. The booster's feature list does.
    booster = mlfcst.models_["LGBMRegressor"].booster_
    assert set(_CALENDAR_FEATURES) <= set(booster.feature_name())
    # The two entry points must agree: calibration fits through one and sweeps
    # through the other, so a divergence would score a model it never fitted.
    assert_frame_equal(forecast_df, predicted_df)


def test_repeated_predict_is_stateless_across_truncation(
    train_df: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """Truncating to k, predicting, then returning to k must reproduce the first
    forecast: calibration sweeps one fitted model and would otherwise score noise."""
    model_config = _sized_for_the_fixture(_notebook_model_config())
    set_truncation_iteration = _notebook_truncation_adapter()
    _, _, mlfcst = invoke_model(
        train_df, model_config, _HORIZON, future_x_df=calendar_df
    )
    k1, k2 = 5, 50

    set_truncation_iteration(mlfcst, k1)
    first = invoke_predict(mlfcst, model_config, _HORIZON, future_x_df=calendar_df)

    set_truncation_iteration(mlfcst, k2)
    second = invoke_predict(mlfcst, model_config, _HORIZON, future_x_df=calendar_df)

    set_truncation_iteration(mlfcst, k1)
    third = invoke_predict(mlfcst, model_config, _HORIZON, future_x_df=calendar_df)

    # Guard against a vacuous test: truncation must actually change the forecast.
    max_abs_diff = np.abs(first["ypred"].to_numpy() - second["ypred"].to_numpy()).max()
    assert max_abs_diff > 1e-6, (
        "k1 and k2 forecasts are identical, so truncation had no effect and the "
        "statelessness assertion below would pass erroneously. Strengthen the "
        "fixture's signal or widen k1/k2."
    )

    assert_frame_equal(first, third)
