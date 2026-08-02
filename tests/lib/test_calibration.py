from __future__ import annotations

import pandas as pd
import pytest
from tsbricks.backtesting.schema import ModelConfig

from fcstnyctaxi.lib.calibration import (
    calibrate_n_estimators,
    most_parsimonious_n_estimators,
)

# ================================================
# Fixtures
#
# 12 weeks, W-SUN, spanning three fiscal months: 202501 (weeks 0-3), 202502
# (weeks 4-7), 202503 (weeks 8-11). The one calibration origin sits at
# week 7 (last week of 202502, origin_month_fraction_elapsed == 1.0), with
# horizon=2 — so the fold's target weeks (8-9) fall entirely in 202503,
# giving exactly one horizon_label group ("horizon_1") to score per k.
#
# The fakes exist because calibrate_n_estimators() never needs a real fitted
# model: the fake fit callable that invoke_model() resolves just tags which
# train_df it was "fit" on; the fake predict callable that invoke_predict()
# resolves (via model_config.predict_callable) returns a fixed weekly forecast
# regardless of k, since truncation's effect on the forecast is models/
# lightgbm_weekly.py's concern, not this function's.
# ================================================


@pytest.fixture
def calendar_df() -> pd.DataFrame:
    dates = pd.date_range("2025-01-05", periods=12, freq="W-SUN")
    fiscal_week_of_month = [1, 2, 3, 4] * 3
    return pd.DataFrame(
        {
            "ds": dates,
            "fiscal_year_month": [202501] * 4 + [202502] * 4 + [202503] * 4,
            "origin_month_fraction_elapsed": [w / 4 for w in fiscal_week_of_month],
        }
    )


@pytest.fixture
def ts_df(calendar_df: pd.DataFrame) -> pd.DataFrame:
    """Two series, constant weekly value each — simple enough that the
    resulting monthly totals are easy to reason about by hand."""
    dates = calendar_df["ds"].tolist()
    return pd.DataFrame(
        {
            "unique_id": [10] * 12 + [20] * 12,
            "ds": dates + dates,
            "y": [10.0] * 12 + [5.0] * 12,
        }
    )


@pytest.fixture
def actual_monthly_df(calendar_df: pd.DataFrame) -> pd.DataFrame:
    """Ground-truth monthly totals for every (unique_id, fiscal_year_month)."""
    months = sorted(calendar_df["fiscal_year_month"].unique())
    return pd.DataFrame(
        [
            {
                "unique_id": uid,
                "fiscal_year_month": month,
                "actual_monthly_total": total,
            }
            for uid, total in [(10, 40.0), (20, 20.0)]
            for month in months
        ]
    )


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        fit_predict_callable="test_calibration._fake_model_callable",
        predict_callable="test_calibration._fake_predict_callable",
        hyperparameters={"freq": "W-SUN"},
    )


def _fake_model_callable(train_df: pd.DataFrame, horizon: int, **kwargs):
    """Stands in for a real fit callable (e.g. lightgbm_weekly()) — returns a
    trivial model_obj since the fake predict callable / set_truncation_iteration
    below never need a real fitted model to do their job in this test."""
    empty = pd.DataFrame({"unique_id": [], "ds": [], "ypred": []})
    return empty, empty, {"fit_on_rows": len(train_df)}


def _fake_predict_callable(
    model_obj, horizon: int, future_x_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Fixed weekly forecast for the last `horizon` weeks of future_x_df,
    regardless of k or model_obj — resolved by invoke_predict() from
    model_config.predict_callable. Truncation's effect on the forecast is
    models/lightgbm_weekly.py's concern, not this function's."""
    assert future_x_df is not None
    future_weeks = future_x_df["ds"].sort_values().iloc[-horizon:].tolist()
    return pd.DataFrame(
        {
            "unique_id": [10] * horizon + [20] * horizon,
            "ds": future_weeks + future_weeks,
            "ypred": [8.0] * horizon + [4.0] * horizon,
        }
    )


def _fake_set_truncation_iteration(model_obj, k: int) -> None:
    """No-op — this test verifies the grid-sweep/scoring loop's mechanics,
    not truncation behavior itself (tested in test_lightgbm_weekly.py)."""


# ================================================
# calibrate_n_estimators()
# ================================================


def test_calibrate_n_estimators_raises_on_empty_grid(
    ts_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    actual_monthly_df: pd.DataFrame,
    model_config: ModelConfig,
) -> None:
    n_estimators_grid: list[int] = []

    with pytest.raises(ValueError, match="grid is empty"):
        calibrate_n_estimators(
            ts_df=ts_df,
            calendar_df=calendar_df,
            actual_monthly_df=actual_monthly_df,
            model_config=model_config,
            n_estimators_grid=n_estimators_grid,
            calibration_origins=[("2025-02-23", 2)],
            set_truncation_iteration=_fake_set_truncation_iteration,
        )


def test_calibrate_n_estimators_raises_on_nan_series_weight(
    ts_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    actual_monthly_df: pd.DataFrame,
    model_config: ModelConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If a forecast unique_id is missing from weight_df, the left-merge yields a
    NaN series_weight, which would otherwise flow into scoring as a silent nan.
    The guard must raise instead. Forced here by patching compute_series_weights
    (in calibration's namespace) to return weights for only one of the two
    series -- dropping unique_id 20 -- so the merge leaves 20's weight NaN."""

    def _weights_missing_id_20(train_df, origin_date, calendar_df) -> pd.DataFrame:
        return pd.DataFrame({"unique_id": [10], "series_weight": [1.0]})

    monkeypatch.setattr(
        "fcstnyctaxi.lib.calibration.compute_series_weights", _weights_missing_id_20
    )

    with pytest.raises(ValueError, match="series_weight NaN"):
        calibrate_n_estimators(
            ts_df=ts_df,
            calendar_df=calendar_df,
            actual_monthly_df=actual_monthly_df,
            model_config=model_config,
            n_estimators_grid=[100],
            calibration_origins=[("2025-02-23", 2)],
            set_truncation_iteration=_fake_set_truncation_iteration,
        )


def test_calibrate_n_estimators_returns_expected_shape(
    ts_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    actual_monthly_df: pd.DataFrame,
    model_config: ModelConfig,
) -> None:
    """One calibration origin, one horizon_label, two n_estimators
    candidates -> exactly 2 result rows, with the expected columns and
    values for origin/n_estimators."""
    n_estimators_grid = [100, 200]
    result = calibrate_n_estimators(
        ts_df=ts_df,
        calendar_df=calendar_df,
        actual_monthly_df=actual_monthly_df,
        model_config=model_config,
        n_estimators_grid=n_estimators_grid,
        calibration_origins=[("2025-02-23", 2)],
        set_truncation_iteration=_fake_set_truncation_iteration,
    )

    assert set(result.columns) == {"origin", "n_estimators", "horizon", "score"}
    assert len(result) == 2
    assert sorted(result["n_estimators"].tolist()) == n_estimators_grid
    assert (result["horizon"] == "horizon_1").all()


def test_calibrate_n_estimators_calls_predict_and_truncation_once_per_origin_and_k(
    ts_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    actual_monthly_df: pd.DataFrame,
    model_config: ModelConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Predict (resolved from model_config.predict_callable) and
    set_truncation_iteration are each invoked exactly once per (origin, k) pair
    — here 1 origin x 2 candidates = 2 calls each, not once per origin or once
    overall."""
    counter_pred = [0]
    counter_trunc = [0]

    # Predict is resolved from config via dynamic_import, not injected, so count
    # it by patching the module attribute the dotted path resolves to. Capture
    # the original first: after the patch, the module-global name IS the wrapper,
    # so calling it by that name inside the wrapper would recurse infinitely.
    original_predict_callable = _fake_predict_callable

    def _counting_predict_callable(model_obj, horizon, future_x_df=None):
        counter_pred[0] += 1
        return original_predict_callable(model_obj, horizon, future_x_df)

    monkeypatch.setattr(
        "test_calibration._fake_predict_callable", _counting_predict_callable
    )

    # Truncation is still an injected parameter, so wrap it directly.
    def _counting_set_truncation_iteration(model_obj, k):
        counter_trunc[0] += 1
        _fake_set_truncation_iteration(model_obj, k)

    _ = calibrate_n_estimators(
        ts_df=ts_df,
        calendar_df=calendar_df,
        actual_monthly_df=actual_monthly_df,
        model_config=model_config,
        n_estimators_grid=[100, 200],
        calibration_origins=[("2025-02-23", 2)],
        set_truncation_iteration=_counting_set_truncation_iteration,
    )
    assert counter_pred[0] == 2
    assert counter_trunc[0] == 2


# ================================================
# most_parsimonious_n_estimators()
# # "k" / "k*" below is a value of n_estimators (the candidate boosting round count).
# The function returns k* = the smallest n_estimators whose smoothed score is
# within tolerance of the best.
# ================================================


def test_most_parsimonious_interior_plateau() -> None:
    """A V-shaped curve with a clear interior minimum -> k* is that interior
    value (well inside the grid, neither floor nor ceiling)."""
    scores = pd.Series(
        [10, 8, 6, 5, 6, 8, 10],
        index=[25, 50, 75, 100, 125, 150, 175],
    )
    assert most_parsimonious_n_estimators(scores) == 100


def test_most_parsimonious_still_improving_at_ceiling() -> None:
    """A monotonically decreasing curve -> k* pins at the grid ceiling. This is
    the 'still improving past the grid' case the inconclusive warning flags."""
    scores = pd.Series(
        [10, 9, 8, 7, 6, 5],
        index=[25, 50, 75, 100, 125, 150],
    )
    assert most_parsimonious_n_estimators(scores) == 150


def test_most_parsimonious_flat_curve_picks_floor() -> None:
    """A flat curve -> every k is within tolerance, so parsimony picks the
    smallest n_estimators (the floor)."""
    scores = pd.Series([5, 5, 5, 5], index=[25, 50, 75, 100])
    assert most_parsimonious_n_estimators(scores) == 25


def test_most_parsimonious_tolerance_beats_raw_argmin() -> None:
    """The distinguishing behavior: when a smaller k is within epsilon of the
    best, k* is that smaller k which is NOT the raw argmin. A plain idxmin
    implementation would fail this test; the tolerance/parsimony logic passes it.
    """
    scores = pd.Series(
        [10, 8, 7.5, 7, 5.99, 5.98, 10],
        index=[25, 50, 75, 100, 125, 150, 175],
    )
    assert most_parsimonious_n_estimators(scores, smoothing_window=1) == 125
    assert scores.idxmin() == 150
