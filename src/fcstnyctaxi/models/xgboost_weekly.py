from pathlib import Path

import pandas as pd
import xgboost as xgb
from mlforecast import MLForecast
from mlforecast.lag_transforms import (
    RollingMean,
)

from fcstnyctaxi.models._utils import _align_ds_dtype


def xgboost_weekly_fit(
    train_df: pd.DataFrame,
    freq: str,
    exog_df: pd.DataFrame | None = None,
    lags: list[int] | None = None,
    rolling_mean_window: int = 4,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    min_child_weight: float = 1.0,
    n_estimators: int = 1000,
    n_jobs: int = 1,
) -> MLForecast:
    """Fit a recursive XGBoost MLForecast on the panel it is handed.

    No ``**kwargs``, so a misspelled hyperparameter is a ``TypeError``. The L1 loss
    matches the weighted-MAE metric the backtest scores.

    Args:
        train_df: Panel of ``unique_id`` / ``ds`` / ``y``.
        freq: Pandas frequency alias, or ``1`` for integer periods.
        exog_df: Exogenous columns to merge onto ``train_df``, as handed. None
            trains on the target's own history alone.
        lags: Autoregressive lags. Defaults to ``[1, 52]``.
        rolling_mean_window: Window of the rolling mean over lag 1.
        max_depth: Maximum tree depth.
        learning_rate: Shrinkage per boosting round.
        min_child_weight: Minimum hessian sum per leaf. Under the L1 loss this is
            not a row count, unlike LightGBM's ``min_data_in_leaf`` (measured).
        n_estimators: Boosting rounds.
        n_jobs: XGBoost threads.

    Returns:
        MLForecast: Fitted, carrying fitted values for ``forecast_fitted_values``.

    Raises:
        TypeError: If ``ds``'s dtype does not match what ``freq`` implies.
    """
    if lags is None:
        lags = [1, 52]

    train_df = _align_ds_dtype(train_df, freq)
    train_df = train_df.astype({"y": "float64"})

    if exog_df is not None:
        train_df = train_df.merge(exog_df, on=["unique_id", "ds"], how="left")

    mlfcst = MLForecast(
        models=[
            xgb.XGBRegressor(
                objective="reg:absoluteerror",
                tree_method="hist",
                max_depth=max_depth,
                learning_rate=learning_rate,
                min_child_weight=min_child_weight,
                n_estimators=n_estimators,
                n_jobs=n_jobs,
                random_state=0,
            )
        ],
        freq=freq,
        lags=lags,
        lag_transforms={1: [RollingMean(window_size=rolling_mean_window)]},  # type: ignore
    )

    mlfcst.fit(train_df, static_features=[], fitted=True)

    return mlfcst


def xgboost_weekly_save(model_obj: MLForecast, model_dir: Path) -> None:
    """Write a fitted model into ``model_dir``, which the caller creates.

    Args:
        model_obj: The fitted model ``xgboost_weekly_fit`` returned.
        model_dir: Directory to write into.
    """
    model_obj.save(model_dir)


def xgboost_weekly(
    train_df: pd.DataFrame,
    horizon: int,
    freq: str,
    future_x_df: pd.DataFrame | None = None,
    lags: list[int] | None = None,
    rolling_mean_window: int = 4,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    min_child_weight: float = 1.0,
    n_estimators: int = 1000,
    n_jobs: int = 1,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame, MLForecast]:
    """Fit through ``xgboost_weekly_fit`` and forecast ``horizon`` steps.

    Delegating is what makes the model the backtest scores the model final_fit
    registers. ``**kwargs`` is accepted for tsbricks compatibility and ignored.

    Args:
        train_df: Panel of ``unique_id`` / ``ds`` / ``y``.
        horizon: Steps to forecast.
        freq: Pandas frequency alias, or ``1`` for integer periods.
        future_x_df: The assembled exogenous frame, for the fit and the forecast.
        lags: As ``xgboost_weekly_fit``.
        rolling_mean_window: As ``xgboost_weekly_fit``.
        max_depth: As ``xgboost_weekly_fit``.
        learning_rate: As ``xgboost_weekly_fit``.
        min_child_weight: As ``xgboost_weekly_fit``.
        n_estimators: As ``xgboost_weekly_fit``.
        n_jobs: As ``xgboost_weekly_fit``.

    Returns:
        tuple: Forecast and fitted values, each ``unique_id`` / ``ds`` / ``ypred``,
            and the fitted model.
    """
    # future_x_df is the name tsbricks passes by, and **kwargs would absorb a
    # renamed parameter silently, so exogenous features would vanish with no error.
    mlfcst = xgboost_weekly_fit(
        train_df,
        freq,
        exog_df=future_x_df,
        lags=lags,
        rolling_mean_window=rolling_mean_window,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_child_weight=min_child_weight,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
    )

    forecast_df = xgboost_weekly_predict(
        mlfcst=mlfcst, horizon=horizon, future_x_df=future_x_df
    )

    fitted_df = mlfcst.forecast_fitted_values(h=1)
    fitted_df = fitted_df.rename(columns={"XGBRegressor": "ypred"})[  # type: ignore
        ["unique_id", "ds", "ypred"]
    ]

    return forecast_df, fitted_df, mlfcst


def xgboost_weekly_predict(
    mlfcst: MLForecast, horizon: int, future_x_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Predict from a fitted or ``MLForecast.load``-ed model, without refitting.

    Args:
        mlfcst: The fitted model.
        horizon: Steps to forecast.
        future_x_df: The assembled exogenous frame, passed whole as ``X_df``;
            MLForecast selects the dates it needs.

    Returns:
        pd.DataFrame: ``unique_id`` / ``ds`` / ``ypred``.
    """
    if future_x_df is not None:
        forecast_df = mlfcst.predict(h=horizon, X_df=future_x_df)
    else:
        forecast_df = mlfcst.predict(h=horizon)
    return forecast_df.rename(columns={"XGBRegressor": "ypred"})[  # type: ignore
        ["unique_id", "ds", "ypred"]
    ]
