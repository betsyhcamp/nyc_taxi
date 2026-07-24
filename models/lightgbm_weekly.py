from typing import cast

import lightgbm as lgb
import numpy as np
import pandas as pd
from mlforecast import MLForecast
from mlforecast.lag_transforms import (
    RollingMean,
)

from fcstnyctaxi.lib.calendar_utils import _build_future_calendar_df
from models._utils import _align_ds_dtype

_CALENDAR_FEATURES = [
    "fiscal_week_of_month",
    "fiscal_month",
    "weeks_in_month",
    "count_workdays",
]


def lightgbm_weekly(
    train_df: pd.DataFrame,
    horizon: int,
    freq: str,
    future_x_df: pd.DataFrame | None = None,
    lags: list[int] | None = None,
    rolling_mean_window: int = 4,
    num_leaves: int = 31,
    learning_rate: float = 0.05,
    min_data_in_leaf: int = 125,
    n_estimators: int = 400,
    n_jobs: int = 1,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame, MLForecast]:
    """Produce a recursive LightGBM forecast via MLForecast, given historical
    data, horizon, freq. future_x_df=None skips the calendar merge and
    X_df-based prediction. **kwargs: Accepted for tsbricks compatibility;
    ignored.
    Return forecast, fitted values, model
    """
    if lags is None:
        lags = [1, 52]

    train_df = _align_ds_dtype(train_df, freq)
    train_df = train_df.astype({"y": "float64"})

    if future_x_df is not None:
        train_df = train_df.merge(
            future_x_df[["ds"] + _CALENDAR_FEATURES], on="ds", how="left"
        )

    mlfcst = MLForecast(
        models=[
            lgb.LGBMRegressor(  # pyright: ignore[reportArgumentType]
                objective="regression_l1",
                num_leaves=num_leaves,
                learning_rate=learning_rate,
                min_data_in_leaf=min_data_in_leaf,
                n_estimators=n_estimators,
                n_jobs=n_jobs,
                random_state=0,
                verbosity=-1,
            )
        ],
        freq=freq,
        lags=lags,
        lag_transforms={1: [RollingMean(window_size=rolling_mean_window)]},  # type: ignore
    )

    mlfcst.fit(train_df, static_features=[], fitted=True)

    forecast_df = lightgbm_weekly_predict(
        mlfcst=mlfcst, horizon=horizon, future_x_df=future_x_df
    )

    fitted_df = mlfcst.forecast_fitted_values(h=1)
    fitted_df = fitted_df.rename(columns={"LGBMRegressor": "ypred"})[  # type: ignore
        ["unique_id", "ds", "ypred"]
    ]

    return forecast_df, fitted_df, mlfcst


def lightgbm_weekly_predict(
    mlfcst: MLForecast, horizon: int, future_x_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Predict from an already fitted MLForecast model, without refitting.
    Derives unique_ids/last_ds from mlfcst.ts rather than a train_df
    parameter, so this works identically whether mlfcst came from a fresh
    .fit() (calibration, backtest) or MLForecast.load() (a production predict
     only pipeline)."""
    if future_x_df is not None:
        future_calendar_df = _build_future_calendar_df(
            unique_ids=np.asarray(mlfcst.ts.uids),
            last_ds=cast(pd.Timestamp, mlfcst.ts.last_dates.max()),
            calendar_df=future_x_df,
            horizon=horizon,
            cal_cols=_CALENDAR_FEATURES,
        )

        forecast_df = mlfcst.predict(h=horizon, X_df=future_calendar_df)
    else:
        forecast_df = mlfcst.predict(h=horizon)
    return forecast_df.rename(columns={"LGBMRegressor": "ypred"})[  # type: ignore
        ["unique_id", "ds", "ypred"]
    ]


def _set_lightgbm_iteration(mlfcst: MLForecast, k: int) -> None:
    """Truncate a fitted MLForecast's LightGBM model to k boosting rounds
    for prediction, without retraining."""

    mlfcst.models_["LGBMRegressor"].booster_.best_iteration = k  # type: ignore[union-attr]
