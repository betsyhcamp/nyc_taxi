from pathlib import Path
from typing import cast

import lightgbm as lgb
import numpy as np
import pandas as pd
from mlforecast import MLForecast
from mlforecast.lag_transforms import (
    RollingMean,
)

from fcstnyctaxi.lib.calendar_utils import _build_future_calendar_df
from fcstnyctaxi.models._utils import _align_ds_dtype

_CALENDAR_FEATURES = [
    "fiscal_week_of_month",
    "fiscal_month",
    "weeks_in_month",
    "count_workdays",
]


def lightgbm_weekly_fit(
    train_df: pd.DataFrame,
    freq: str,
    exog_df: pd.DataFrame | None = None,
    lags: list[int] | None = None,
    rolling_mean_window: int = 4,
    num_leaves: int = 31,
    learning_rate: float = 0.05,
    min_data_in_leaf: int = 125,
    n_estimators: int = 400,
    n_jobs: int = 1,
) -> MLForecast:
    """Fit a recursive LightGBM MLForecast on the panel it is handed.

    No ``**kwargs``, unlike the fit-predict callables: those carry it because
    tsbricks' ``resolve_model`` merges ``predict_params`` into what it forwards,
    and a fit-only call receives ``hyperparameters`` alone. Being strict is what
    makes a misspelled hyperparameter a ``TypeError`` here rather than a value
    silently dropped.

    ``exog_df`` rather than ``future_x_df``: tsbricks never resolves this
    callable, so the accurate name is free. It is merged as handed, with no
    column selection, so whoever assembles it decides the model's feature set.

    Args:
        train_df: Panel of ``unique_id`` / ``ds`` / ``y``.
        freq: Pandas frequency alias, or ``1`` for integer periods.
        exog_df: Exogenous columns to merge onto ``train_df``. None trains on
            the target's own history alone.
        lags: Autoregressive lags. Defaults to ``[1, 52]``.
        rolling_mean_window: Window of the rolling mean over lag 1.
        num_leaves: LightGBM leaves per tree.
        learning_rate: LightGBM learning rate.
        min_data_in_leaf: LightGBM minimum observations per leaf.
        n_estimators: Boosting rounds.
        n_jobs: LightGBM threads. 1 keeps a run reproducible.

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
        train_df = train_df.merge(exog_df, on="ds", how="left")

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

    return mlfcst


def lightgbm_weekly_save(model_obj: MLForecast, model_dir: Path) -> None:
    """Write a fitted model to ``model_dir``, which the caller creates.

    Positional, unlike ``trim_to_allowlist``: the two parameters have unrelated
    types, so a transposition fails immediately rather than half-succeeding.

    The caller owns creating the directory because the impl's "wrote something"
    check has to hold for any contributor's save callable. ``MLForecast.save``
    happens to create a missing one, measured, so nothing here enforces that
    precondition and a caller leaning on this implementation breaks on the next
    model.

    Args:
        model_obj: The fitted model ``lightgbm_weekly_fit`` returned.
        model_dir: Directory to write into.
    """
    model_obj.save(model_dir)


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

    Delegates to lightgbm_weekly_fit rather than fitting here, so the model the
    backtest scores and the model final_fit registers are one implementation.
    """
    # future_x_df is the name tsbricks passes by, and **kwargs would absorb a
    # renamed parameter silently, so exogenous features would vanish with no error.
    exog_df = (
        future_x_df[["ds"] + _CALENDAR_FEATURES] if future_x_df is not None else None
    )

    mlfcst = lightgbm_weekly_fit(
        train_df,
        freq,
        exog_df=exog_df,
        lags=lags,
        rolling_mean_window=rolling_mean_window,
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        min_data_in_leaf=min_data_in_leaf,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
    )

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
