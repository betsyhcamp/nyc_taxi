import lightgbm as lgb
import numpy as np
import pandas as pd
from mlforecast import MLForecast
from mlforecast.lag_transforms import (
    RollingMean,
)

from models._utils import _align_ds_dtype

_CALENDAR_FEATURES = [
    "fiscal_week_of_month",
    "fiscal_month",
    "weeks_in_month",
    "count_workdays",
]


def _build_future_calendar_df(
    unique_ids: np.ndarray,
    last_ds: pd.Timestamp,
    calendar_df: pd.DataFrame,
    horizon: int,
    cal_cols: list[str],
) -> pd.DataFrame:
    """Build one row per (unique_id, ds) for the horizon calendar dates
    immediately following last_ds, carrying cal_cols.

    Args:
        unique_ids: Series identifiers to cross-join against the future dates.
        last_ds: Last observed date in the training panel; future dates are
            strictly after this.
        calendar_df: Fiscal calendar table with a "ds" column and cal_cols.
        horizon: Number of future dates required.
        cal_cols: Calendar columns to attach from calendar_df.

    Returns:
        DataFrame with columns unique_id, ds, and cal_cols. Exactly
        len(unique_ids) * horizon rows.

    Raises:
        ValueError: If calendar_df does not extend horizon dates past last_ds.
    """
    future_dates = (
        calendar_df.loc[calendar_df["ds"] > last_ds, "ds"]
        .sort_values()
        .head(horizon)
        .values
    )

    if len(future_dates) != horizon:
        raise ValueError(
            f"Horizon is {horizon} but not equal to number of "
            f"future_dates is {len(future_dates)} in calendar_df"
        )

    return (
        pd.DataFrame({"unique_id": unique_ids})
        .merge(pd.DataFrame({"ds": future_dates}), how="cross")
        .merge(calendar_df[["ds"] + cal_cols], how="left", on="ds")
    )


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
        lag_transforms={1: [RollingMean(window_size=rolling_mean_window)]},  # pyright: ignore[reportArgumentType]
    )

    mlfcst.fit(train_df, static_features=[], fitted=True)

    if future_x_df is not None:
        future_calendar_df = _build_future_calendar_df(
            unique_ids=np.asarray(train_df["unique_id"].unique()),
            last_ds=train_df["ds"].max(),
            calendar_df=future_x_df,
            horizon=horizon,
            cal_cols=_CALENDAR_FEATURES,
        )

        forecast_df = mlfcst.predict(h=horizon, X_df=future_calendar_df)
    else:
        forecast_df = mlfcst.predict(h=horizon)

    forecast_df = forecast_df.rename(columns={"LGBMRegressor": "ypred"})[  # type: ignore
        ["unique_id", "ds", "ypred"]
    ]

    fitted_df = mlfcst.forecast_fitted_values(h=1)
    fitted_df = fitted_df.rename(columns={"LGBMRegressor": "ypred"})[  # type: ignore
        ["unique_id", "ds", "ypred"]
    ]

    return forecast_df, fitted_df, mlfcst
