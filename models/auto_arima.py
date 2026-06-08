import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

from models._utils import _ensure_unique_id_column


def auto_arima_weekly(
    train_df: pd.DataFrame, horizon: int, freq=1, season_length=52, **kwargs
) -> tuple[pd.DataFrame, pd.DataFrame, StatsForecast]:
    """Fit AutoARIMA per series via StatsForecast given historical data, horizon, freq
    Return forecast, fitted, model. **kwargs: Accepted for tsbricks
    compatibility; ignored."""
    if freq == 1:
        dtype_map = {"ds": "int64", "y": "float64"}
    else:
        dtype_map = {"y": "float64"}
    train_df = train_df.astype(dtype_map)

    sf = StatsForecast(
        models=[AutoARIMA(season_length=season_length)],
        freq=freq,
    )

    preds = _ensure_unique_id_column(sf.forecast(df=train_df, h=horizon, fitted=True))

    forecast_df = preds.rename(columns={"AutoARIMA": "ypred"})[
        ["ds", "unique_id", "ypred"]
    ]

    fitted_raw = _ensure_unique_id_column(sf.forecast_fitted_values())
    fitted_df = fitted_raw.rename(columns={"AutoARIMA": "ypred"})[
        ["unique_id", "ds", "ypred"]
    ]

    return forecast_df, fitted_df, sf
