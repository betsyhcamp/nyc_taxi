import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS

from models._utils import _align_ds_dtype, _ensure_unique_id_column


def auto_ets_weekly(
    train_df: pd.DataFrame, horizon: int, freq, season_length, **kwargs
) -> tuple[pd.DataFrame, pd.DataFrame, StatsForecast]:
    """Fit AutoETS per series via StatsForecast given historical data, horizon, freq
    Return forecast, fitted, model. **kwargs: Accepted for tsbricks
    compatibility; ignored."""

    train_df = _align_ds_dtype(train_df, freq)
    train_df = train_df.astype({"y": "float64"})

    sf = StatsForecast(
        models=[AutoETS(season_length=season_length)],
        freq=freq,
    )

    forecast_df = _ensure_unique_id_column(
        sf.forecast(df=train_df, h=horizon, fitted=True)
    )

    forecast_df = forecast_df.rename(columns={"AutoETS": "ypred"})[
        ["ds", "unique_id", "ypred"]
    ]

    fitted_raw = _ensure_unique_id_column(sf.forecast_fitted_values())
    fitted_df = fitted_raw.rename(columns={"AutoETS": "ypred"})[
        ["unique_id", "ds", "ypred"]
    ]

    return forecast_df, fitted_df, sf
