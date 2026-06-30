import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import Naive

from models._utils import _align_ds_dtype, _ensure_unique_id_column


def naive_weekly(train_df: pd.DataFrame, horizon: int, freq, **kwargs) -> pd.DataFrame:
    """Produce Naive forecast given historical data, horizon, frequency.
    **kwargs: Accepted for tsbricks compatibility; ignored."""

    train_df = _align_ds_dtype(train_df, freq)
    train_df = train_df.astype({"y": "float64"})

    sf = StatsForecast(models=[Naive()], freq=freq)
    sf.fit(df=train_df)
    forecast_df = _ensure_unique_id_column(sf.predict(h=horizon))

    return forecast_df.rename(columns={"Naive": "ypred"})[["unique_id", "ds", "ypred"]]
