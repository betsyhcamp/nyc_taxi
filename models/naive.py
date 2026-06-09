import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import Naive

from models._utils import _ensure_unique_id_column


def naive_weekly(
    train_df: pd.DataFrame, horizon: int, freq=1, **kwargs
) -> pd.DataFrame:
    """Produce Naive forecast given historical data, horizon, frequency.
    **kwargs: Accepted for tsbricks compatibility; ignored."""

    if freq == 1:
        dtype_map = {"ds": "int64", "y": "float64"}
    else:
        dtype_map = {"y": "float64"}
    train_df = train_df.astype(dtype_map)

    sf = StatsForecast(models=[Naive()], freq=freq)
    sf.fit(df=train_df)
    forecast_df = _ensure_unique_id_column(sf.predict(h=horizon))

    return forecast_df.rename(columns={"Naive": "ypred"})[["unique_id", "ds", "ypred"]]
