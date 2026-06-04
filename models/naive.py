from typing import Any

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import Naive


def naive_weekly(train_df: pd.DataFrame, horizon: int, **kwargs) -> pd.DataFrame:
    train = train_df.copy()
    dtype_map = {"ds": "int64", "y": "float64"}
    train = train.astype(dtype_map)

    sf = StatsForecast(models=[Naive()], freq=1)
    sf.fit(df=train)
    forecast = _ensure_unique_id_column(sf.predict(h=horizon))

    return forecast.rename(columns={"Naive": "ypred"})[["unique_id", "ds", "ypred"]]


def _ensure_unique_id_column(df: Any) -> pd.DataFrame:
    """statsforecast may return unique_id as a column or as the index."""
    if "unique_id" not in df.columns and df.index.name == "unique_id":
        df = df.reset_index()
    return df
