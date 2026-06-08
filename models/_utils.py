from typing import Any

import pandas as pd


def _ensure_unique_id_column(df: Any) -> pd.DataFrame:
    """statsforecast may return unique_id as a column or as the index."""
    if "unique_id" not in df.columns and df.index.name == "unique_id":
        df = df.reset_index()
    return df
