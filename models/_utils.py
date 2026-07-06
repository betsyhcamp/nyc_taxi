from typing import Any

import pandas as pd


def _ensure_unique_id_column(df: Any) -> pd.DataFrame:
    """statsforecast may return unique_id as a column or as the index."""
    if "unique_id" not in df.columns and df.index.name == "unique_id":
        df = df.reset_index()
    return df


def _align_ds_dtype(df: pd.DataFrame, freq: int | str) -> pd.DataFrame:
    """Validate that ``ds``'s dtype matches what ``freq`` implies.

    Args:
        df: Panel DataFrame with a ``ds`` column.
        freq: ``1`` for integer-indexed periods, otherwise a pandas
            frequency alias (e.g. ``"W-SUN"``) for datetime-indexed periods.

    Returns:
        ``df`` unchanged.

    Raises:
        TypeError: If ``ds``'s dtype doesn't match what ``freq`` implies.
    """
    is_integer_freq = freq == 1
    is_integer_ds = pd.api.types.is_integer_dtype(df["ds"])
    is_datetime_ds = pd.api.types.is_datetime64_any_dtype(df["ds"])

    if is_integer_freq and not is_integer_ds:
        raise TypeError(
            f"freq=1 requires integer 'ds' column, got dtype {df['ds'].dtype}."
        )
    if not is_integer_freq and not is_datetime_ds:
        raise TypeError(
            f"freq={freq!r} requires datetime 'ds' column, got dtype {df['ds'].dtype}."
        )
    return df
