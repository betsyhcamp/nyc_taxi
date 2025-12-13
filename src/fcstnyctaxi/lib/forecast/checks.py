from __future__ import annotations

from typing import Iterable, FrozenSet

try:
    import pandas as pd
except ImportError:
    # pandas dependency not present in environment
    pd = None
try:
    import polars as pl
except ImportError:
    # polars dependency not present in environment
    pl = None

_REQUIRED: FrozenSet[str] = frozenset({"unique_id", "ds", "y"})


def _is_pandas_df(obj) -> bool:
    """Determine if pandas dependency available and if object is pandas dataframe"""
    return pd is not None and isinstance(obj, pd.DataFrame)


def _is_polars_df(obj) -> bool:
    """Determine if polars dependency available and if object is polars dataframe"""
    return pl is not None and isinstance(obj, pl.DataFrame)


def missing_required_columns(df, required: Iterable[str] = _REQUIRED) -> None:
    """Assert that `df` contains the required columns; raise if any are missing.

    This function validates that the provided DataFrame (pandas or polars)
    includes all columns listed in `required`. If the object is not a
    supported DataFrame type, a TypeError is raised. If one or more
    required columns are absent, a `ValueError` is raised enumerating the
    missing names.

    Args:
        df (dataframe): The input table-like expected to be a pandas or polars
            DataFrame.
        required(Iterable[str]): Iterable of column names that must be present.
            Defaults to the canonical set `{"unique_id", "ds", "y"}`.

    Raises:
        TypeError: If `df` is not a pandas or polars DataFrame.
        ValueError: If one or more required columns are missing.
    """

    if _is_pandas_df(df) or _is_polars_df(df):
        columns = list(df.columns)
    else:
        raise TypeError("Expects pandas or polars dataframe, got ", f"{type(df).__name__}")

    missing = set(required) - set(columns)
    if missing:
        raise ValueError(f"Input DataFrame missing required columns: {sorted(missing)}")
