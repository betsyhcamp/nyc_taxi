"""Column preconditions on DataFrames, shared across lib/.

Raises ValueError on a caller-contract violation. Distinct from the runtime
gates in origin_modeling_table/gates.py, which assert on values a run produced
and return violation frames.

Flat module by design: it holds one check family. If a duplicated dtype or key
check ever appears, this becomes checks/columns.py alongside them.
"""

from __future__ import annotations

import pandas as pd


def require_columns(df: pd.DataFrame, required: list[str], frame_name: str) -> None:
    """Raise if df lacks any required column, naming the frame and what is missing.

    Checks a subset, not an exact column set, so extra columns pass. An empty
    required list passes vacuously.

    Args:
        df: Frame to check.
        required: Column names that must be present.
        frame_name: Name of the frame as the caller knows it, used in the error
            message so a missing column names its own source rather than
            surfacing as an error from somewhere downstream.

    Raises:
        ValueError: If any required column is absent, listing every missing
            name rather than only the first.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")
