"""Frame preconditions shared across this package.

Private to origin_modeling_table so the package keeps depending on nothing else in
lib/, which is what makes it liftable for the work-data port.
"""

from __future__ import annotations

import pandas as pd


def require_columns(df: pd.DataFrame, required: list[str], frame_name: str) -> None:
    """Raise if df lacks any required column, naming the frame and what is missing."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")
