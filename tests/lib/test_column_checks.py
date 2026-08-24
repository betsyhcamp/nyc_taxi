from __future__ import annotations

import pandas as pd
import pytest

from fcstnyctaxi.lib.column_checks import require_columns


def test_require_columns_passes_when_all_present() -> None:
    """Returns None rather than raising when every required column is there."""
    df = pd.DataFrame({"a": [1], "b": [2]})
    assert require_columns(df, ["a", "b"], "df") is None


def test_require_columns_ignores_extra_columns() -> None:
    """Requires a subset, not an exact column set, so extras pass."""
    df = pd.DataFrame({"a": [1], "b": [2], "extra": [3]})
    assert require_columns(df, ["a"], "df") is None


def test_require_columns_passes_vacuously_on_empty_required() -> None:
    """An empty requirement list is a silent pass, and that is intentional.

    Pinned rather than left implied: a caller that builds `required` dynamically
    and produces [] gets no check at all, so the vacuous pass should be a stated
    contract instead of something a reader has to infer from the comprehension.
    """
    df = pd.DataFrame({"a": [1]})
    assert require_columns(df, [], "df") is None


def test_require_columns_raises_naming_frame_and_missing_columns() -> None:
    """The message names which frame is at fault, which a bare KeyError cannot."""
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match=r"panel_df is missing required columns"):
        require_columns(df, ["a", "b"], "panel_df")


def test_require_columns_reports_every_missing_column_not_just_the_first() -> None:
    """One call surfaces all missing columns instead of one fix-and-retry each."""
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError) as excinfo:
        require_columns(df, ["a", "b", "c"], "df")
    assert "'b'" in str(excinfo.value)
    assert "'c'" in str(excinfo.value)
