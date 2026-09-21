import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from fcstnyctaxi.lib.exog import build_exog_frame

# ================================================
# build_exog_frame
#
# The calendar carries a column outside exog_features, so the allowlist has
# something to drop.
# ================================================

_WEEKS = pd.date_range("2025-01-05", periods=6, freq="W-SUN")
_FEATURES = ("fiscal_week_of_month", "count_workdays")


def _accept_anything(frame: pd.DataFrame) -> None:
    """Stands in for a guard, so a later one can be reached."""
    return None


@pytest.fixture
def calendar_df() -> pd.DataFrame:
    """A ds-keyed calendar, unique on ds, carrying one unselected column."""
    week_of_month = (np.arange(len(_WEEKS)) % 4) + 1
    return pd.DataFrame(
        {
            "ds": _WEEKS,
            "fiscal_week_of_month": week_of_month,
            "count_workdays": 20 + week_of_month,
            "fiscal_year": 2025,
        }
    )


def _panel(first_week_by_series: dict[int, int]) -> pd.DataFrame:
    """A panel where each series becomes active at its own week."""
    return pd.DataFrame(
        [
            {"unique_id": uid, "ds": _WEEKS[week], "y": float(uid * 10 + week)}
            for uid, first_week in first_week_by_series.items()
            for week in range(first_week, len(_WEEKS))
        ]
    )


@pytest.fixture
def rectangular_panel() -> pd.DataFrame:
    """Two series over every week, the shape today's data happens to have."""
    return _panel({10: 0, 20: 0})


@pytest.fixture
def ragged_panel() -> pd.DataFrame:
    """One series over every week and one activating later, the shape to build for."""
    return _panel({10: 0, 20: 3})


def test_the_broadcast_reproduces_a_single_key_merge(
    rectangular_panel: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """Assembling on two keys must not move a number against the ds-keyed merge."""
    expected = rectangular_panel.merge(
        calendar_df[["ds", *_FEATURES]], on="ds", how="left"
    )

    exog_df = build_exog_frame(rectangular_panel, calendar_df, exog_features=_FEATURES)
    joined = rectangular_panel.merge(exog_df, on=["unique_id", "ds"], how="left")

    assert_frame_equal(expected, joined)


def test_the_frame_carries_the_keys_and_only_the_named_features(
    rectangular_panel: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """An unnamed column would be adopted as a feature at fit, silently."""
    exog_df = build_exog_frame(rectangular_panel, calendar_df, exog_features=_FEATURES)

    assert list(exog_df.columns) == ["unique_id", "ds", *_FEATURES]


def test_the_frame_spans_every_calendar_date_not_only_the_panels(
    ragged_panel: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """One frame serves the training merge and the horizon a fold predicts into."""
    exog_df = build_exog_frame(ragged_panel, calendar_df, exog_features=_FEATURES)

    per_series = exog_df.groupby("unique_id")["ds"].nunique()
    assert set(per_series) == {len(_WEEKS)}


def test_a_ragged_panel_is_assembled_without_fan_out(
    ragged_panel: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """The row-count form this replaced fired here, on correct data."""
    exog_df = build_exog_frame(ragged_panel, calendar_df, exog_features=_FEATURES)
    joined = ragged_panel.merge(exog_df, on=["unique_id", "ds"], how="left")

    assert len(joined) == len(ragged_panel)
    assert len(exog_df) > len(ragged_panel)
    assert not joined[list(_FEATURES)].isna().any().any()


def test_no_features_yields_a_frame_of_keys(
    rectangular_panel: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """A model with no exogenous inputs still declares an entry."""
    exog_df = build_exog_frame(rectangular_panel, calendar_df, exog_features=())

    assert list(exog_df.columns) == ["unique_id", "ds"]


# ================================================
# The three guards
# ================================================


def test_a_repeated_calendar_date_raises(
    rectangular_panel: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """Unguarded it duplicates a training row per series while the run looks fine."""
    duplicated = pd.concat([calendar_df, calendar_df.iloc[[2]]], ignore_index=True)

    with pytest.raises(ValueError, match="repeats ds"):
        build_exog_frame(rectangular_panel, duplicated, exog_features=_FEATURES)


def test_a_duplicate_assembled_key_raises(
    rectangular_panel: pd.DataFrame,
    calendar_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reachable only past the calendar check, which the additional artifact ends."""
    monkeypatch.setattr(
        "fcstnyctaxi.lib.exog._require_unique_calendar_dates", _accept_anything
    )
    duplicated = pd.concat([calendar_df, calendar_df.iloc[[2]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate key row"):
        build_exog_frame(rectangular_panel, duplicated, exog_features=_FEATURES)


def test_a_join_that_changes_the_panels_row_count_raises(
    rectangular_panel: pd.DataFrame,
    calendar_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The backstop behind both key checks, so it is reachable only past both."""
    for guard in ("_require_unique_calendar_dates", "_require_unique_exog_keys"):
        monkeypatch.setattr(f"fcstnyctaxi.lib.exog.{guard}", _accept_anything)
    duplicated = pd.concat([calendar_df, calendar_df.iloc[[2]]], ignore_index=True)

    with pytest.raises(ValueError, match="changed the panel"):
        build_exog_frame(rectangular_panel, duplicated, exog_features=_FEATURES)


def test_a_panel_date_the_calendar_omits_raises(
    rectangular_panel: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """A fit accepts NaN as a value, so a gap degrades the numbers silently."""
    with pytest.raises(ValueError, match="no exogenous values"):
        build_exog_frame(
            rectangular_panel, calendar_df.iloc[:4], exog_features=_FEATURES
        )
