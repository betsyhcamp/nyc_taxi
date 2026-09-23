import numpy as np
import pandas as pd
import pytest

from fcstnyctaxi.lib.exog import build_exog_frame
from fcstnyctaxi.schemas.run_outputs import ADDITIONAL_EXOG_REQUIRED_COLUMNS

# ================================================
# build_exog_frame
#
# The additional exogenous file drives and the calendar is broadcast onto it, so
# the calendar carries a week the file does not and a column outside
# exog_features: one so the row set has a side to come from, the other so the
# selection has something to drop.
# ================================================

_WEEKS = pd.date_range("2025-01-05", periods=6, freq="W-SUN")
# Three nested spans, as production has them: the file stops short of the calendar,
# so a reversed join shows in the row set rather than only in column order, and the
# panel stops short of the file, so a calendar can be shortened to sit between them.
_FILE_WEEKS = _WEEKS[:5]
_PANEL_WEEKS = _WEEKS[:4]
_SERIES = (10, 20)

# One column from each contract, so the selection spans the merged frame and a
# select-before-merge cannot pass by keeping the file's own columns alone.
_FEATURES = ("fiscal_week_of_month", "week_sin")


def _accept_anything(*args: object, **kwargs: object) -> None:
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
            "count_workdays": 5 - (np.arange(len(_WEEKS)) % 2),
            "fiscal_year": 2025,
        }
    )


def _additional_exog(
    calendar_df: pd.DataFrame,
    series_ids: tuple = _SERIES,
    weeks: pd.Index = _FILE_WEEKS,
) -> pd.DataFrame:
    """The exogenous file as Feature's producer emits it: every series over every
    week it covers, its features derived from the calendar and from ds."""
    frame = pd.MultiIndex.from_product(
        [list(series_ids), weeks], names=["unique_id", "ds"]
    ).to_frame(index=False)
    workdays = frame["ds"].map(calendar_df.set_index("ds")["count_workdays"])
    angle = 2 * np.pi * frame["ds"].dt.dayofyear / 365.25
    return frame.assign(
        holiday_days_in_week=5 - workdays,
        week_sin=np.sin(angle),
        week_cos=np.cos(angle),
    )


@pytest.fixture
def additional_exog_df(calendar_df: pd.DataFrame) -> pd.DataFrame:
    """The file both series are covered by, over every week it spans."""
    return _additional_exog(calendar_df)


def _panel(first_week_by_series: dict[int, int]) -> pd.DataFrame:
    """A panel where each series becomes active at its own week."""
    return pd.DataFrame(
        [
            {"unique_id": uid, "ds": _PANEL_WEEKS[week], "y": float(uid * 10 + week)}
            for uid, first_week in first_week_by_series.items()
            for week in range(first_week, len(_PANEL_WEEKS))
        ]
    )


@pytest.fixture
def rectangular_panel() -> pd.DataFrame:
    """Two series over every week, the shape today's data happens to have."""
    return _panel({10: 0, 20: 0})


@pytest.fixture
def ragged_panel() -> pd.DataFrame:
    """One series over every week and one activating later, the shape to build for."""
    return _panel({10: 0, 20: 2})


def test_the_fixture_has_the_properties_the_published_file_has(
    additional_exog_df: pd.DataFrame, ragged_panel: pd.DataFrame
) -> None:
    """A broken fixture reads as a broken check, so it states its own shape."""
    assert tuple(additional_exog_df.columns) == ADDITIONAL_EXOG_REQUIRED_COLUMNS
    assert not additional_exog_df.duplicated(["unique_id", "ds"]).any()
    assert not additional_exog_df.isna().any().any()

    file_keys = set(zip(*additional_exog_df[["unique_id", "ds"]].values.T, strict=True))
    panel_keys = set(zip(*ragged_panel[["unique_id", "ds"]].values.T, strict=True))
    assert panel_keys <= file_keys


def test_the_frame_carries_the_keys_and_only_the_named_features(
    rectangular_panel: pd.DataFrame,
    calendar_df: pd.DataFrame,
    additional_exog_df: pd.DataFrame,
) -> None:
    """An unnamed column would be adopted as a feature at fit, silently."""
    exog_df = build_exog_frame(
        rectangular_panel, calendar_df, additional_exog_df, exog_features=_FEATURES
    )

    assert list(exog_df.columns) == ["unique_id", "ds", *_FEATURES]


def test_a_ragged_panel_is_assembled_without_fan_out(
    ragged_panel: pd.DataFrame,
    calendar_df: pd.DataFrame,
    additional_exog_df: pd.DataFrame,
) -> None:
    """The file decides the row set and the panel merge keeps its own count.
    The calendar holds a week the file does not, so a reversal shows up here."""
    exog_df = build_exog_frame(
        ragged_panel, calendar_df, additional_exog_df, exog_features=_FEATURES
    )
    joined = ragged_panel.merge(exog_df, on=["unique_id", "ds"], how="left")

    assert set(map(tuple, exog_df[["unique_id", "ds"]].to_numpy())) == set(
        map(tuple, additional_exog_df[["unique_id", "ds"]].to_numpy())
    )
    assert len(joined) == len(ragged_panel)
    assert len(exog_df) > len(ragged_panel)
    assert not joined[list(_FEATURES)].isna().any().any()


def test_no_features_yields_a_frame_of_keys(
    rectangular_panel: pd.DataFrame,
    calendar_df: pd.DataFrame,
    additional_exog_df: pd.DataFrame,
) -> None:
    """A model with no exogenous inputs still declares an entry."""
    exog_df = build_exog_frame(
        rectangular_panel, calendar_df, additional_exog_df, exog_features=()
    )

    assert list(exog_df.columns) == ["unique_id", "ds"]


def test_a_calendar_short_of_the_file_leaves_the_panels_rows_intact(
    rectangular_panel: pd.DataFrame,
    calendar_df: pd.DataFrame,
    additional_exog_df: pd.DataFrame,
) -> None:
    """Current behavior, not a guard: nothing sees the nulls past the panel."""
    # Every panel week, short of the file, so coverage has nothing to say.
    short = calendar_df[calendar_df["ds"].isin(_PANEL_WEEKS)]

    exog_df = build_exog_frame(
        rectangular_panel, short, additional_exog_df, exog_features=_FEATURES
    )

    past_the_calendar = ~exog_df["ds"].isin(short["ds"])
    assert past_the_calendar.any()
    assert exog_df.loc[past_the_calendar, "fiscal_week_of_month"].isna().all()
    assert not exog_df.loc[~past_the_calendar, list(_FEATURES)].isna().any().any()


# ================================================
# The guards
# ================================================


def test_a_repeated_calendar_date_raises(
    rectangular_panel: pd.DataFrame,
    calendar_df: pd.DataFrame,
    additional_exog_df: pd.DataFrame,
) -> None:
    """Unguarded it duplicates a training row per series while the run looks fine."""
    duplicated = pd.concat([calendar_df, calendar_df.iloc[[2]]], ignore_index=True)

    with pytest.raises(ValueError, match="repeats ds"):
        build_exog_frame(
            rectangular_panel, duplicated, additional_exog_df, exog_features=_FEATURES
        )


def test_a_duplicate_assembled_key_raises(
    rectangular_panel: pd.DataFrame,
    calendar_df: pd.DataFrame,
    additional_exog_df: pd.DataFrame,
) -> None:
    """A repeated key in the file, which is a real join now rather than a product."""
    duplicated = pd.concat(
        [additional_exog_df, additional_exog_df.iloc[[2]]], ignore_index=True
    )

    with pytest.raises(ValueError, match="duplicate key row"):
        build_exog_frame(
            rectangular_panel, calendar_df, duplicated, exog_features=_FEATURES
        )


def test_a_join_that_changes_the_panels_row_count_raises(
    rectangular_panel: pd.DataFrame,
    calendar_df: pd.DataFrame,
    additional_exog_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The backstop behind the key checks, so it is reachable only past them."""
    monkeypatch.setattr(
        "fcstnyctaxi.lib.exog._require_unique_exog_keys", _accept_anything
    )
    duplicated = pd.concat(
        [additional_exog_df, additional_exog_df.iloc[[2]]], ignore_index=True
    )

    with pytest.raises(ValueError, match="changed the panel"):
        build_exog_frame(
            rectangular_panel, calendar_df, duplicated, exog_features=_FEATURES
        )


def test_a_panel_key_the_file_omits_raises(
    rectangular_panel: pd.DataFrame,
    calendar_df: pd.DataFrame,
    additional_exog_df: pd.DataFrame,
) -> None:
    """A file short of the panel surfaces here; the cross join covered every
    series-date pair by construction, so this was unrepresentable."""
    missing_a_series = additional_exog_df[additional_exog_df["unique_id"] != 20]

    with pytest.raises(ValueError, match="no exogenous values"):
        build_exog_frame(
            rectangular_panel,
            calendar_df,
            missing_a_series,
            exog_features=_FEATURES,
        )


# ================================================
# _require_comparable_series_ids
#
# Pandas refuses exactly the numeric-against-text key pairings and merges every
# within-family one, so the guard fires on that split and not on dtype equality.
# ================================================


def test_a_numeric_panel_id_against_a_text_file_id_raises_naming_both(
    rectangular_panel: pd.DataFrame,
    calendar_df: pd.DataFrame,
    additional_exog_df: pd.DataFrame,
) -> None:
    """Pandas names two dtypes; the guard names the two artifacts behind them."""
    text_ids = additional_exog_df.assign(
        unique_id=additional_exog_df["unique_id"].astype(str)
    )

    with pytest.raises(ValueError, match="key series differently") as excinfo:
        build_exog_frame(
            rectangular_panel, calendar_df, text_ids, exog_features=_FEATURES
        )

    message = str(excinfo.value)
    assert "panel" in message and "additional exogenous" in message


@pytest.mark.parametrize(
    ("panel_dtype", "file_dtype"),
    [("int64", "Int64"), ("int64", "int32"), ("object", "string")],
    ids=["nullable_backing", "narrower_width", "object_against_string"],
)
def test_ids_of_one_family_merge_rather_than_raising(
    rectangular_panel: pd.DataFrame,
    calendar_df: pd.DataFrame,
    additional_exog_df: pd.DataFrame,
    panel_dtype: str,
    file_dtype: str,
) -> None:
    """Dtype equality would refuse all three and pandas merges all three; pandas
    2 reads parquet strings as `object` or `string` by writer, so that pair is live."""
    as_text = panel_dtype == "object"
    panel = rectangular_panel.assign(
        unique_id=rectangular_panel["unique_id"].astype(str if as_text else panel_dtype)
    )
    if as_text:
        panel = panel.astype({"unique_id": "object"})
    file_ids = additional_exog_df["unique_id"]
    exog_file = additional_exog_df.assign(
        unique_id=(file_ids.astype(str) if as_text else file_ids).astype(file_dtype)
    )
    # Self-check: the two dtypes really do differ, so the guard has a case to pass.
    assert str(panel["unique_id"].dtype) != str(exog_file["unique_id"].dtype)

    exog_df = build_exog_frame(panel, calendar_df, exog_file, exog_features=_FEATURES)

    assert len(exog_df) == len(exog_file)


def test_a_ds_unit_mismatch_merges_rather_than_raising(
    rectangular_panel: pd.DataFrame,
    calendar_df: pd.DataFrame,
    additional_exog_df: pd.DataFrame,
) -> None:
    """Widening the guard to ds would refuse a unit pair pandas resolves silently."""
    microseconds = additional_exog_df.astype({"ds": "datetime64[us]"})
    # Self-check: the panel and the file disagree on the unit.
    assert rectangular_panel["ds"].dtype != microseconds["ds"].dtype

    exog_df = build_exog_frame(
        rectangular_panel, calendar_df, microseconds, exog_features=_FEATURES
    )

    assert len(exog_df) == len(microseconds)
