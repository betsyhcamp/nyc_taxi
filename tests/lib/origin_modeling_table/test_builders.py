from __future__ import annotations

import pandas as pd
import pytest
from mlforecast.lag_transforms import RollingMean

from fcstnyctaxi.lib.origin_modeling_table.builders import (
    _require_columns,
    attach_workday_progress,
    build_weekly_features,
    enumerate_origins,
    trim_incomplete_series_months,
)

# ================================================
# Fixtures
#
# 9 consecutive W-SUN weeks spanning two fiscal months with DIFFERENT week
# counts — 202501 has 4 weeks, 202502 has 5. The differing counts matter: a
# predicate that hardcoded 4 instead of reading weeks_in_month would pass on a
# fixture where every month were the same length.
#
#   id=10: complete in both months (4 + 5 = 9 rows)      -> both pairs kept
#   id=20: complete in 202501 (4 rows), only 3 of 5
#          weeks in 202502 (3 rows)                      -> 202502 dropped
#
# Worked by hand: 16 panel rows in, 13 out, and exactly one dropped pair
# (20, 202502) with weeks_present=3 against weeks_in_month=5.
# ================================================

WEEKS = pd.date_range("2025-01-05", periods=9, freq="W-SUN")


@pytest.fixture
def calendar_df() -> pd.DataFrame:
    """202501 has 4 weeks / 17 workdays; 202502 has 5 weeks / 21 workdays.

    count_workdays is deliberately non-uniform so an off-by-one in the cumulative
    sum shows up as a wrong number rather than a coincidentally right one. Cumulative
    workdays are 5, 9, 14, 17 for 202501 and 5, 10, 14, 19, 21 for 202502.
    """
    return pd.DataFrame(
        {
            "ds": WEEKS,
            "fiscal_year_month": [202501] * 4 + [202502] * 5,
            "fiscal_week_of_month": [1, 2, 3, 4, 1, 2, 3, 4, 5],
            "weeks_in_month": [4] * 4 + [5] * 5,
            "origin_month_fraction_elapsed": [
                0.25,
                0.5,
                0.75,
                1.0,
                0.2,
                0.4,
                0.6,
                0.8,
                1.0,
            ],
            "count_workdays": [5, 4, 5, 3, 5, 5, 4, 5, 2],
        }
    )


@pytest.fixture
def panel_df() -> pd.DataFrame:
    """id=10 complete throughout; id=20 partial in 202502 (weeks 1-3 only)."""
    id_10 = pd.DataFrame({"unique_id": 10, "ds": WEEKS, "y": range(10, 100, 10)})
    id_20 = pd.DataFrame({"unique_id": 20, "ds": WEEKS[:7], "y": range(100, 800, 100)})
    return pd.concat([id_10, id_20], ignore_index=True)


@pytest.fixture
def duplicated_week_panel_df() -> pd.DataFrame:
    """id=30 has 4 rows in the 4-week 202501, but only 3 DISTINCT weeks.

    Week 2 appears twice and week 4 never. A row count would say 4 and keep the
    pair; a distinct-week count says 3 and drops it. This is the only fixture
    that separates nunique from size.
    """
    return pd.DataFrame(
        {
            "unique_id": 30,
            "ds": [WEEKS[0], WEEKS[1], WEEKS[1], WEEKS[2]],
            "y": [1, 2, 3, 4],
        }
    )


@pytest.fixture
def complete_panel_df() -> pd.DataFrame:
    """Two gapless series, for feature tests that need a regular frequency."""
    id_10 = pd.DataFrame({"unique_id": 10, "ds": WEEKS, "y": range(10, 100, 10)})
    id_20 = pd.DataFrame({"unique_id": 20, "ds": WEEKS, "y": range(100, 1000, 100)})
    return pd.concat([id_10, id_20], ignore_index=True)


# ================================================
# _require_columns
# ================================================


def test_require_columns_passes_when_all_present() -> None:
    """Returns None rather than raising when every required column is there."""
    df = pd.DataFrame({"a": [1], "b": [2]})
    assert _require_columns(df, ["a", "b"], "df") is None


def test_require_columns_ignores_extra_columns() -> None:
    """Requires a subset, not an exact column set, so extras pass."""
    df = pd.DataFrame({"a": [1], "b": [2], "extra": [3]})
    assert _require_columns(df, ["a"], "df") is None


def test_require_columns_raises_naming_frame_and_missing_columns() -> None:
    """The message names which frame is at fault, which a bare KeyError cannot."""
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match=r"panel_df is missing required columns"):
        _require_columns(df, ["a", "b"], "panel_df")


def test_require_columns_reports_every_missing_column_not_just_the_first() -> None:
    """One call surfaces all missing columns instead of one fix-and-retry each."""
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError) as excinfo:
        _require_columns(df, ["a", "b", "c"], "df")
    assert "'b'" in str(excinfo.value)
    assert "'c'" in str(excinfo.value)


# ================================================
# trim_incomplete_series_months — trim behavior
# ================================================


def test_trim_keeps_complete_pairs_and_drops_the_partial_one(
    panel_df: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """16 panel rows in, 13 out: only id=20's partial 202502 is removed."""
    trimmed, _ = trim_incomplete_series_months(panel_df, calendar_df)

    assert len(trimmed) == 13
    kept = set(zip(trimmed["unique_id"], trimmed["ds"].dt.month, strict=True))
    assert (10, 1) in kept and (10, 2) in kept  # id=10 complete in both months
    assert (20, 1) in kept  # id=20 complete in 202501


def test_trim_drops_whole_pairs_never_a_subset_of_one(
    panel_df: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """The keep predicate is group-constant, so no row of a dropped pair survives."""
    trimmed, _ = trim_incomplete_series_months(panel_df, calendar_df)

    id_20_feb = trimmed[(trimmed["unique_id"] == 20) & (trimmed["ds"] >= WEEKS[4])]
    assert id_20_feb.empty


def test_trim_preserves_panel_columns(
    panel_df: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """Returns the panel's three columns, not the calendar columns joined in."""
    trimmed, _ = trim_incomplete_series_months(panel_df, calendar_df)
    assert list(trimmed.columns) == ["unique_id", "ds", "y"]


def test_trim_reads_weeks_in_month_rather_than_assuming_a_fixed_length(
    panel_df: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """id=10's 202502 has 5 weeks and is kept; a hardcoded 4 would drop it."""
    trimmed, dropped = trim_incomplete_series_months(panel_df, calendar_df)

    id_10_feb = trimmed[(trimmed["unique_id"] == 10) & (trimmed["ds"] >= WEEKS[4])]
    assert len(id_10_feb) == 5
    assert 10 not in set(dropped["unique_id"])


# ================================================
# trim_incomplete_series_months — dropped_pairs contract
# ================================================


def test_dropped_pairs_names_exactly_the_dropped_pair(
    panel_df: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """Pins the whole frame: one row, four columns, exact values."""
    _, dropped = trim_incomplete_series_months(panel_df, calendar_df)

    expected = pd.DataFrame(
        {
            "unique_id": [20],
            "fiscal_year_month": [202502],
            "weeks_present": [3],
            "weeks_in_month": [5],
        }
    )
    pd.testing.assert_frame_equal(dropped, expected, check_dtype=False)


def test_dropped_pairs_is_pair_grain_not_row_grain(
    panel_df: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """One row per dropped (series, month), though 3 panel rows were removed."""
    _, dropped = trim_incomplete_series_months(panel_df, calendar_df)
    assert len(dropped) == 1
    assert not dropped.duplicated(["unique_id", "fiscal_year_month"]).any()


def test_trim_counts_distinct_weeks_so_a_duplicated_week_cannot_fake_completeness(
    duplicated_week_panel_df: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """4 rows in a 4-week month, but only 3 distinct weeks — must still drop."""
    trimmed, _ = trim_incomplete_series_months(duplicated_week_panel_df, calendar_df)
    assert trimmed.empty


def test_dropped_pairs_weeks_present_counts_distinct_weeks_not_rows(
    duplicated_week_panel_df: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """weeks_present reports distinct weeks, so a duplicate does not inflate it."""
    _, dropped = trim_incomplete_series_months(duplicated_week_panel_df, calendar_df)
    assert dropped.loc[0, "weeks_present"] == 3  # not 4, the row count


def test_dropped_pairs_is_empty_with_declared_columns_when_nothing_is_dropped(
    complete_panel_df: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """An empty result still carries the four declared columns, not zero columns."""
    trimmed, dropped = trim_incomplete_series_months(complete_panel_df, calendar_df)

    assert len(trimmed) == len(complete_panel_df)
    assert dropped.empty
    assert list(dropped.columns) == [
        "unique_id",
        "fiscal_year_month",
        "weeks_present",
        "weeks_in_month",
    ]


# ================================================
# trim_incomplete_series_months — guard clauses
#
# The postcondition (no partial pair survives) has no test: it is unreachable
# by construction, since the keep predicate is group-constant and therefore
# removes whole pairs. It guards against a future edit breaking that property,
# not against any input.
# ================================================


@pytest.mark.parametrize("missing", ["unique_id", "ds", "y"])
def test_trim_raises_when_panel_lacks_a_required_column(
    panel_df: pd.DataFrame, calendar_df: pd.DataFrame, missing: str
) -> None:
    """Each panel column is required; dropping any one of them halts the trim."""
    with pytest.raises(ValueError, match=r"panel_df is missing required columns"):
        trim_incomplete_series_months(panel_df.drop(columns=[missing]), calendar_df)


@pytest.mark.parametrize(
    "missing", ["fiscal_year_month", "fiscal_week_of_month", "weeks_in_month"]
)
def test_trim_raises_when_calendar_lacks_a_required_column(
    panel_df: pd.DataFrame, calendar_df: pd.DataFrame, missing: str
) -> None:
    """Each calendar column is required; dropping any one of them halts the trim."""
    with pytest.raises(ValueError, match=r"calendar_df is missing required columns"):
        trim_incomplete_series_months(panel_df, calendar_df.drop(columns=[missing]))


def test_trim_raises_when_calendar_does_not_cover_a_panel_date(
    panel_df: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """An unlabeled row would otherwise vanish silently, absent from dropped_pairs."""
    orphan = pd.DataFrame(
        {"unique_id": [10], "ds": [pd.Timestamp("1999-01-03")], "y": [1]}
    )
    panel_with_orphan = pd.concat([panel_df, orphan], ignore_index=True)

    with pytest.raises(ValueError, match=r"does not cover 1 panel date"):
        trim_incomplete_series_months(panel_with_orphan, calendar_df)


def test_trim_coverage_error_reports_distinct_dates_not_rows(
    panel_df: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """One bad date shared by many series is one problem, not many."""
    orphans = pd.DataFrame(
        {
            "unique_id": [10, 20],
            "ds": [pd.Timestamp("1999-01-03")] * 2,
            "y": [1, 2],
        }
    )
    panel_with_orphans = pd.concat([panel_df, orphans], ignore_index=True)

    with pytest.raises(ValueError, match=r"does not cover 1 panel date"):
        trim_incomplete_series_months(panel_with_orphans, calendar_df)


# ================================================
# build_weekly_features
# ================================================


def test_build_weekly_features_renames_ds_to_feature_ds(
    complete_panel_df: pd.DataFrame,
) -> None:
    """The rename happens at the source, so no caller sees a bare ds."""
    feats = build_weekly_features(
        complete_panel_df, "W-SUN", lags=[1], lag_transforms={}
    )
    assert "feature_ds" in feats.columns
    assert "ds" not in feats.columns


def test_build_weekly_features_keeps_every_panel_row(
    complete_panel_df: pd.DataFrame,
) -> None:
    """dropna=False: early weeks with NaN lags must survive for Layer 3's join."""
    feats = build_weekly_features(
        complete_panel_df, "W-SUN", lags=[1], lag_transforms={1: [RollingMean(4)]}
    )
    assert len(feats) == len(complete_panel_df)
    assert feats["lag1"].isna().sum() == 2  # one first week per series


def test_build_weekly_features_emits_native_mlforecast_names(
    complete_panel_df: pd.DataFrame,
) -> None:
    """Native names are kept, which is what MLF_FEATURES is drift-gated against."""
    feats = build_weekly_features(
        complete_panel_df, "W-SUN", lags=[1], lag_transforms={1: [RollingMean(4)]}
    )
    emitted = set(feats.columns) - {"unique_id", "feature_ds", "y"}
    assert emitted == {"lag1", "rolling_mean_lag1_window_size4"}


def test_build_weekly_features_lag1_is_the_prior_week_within_each_series(
    complete_panel_df: pd.DataFrame,
) -> None:
    """lag1 never crosses a series boundary; only each series' first week is NaN."""
    feats = build_weekly_features(
        complete_panel_df, "W-SUN", lags=[1], lag_transforms={}
    ).sort_values(["unique_id", "feature_ds"])

    prior_y = feats.groupby("unique_id")["y"].shift(1)
    observed = feats["lag1"].notna()
    assert (feats.loc[observed, "lag1"] == prior_y[observed]).all()
    assert feats.loc[~observed, "feature_ds"].tolist() == [WEEKS[0], WEEKS[0]]


# ================================================
# enumerate_origins
#
# Against the calendar fixture, only 202502 can have origins: origins targeting
# 202501 are dropped (it is the first month, with no prior month to learn from)
# and the final week is dropped (its shifted target month is NaN). So the spine is
# 5 rows — weeks_actualized 0 through 4 — and every one carries weeks_in_month=5.
# ================================================


def test_enumerate_origins_returns_the_expected_spine(
    calendar_df: pd.DataFrame,
) -> None:
    """Pins the whole frame: 5 origins for 202502, weeks_actualized 0 through 4."""
    expected = pd.DataFrame(
        {
            "target_month": [202502] * 5,
            "forecast_origin_date": WEEKS[3:8],
            "weeks_actualized": [0, 1, 2, 3, 4],
            "weeks_in_month": [5] * 5,
        }
    )
    pd.testing.assert_frame_equal(
        enumerate_origins(calendar_df), expected, check_dtype=False
    )


def test_enumerate_origins_rolls_a_month_end_origin_forward(
    calendar_df: pd.DataFrame,
) -> None:
    """At fraction_elapsed == 1 the month is done, so the origin targets the next."""
    spine = enumerate_origins(calendar_df)
    month_end = spine[spine["forecast_origin_date"] == WEEKS[3]]

    assert len(month_end) == 1
    assert month_end["target_month"].item() == 202502
    assert month_end["weeks_actualized"].item() == 0


def test_enumerate_origins_takes_weeks_in_month_from_the_target_not_the_origin(
    calendar_df: pd.DataFrame,
) -> None:
    """The rolled-forward origin sits in 4-week 202501 but must carry 202502's 5."""
    spine = enumerate_origins(calendar_df)
    month_end = spine[spine["forecast_origin_date"] == WEEKS[3]]
    assert month_end["weeks_in_month"].item() == 5


def test_enumerate_origins_excludes_the_first_calendar_month(
    calendar_df: pd.DataFrame,
) -> None:
    """202501 has no prior month, so no origin may target it."""
    assert 202501 not in set(enumerate_origins(calendar_df)["target_month"])


def test_enumerate_origins_drops_the_final_week_with_no_next_month(
    calendar_df: pd.DataFrame,
) -> None:
    """The last week's shifted target month is NaN and cannot be an origin."""
    assert WEEKS[8] not in set(enumerate_origins(calendar_df)["forecast_origin_date"])


def test_enumerate_origins_is_unique_on_target_month_and_weeks_actualized(
    calendar_df: pd.DataFrame,
) -> None:
    """attach_workday_progress validates one_to_one, which relies on this."""
    spine = enumerate_origins(calendar_df)
    assert not spine.duplicated(["target_month", "weeks_actualized"]).any()


def test_enumerate_origins_does_not_mutate_the_calendar(
    calendar_df: pd.DataFrame,
) -> None:
    """It writes target_month and weeks_actualized, but only to its own copy."""
    before = calendar_df.copy()
    enumerate_origins(calendar_df)
    pd.testing.assert_frame_equal(calendar_df, before)


@pytest.mark.parametrize(
    "missing",
    [
        "ds",
        "fiscal_year_month",
        "fiscal_week_of_month",
        "weeks_in_month",
        "origin_month_fraction_elapsed",
    ],
)
def test_enumerate_origins_raises_when_the_calendar_lacks_a_required_column(
    calendar_df: pd.DataFrame, missing: str
) -> None:
    """Each of the five is read by the body; none is optional."""
    with pytest.raises(ValueError, match=r"calendar_df is missing required columns"):
        enumerate_origins(calendar_df.drop(columns=[missing]))


# ================================================
# attach_workday_progress
#
# Expected for 202502 (21 workdays total), by weeks_actualized:
#   0 -> elapsed 0,  remaining 21      3 -> elapsed 14, remaining 7
#   1 -> elapsed 5,  remaining 16      4 -> elapsed 19, remaining 2
#   2 -> elapsed 10, remaining 11
# ================================================


@pytest.fixture
def origin_spine(calendar_df: pd.DataFrame) -> pd.DataFrame:
    return enumerate_origins(calendar_df)


def test_attach_workday_progress_manufactures_zero_at_the_month_end_origin(
    origin_spine: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """weeks_actualized=0 has no calendar week, so its 0 must be supplied, not NaN."""
    progressed = attach_workday_progress(origin_spine, calendar_df)
    zero = progressed[progressed["weeks_actualized"] == 0]

    assert len(zero) == 1
    assert zero["workdays_elapsed"].notna().all()
    assert zero["workdays_elapsed"].item() == 0
    assert zero["workdays_remaining"].item() == zero["number_workdays"].item()


def test_attach_workday_progress_computes_the_expected_columns(
    origin_spine: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """Non-uniform workday counts, so an off-by-one in the cumsum is visible."""
    progressed = attach_workday_progress(origin_spine, calendar_df)

    assert progressed["workdays_elapsed"].tolist() == [0, 5, 10, 14, 19]
    assert progressed["workdays_remaining"].tolist() == [21, 16, 11, 7, 2]


def test_attach_workday_progress_uses_the_target_months_workday_total(
    origin_spine: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """The rolled-forward origin sits in 202501 (17) but must carry 202502's 21."""
    progressed = attach_workday_progress(origin_spine, calendar_df)
    month_end = progressed[progressed["forecast_origin_date"] == WEEKS[3]]

    assert month_end["number_workdays"].item() == 21


def test_attach_workday_progress_leaves_no_nulls(
    origin_spine: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """The week-zero row makes the join total, so no column may come back null."""
    progressed = attach_workday_progress(origin_spine, calendar_df)
    added = ["number_workdays", "workdays_elapsed", "workdays_remaining"]
    assert not progressed[added].isna().any().any()


def test_attach_workday_progress_preserves_spine_rows_and_order(
    origin_spine: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """A lookup must not reorder or drop the frame it is attached to."""
    progressed = attach_workday_progress(origin_spine, calendar_df)
    pd.testing.assert_frame_equal(
        progressed[origin_spine.columns.tolist()], origin_spine
    )


def test_attach_workday_progress_does_not_mutate_the_spine(
    origin_spine: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """The caller's frame must not gain columns as a side effect."""
    before = origin_spine.copy()
    attach_workday_progress(origin_spine, calendar_df)
    pd.testing.assert_frame_equal(origin_spine, before)


def test_attach_workday_progress_raises_on_a_duplicated_calendar_week(
    origin_spine: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """A duplicate lookup key would silently fan origins out; one_to_one blocks it."""
    duplicated = pd.concat([calendar_df, calendar_df.iloc[[5]]], ignore_index=True)
    with pytest.raises(pd.errors.MergeError):
        attach_workday_progress(origin_spine, duplicated)


def test_attach_workday_progress_raises_on_a_duplicated_spine_origin(
    origin_spine: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """one_to_one guards the left side too, catching a malformed spine."""
    duplicated = pd.concat([origin_spine, origin_spine.iloc[[2]]], ignore_index=True)
    with pytest.raises(pd.errors.MergeError):
        attach_workday_progress(duplicated, calendar_df)


def test_attach_workday_progress_raises_when_a_target_month_is_absent(
    origin_spine: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """A month missing from the calendar must halt, not silently become zero."""
    without_target = calendar_df[calendar_df["fiscal_year_month"] != 202502]
    with pytest.raises(ValueError, match=r"NaN"):
        attach_workday_progress(origin_spine, without_target)


@pytest.mark.parametrize(
    "missing", ["fiscal_year_month", "fiscal_week_of_month", "count_workdays"]
)
def test_attach_workday_progress_raises_when_the_calendar_lacks_a_column(
    origin_spine: pd.DataFrame, calendar_df: pd.DataFrame, missing: str
) -> None:
    """The workday lookup needs all three; none is optional."""
    with pytest.raises(ValueError, match=r"calendar_df is missing required columns"):
        attach_workday_progress(origin_spine, calendar_df.drop(columns=[missing]))


@pytest.mark.parametrize("missing", ["target_month", "weeks_actualized"])
def test_attach_workday_progress_raises_when_the_spine_lacks_a_column(
    origin_spine: pd.DataFrame, calendar_df: pd.DataFrame, missing: str
) -> None:
    """Both are merge keys; the other spine columns only pass through."""
    with pytest.raises(ValueError, match=r"origin_spine is missing required columns"):
        attach_workday_progress(origin_spine.drop(columns=[missing]), calendar_df)


def test_attach_workday_progress_is_insensitive_to_calendar_row_order(
    origin_spine: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """cumsum is order-dependent, so the function must sort before accumulating."""
    shuffled = calendar_df.sample(frac=1, random_state=0)
    pd.testing.assert_frame_equal(
        attach_workday_progress(origin_spine, calendar_df),
        attach_workday_progress(origin_spine, shuffled),
    )
