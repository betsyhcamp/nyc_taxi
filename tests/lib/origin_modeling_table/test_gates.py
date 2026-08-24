from __future__ import annotations

import pandas as pd
import pytest

from fcstnyctaxi.lib.origin_modeling_table.column_roles import ModelingTableSchema
from fcstnyctaxi.lib.origin_modeling_table.gates import (
    assert_all_horizon_1,
    assert_benchmark_key_parity,
    assert_fold_is_populated,
    assert_forecast_reconstruction,
    assert_join_integrity,
    assert_lag_alignment,
    assert_month_total_reconciliation,
    assert_mtd_construction,
    assert_no_future_leakage,
    assert_preprocess_feature_drift,
    assert_remaining_target_identity,
    assert_shared_cutoff,
    assert_tier_categorical,
    expected_sidecar_counts,
    weekly_actuals_by_fiscal_week,
)

# ================================================
# Fixtures
#
# Two series over two 4-week fiscal months, small enough that every expected
# value below is checkable by hand.
#
#   weeks   202501: 01-05 01-12 01-19 01-26     202502: 02-02 02-09 02-16 02-23
#   id=10   y =        10    20    30    40                50    60    70    80
#   id=20   y =         1     2     3     4                 5     6     7     8
#
# enumerate_origins drops origins targeting the first month, so every origin
# here targets 202502 with weeks_actualized 0..3:
#
#   wa=0 -> origin 01-26   mtd(10) =   0   mtd(20) = 0
#   wa=1 -> origin 02-02   mtd(10) =  50   mtd(20) = 5
#   wa=2 -> origin 02-09   mtd(10) = 110   mtd(20) = 11
#   wa=3 -> origin 02-16   mtd(10) = 180   mtd(20) = 18
#
# month totals for 202502: id=10 -> 260, id=20 -> 26. The last origin is wa=3,
# where mtd + the final week (80 / 8) reconstructs the total exactly.
# ================================================

JAN = pd.to_datetime(["2025-01-05", "2025-01-12", "2025-01-19", "2025-01-26"])
FEB = pd.to_datetime(["2025-02-02", "2025-02-09", "2025-02-16", "2025-02-23"])
WEEKS = JAN.append(FEB)
ORIGINS = pd.to_datetime(["2025-01-26", "2025-02-02", "2025-02-09", "2025-02-16"])


@pytest.fixture
def calendar_df() -> pd.DataFrame:
    """Two 4-week fiscal months, one row per week."""
    return pd.DataFrame(
        {
            "ds": WEEKS,
            "fiscal_year_month": [202501] * 4 + [202502] * 4,
            "fiscal_week_of_month": [1, 2, 3, 4] * 2,
            "weeks_in_month": [4] * 8,
        }
    )


@pytest.fixture
def panel() -> pd.DataFrame:
    """id=10 at 10..80, id=20 at 1..8 — distinct scales so a swap is visible."""
    return pd.concat(
        [
            pd.DataFrame({"unique_id": 10, "ds": WEEKS, "y": range(10, 90, 10)}),
            pd.DataFrame({"unique_id": 20, "ds": WEEKS, "y": range(1, 9)}),
        ],
        ignore_index=True,
    )


@pytest.fixture
def actual_monthly_df() -> pd.DataFrame:
    """Realized month totals: 100/260 for id=10, 10/26 for id=20."""
    return pd.DataFrame(
        {
            "unique_id": [10, 10, 20, 20],
            "fiscal_year_month": [202501, 202502, 202501, 202502],
            "actual_monthly_total": [100, 260, 10, 26],
        }
    )


@pytest.fixture
def weekly_actuals(panel: pd.DataFrame, calendar_df: pd.DataFrame) -> pd.DataFrame:
    return weekly_actuals_by_fiscal_week(panel, calendar_df)


@pytest.fixture
def origin_spine() -> pd.DataFrame:
    """Four origins, all targeting 202502."""
    return pd.DataFrame(
        {
            "target_month": [202502] * 4,
            "forecast_origin_date": ORIGINS,
            "weeks_actualized": [0, 1, 2, 3],
            "weeks_in_month": [4] * 4,
        }
    )


@pytest.fixture
def origin_target_table(origin_spine: pd.DataFrame) -> pd.DataFrame:
    """The spine crossed with both series, carrying hand-computed MTD."""
    rows = []
    for uid, scale in ((10, 10), (20, 1)):
        mtd = [0, 5 * scale, 11 * scale, 18 * scale]
        rows.append(
            origin_spine.assign(
                unique_id=uid,
                mtd_revenue=mtd,
                target_month_total_revenue=26 * scale,
            )
        )
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def weekly_features(panel: pd.DataFrame) -> pd.DataFrame:
    """MLForecast's shape: ds renamed feature_row_ds, lag1 = the prior week's y."""
    out = panel.rename(columns={"ds": "feature_row_ds"}).sort_values(
        ["unique_id", "feature_row_ds"]
    )
    out["lag1"] = out.groupby("unique_id")["y"].shift(1)
    return out.reset_index(drop=True)


@pytest.fixture
def schema() -> ModelingTableSchema:
    return ModelingTableSchema(
        key_cols=("unique_id", "forecast_origin_date", "target_month"),
        feature_cols=("mtd_revenue", "lag1"),
        target_col="target_month_total_revenue",
        passthrough_cols=("feature_row_ds",),
        progress_cols=("weeks_actualized", "weeks_in_month"),
    )


@pytest.fixture
def modeling_table(
    origin_target_table: pd.DataFrame, weekly_features: pd.DataFrame
) -> pd.DataFrame:
    """The joined frame: features keyed one week after the origin."""
    keyed = origin_target_table.copy()
    keyed["feature_row_ds"] = keyed["forecast_origin_date"] + pd.Timedelta(weeks=1)
    return keyed.merge(
        weekly_features[["unique_id", "feature_row_ds", "lag1"]],
        on=["unique_id", "feature_row_ds"],
        how="left",
    )


@pytest.fixture
def monthly_series() -> pd.DataFrame:
    """A minimal sidecar frame: two origins x two series, all horizon_1."""
    return pd.DataFrame(
        {
            "forecast_origin_date": list(ORIGINS[:2]) * 2,
            "predicted_fiscal_year_month": [202502] * 4,
            "unique_id": [10, 10, 20, 20],
            "tier": pd.Categorical(
                ["high", "high", "low", "low"],
                categories=["very_low", "low", "middle", "high", "very_high"],
            ),
            "monthly_forecast": [250.0, 255.0, 25.0, 26.0],
            "actual_monthly_total": [260.0, 260.0, 26.0, 26.0],
            "series_weight": [0.9, 0.9, 0.1, 0.1],
            "origin_month_fraction_elapsed": [1.0, 0.25, 1.0, 0.25],
        }
    )


# ================================================
# assert_preprocess_feature_drift
# ================================================


def test_drift_passes_when_emission_matches_the_declaration(
    weekly_features: pd.DataFrame,
) -> None:
    """The clean case: one declared name, one lag, one emitted column."""
    assert (
        assert_preprocess_feature_drift(
            weekly_features, ["lag1"], lags=[1], lag_transforms={}
        )
        is None
    )


def test_drift_raises_when_a_declared_name_was_never_emitted(
    weekly_features: pd.DataFrame,
) -> None:
    """A mistyped native transform name is the defect this exists to catch."""
    with pytest.raises(AssertionError, match=r"feature drift"):
        assert_preprocess_feature_drift(
            weekly_features,
            ["lag1", "rolling_mean_lag1"],
            lags=[1, 2],
            lag_transforms={},
        )


def test_drift_raises_when_the_declaration_disagrees_with_the_lags(
    weekly_features: pd.DataFrame,
) -> None:
    """The version-drift canary: names match emission but not lags/transforms."""
    with pytest.raises(AssertionError, match=r"declaration drift"):
        assert_preprocess_feature_drift(
            weekly_features, ["lag1"], lags=[1, 2], lag_transforms={}
        )


def test_drift_accepts_a_framing_with_no_mlforecast_features() -> None:
    """Consistent emptying is a legitimate configuration, not a hole (§6.2.3)."""
    bare = pd.DataFrame({"unique_id": [10], "feature_row_ds": WEEKS[:1], "y": [10]})
    assert assert_preprocess_feature_drift(bare, [], lags=[], lag_transforms={}) is None


# ================================================
# assert_lag_alignment
# ================================================


def test_lag_alignment_passes_on_a_correct_frame(
    weekly_features: pd.DataFrame,
) -> None:
    """lag1 is the prior week's y within each series."""
    assert assert_lag_alignment(weekly_features, [1]) is None


def test_lag_alignment_raises_when_the_lag_is_shifted_wrong(
    weekly_features: pd.DataFrame,
) -> None:
    """The defect the gate exists for: lag1 no longer equals y shifted one week.

    Perturbed rather than replaced, so the NaN pattern is unchanged and the coverage
    guard stays satisfied — otherwise that guard fires first and this never runs.
    """
    shifted = weekly_features.assign(lag1=weekly_features["lag1"] + 1)
    with pytest.raises(AssertionError, match=r"lag1 does not equal"):
        assert_lag_alignment(shifted, [1])


def test_lag_alignment_vacuity_guard_raises_on_an_all_nan_lag(
    weekly_features: pd.DataFrame,
) -> None:
    """An all-NaN lag makes the masked comparison pass over an empty selection."""
    with pytest.raises(AssertionError, match=r"checked 0 rows"):
        assert_lag_alignment(weekly_features.assign(lag1=float("nan")), [1])


def test_lag_alignment_returns_early_when_no_lags_are_declared(
    weekly_features: pd.DataFrame,
) -> None:
    """No declared lags means no lag convention to verify, not a vacuous pass."""
    assert assert_lag_alignment(weekly_features.drop(columns="lag1"), []) is None


# ================================================
# assert_mtd_construction
# ================================================


def test_mtd_construction_passes_on_a_correct_table(
    origin_target_table: pd.DataFrame,
    weekly_actuals: pd.DataFrame,
    actual_monthly_df: pd.DataFrame,
) -> None:
    """MTD starts at zero and grows by exactly that week's raw actual."""
    assert (
        assert_mtd_construction(origin_target_table, weekly_actuals, actual_monthly_df)
        is None
    )


def test_mtd_construction_raises_on_a_nonzero_mtd_at_the_zero_origin(
    origin_target_table: pd.DataFrame,
    weekly_actuals: pd.DataFrame,
    actual_monthly_df: pd.DataFrame,
) -> None:
    """weeks_actualized == 0 means nothing of the month is observed yet."""
    bad = origin_target_table.copy()
    bad.loc[bad["weeks_actualized"] == 0, "mtd_revenue"] = 1.0
    with pytest.raises(AssertionError, match=r"nonzero mtd_revenue"):
        assert_mtd_construction(bad, weekly_actuals, actual_monthly_df)


def test_mtd_construction_raises_when_an_increment_is_not_that_weeks_actual(
    origin_target_table: pd.DataFrame,
    weekly_actuals: pd.DataFrame,
    actual_monthly_df: pd.DataFrame,
) -> None:
    """The identity itself: each step of MTD is one week of raw y."""
    bad = origin_target_table.astype({"mtd_revenue": float})
    bad.loc[bad["weeks_actualized"] == 2, "mtd_revenue"] *= 1.5
    with pytest.raises(AssertionError, match=r"MTD increment"):
        assert_mtd_construction(bad, weekly_actuals, actual_monthly_df)


def test_mtd_construction_coverage_guard_raises_on_a_missing_active_pair(
    origin_target_table: pd.DataFrame,
    weekly_actuals: pd.DataFrame,
    actual_monthly_df: pd.DataFrame,
) -> None:
    """The row universe is active pairs; a dropped series is not a smaller check."""
    short = origin_target_table[origin_target_table["unique_id"] != 20]
    with pytest.raises(AssertionError, match=r"MTD=0 coverage mismatch"):
        assert_mtd_construction(short, weekly_actuals, actual_monthly_df)


# ================================================
# assert_month_total_reconciliation
# ================================================


def test_month_total_passes_on_a_correct_table(
    origin_target_table: pd.DataFrame,
    weekly_actuals: pd.DataFrame,
    actual_monthly_df: pd.DataFrame,
) -> None:
    """MTD at the last origin plus the final week reconstructs the month total."""
    assert (
        assert_month_total_reconciliation(
            origin_target_table, weekly_actuals, actual_monthly_df
        )
        is None
    )


def test_month_total_raises_when_the_total_does_not_reconcile(
    origin_target_table: pd.DataFrame,
    weekly_actuals: pd.DataFrame,
    actual_monthly_df: pd.DataFrame,
) -> None:
    """Reads the total from actual_monthly_df, so corrupting it must be caught."""
    bad = actual_monthly_df.assign(
        actual_monthly_total=actual_monthly_df["actual_monthly_total"] + 1
    )
    with pytest.raises(AssertionError, match=r"month total"):
        assert_month_total_reconciliation(origin_target_table, weekly_actuals, bad)


def test_month_total_coverage_guard_raises_when_a_pair_has_no_last_origin(
    origin_target_table: pd.DataFrame,
    weekly_actuals: pd.DataFrame,
    actual_monthly_df: pd.DataFrame,
) -> None:
    """Exactly one last-origin row per active pair; fewer is a vacuous check."""
    short = origin_target_table[
        ~(
            (origin_target_table["unique_id"] == 20)
            & (origin_target_table["weeks_actualized"] == 3)
        )
    ]
    with pytest.raises(AssertionError, match=r"last-origin coverage mismatch"):
        assert_month_total_reconciliation(short, weekly_actuals, actual_monthly_df)


# ================================================
# assert_remaining_target_identity  (§6.4 — no caller in this PR)
# ================================================


def _with_remaining(origin_target_table: pd.DataFrame) -> pd.DataFrame:
    """The framing's target: month total minus what is already observed."""
    return origin_target_table.assign(
        remaining_month_revenue=origin_target_table["target_month_total_revenue"]
        - origin_target_table["mtd_revenue"]
    )


def test_remaining_identity_passes_when_derived_from_raw_weeks(
    origin_target_table: pd.DataFrame, weekly_actuals: pd.DataFrame
) -> None:
    """The subtraction agrees with the sum of the post-origin weeks."""
    assert (
        assert_remaining_target_identity(
            _with_remaining(origin_target_table), weekly_actuals
        )
        is None
    )


def test_remaining_identity_raises_on_a_wrong_target(
    origin_target_table: pd.DataFrame, weekly_actuals: pd.DataFrame
) -> None:
    """The arithmetic check, independent of where the target came from."""
    bad = _with_remaining(origin_target_table)
    bad["remaining_month_revenue"] *= 1.01
    with pytest.raises(AssertionError, match=r"remaining_month_revenue"):
        assert_remaining_target_identity(bad, weekly_actuals)


def test_remaining_identity_raises_when_the_weekly_actuals_are_missing_a_pair(
    origin_target_table: pd.DataFrame, weekly_actuals: pd.DataFrame
) -> None:
    """The self-comparison hole §6.4 names: no derivation must not read as zero.

    Deleting a pair's weekly rows leaves the derivation with nothing to sum. An
    earlier version defaulted that to 0.0, so any origin whose remaining happened
    to be zero passed while nothing had actually been compared.
    """
    holed = weekly_actuals[weekly_actuals["unique_id"] != 20]
    bad = _with_remaining(origin_target_table)
    bad.loc[bad["unique_id"] == 20, "remaining_month_revenue"] = 0.0
    with pytest.raises(AssertionError, match=r"lack the post-origin weeks"):
        assert_remaining_target_identity(bad, holed)


# ================================================
# assert_join_integrity
# ================================================


def test_join_integrity_passes_on_a_complete_join(
    modeling_table: pd.DataFrame,
    origin_target_table: pd.DataFrame,
    weekly_features: pd.DataFrame,
) -> None:
    """Every origin found a feature row and none were duplicated."""
    assert (
        assert_join_integrity(modeling_table, origin_target_table, weekly_features)
        is None
    )


def test_join_integrity_raises_when_origins_were_dropped(
    modeling_table: pd.DataFrame,
    origin_target_table: pd.DataFrame,
    weekly_features: pd.DataFrame,
) -> None:
    """Cardinality: the join must not lose rows the origin table carried."""
    with pytest.raises(AssertionError, match=r"changed the row count"):
        assert_join_integrity(
            modeling_table.head(3), origin_target_table, weekly_features
        )


def test_join_integrity_raises_on_a_duplicated_feature_key(
    modeling_table: pd.DataFrame,
    origin_target_table: pd.DataFrame,
    weekly_features: pd.DataFrame,
) -> None:
    """A repeated feature row would fan origins out on the join."""
    duplicated = pd.concat(
        [weekly_features, weekly_features.iloc[[4]]], ignore_index=True
    )
    with pytest.raises(AssertionError, match=r"repeats"):
        assert_join_integrity(modeling_table, origin_target_table, duplicated)


def test_join_integrity_raises_when_an_origin_has_no_feature_row(
    modeling_table: pd.DataFrame,
    origin_target_table: pd.DataFrame,
    weekly_features: pd.DataFrame,
) -> None:
    """The anti-join: a left-only origin means the key convention is wrong."""
    short = weekly_features[weekly_features["feature_row_ds"] != FEB[1]]
    with pytest.raises(AssertionError, match=r"found no feature row"):
        assert_join_integrity(modeling_table, origin_target_table, short)


def test_join_integrity_vacuity_guard_raises_on_an_empty_origin_table(
    modeling_table: pd.DataFrame, weekly_features: pd.DataFrame
) -> None:
    """Zero origins would satisfy every check below it."""
    empty = modeling_table.head(0)
    with pytest.raises(AssertionError, match=r"checked no rows"):
        assert_join_integrity(empty, empty, weekly_features)


# ================================================
# assert_shared_cutoff
# ================================================


def test_shared_cutoff_passes_on_a_correct_table(
    modeling_table: pd.DataFrame, panel: pd.DataFrame
) -> None:
    """MTD increments and lag1 both reflect observed-through-W."""
    assert assert_shared_cutoff(modeling_table, panel, [1]) is None


def test_shared_cutoff_raises_when_mtd_and_the_lag_disagree(
    modeling_table: pd.DataFrame, panel: pd.DataFrame
) -> None:
    """The cross-check: a stale lag against a current MTD is a cutoff mismatch."""
    bad = modeling_table.assign(lag1=modeling_table["lag1"] + 1)
    with pytest.raises(AssertionError, match=r"MTD increment"):
        assert_shared_cutoff(bad, panel, [1])


def test_shared_cutoff_raises_when_the_join_offset_is_wrong(
    origin_target_table: pd.DataFrame,
    weekly_features: pd.DataFrame,
    panel: pd.DataFrame,
) -> None:
    """The only check on feature_row_ds = W + 1.

    Keying the join at W instead leaves every lag a week stale. assert_lag_alignment
    would still pass — the features frame is untouched — so only a comparison
    against the panel by origin date catches it.
    """
    at_origin = origin_target_table.merge(
        weekly_features.rename(columns={"feature_row_ds": "forecast_origin_date"})[
            ["unique_id", "forecast_origin_date", "lag1"]
        ],
        on=["unique_id", "forecast_origin_date"],
        how="left",
    ).assign(feature_row_ds=pd.NaT)
    with pytest.raises(AssertionError):
        assert_shared_cutoff(at_origin, panel, [1])


def test_shared_cutoff_vacuity_guard_raises_when_no_origin_is_in_the_panel(
    modeling_table: pd.DataFrame, panel: pd.DataFrame
) -> None:
    """A merge-key dtype mismatch produces exactly this: an all-NaN probe."""
    orphaned = modeling_table.assign(forecast_origin_date=pd.Timestamp("2099-01-03"))
    with pytest.raises(AssertionError, match=r"selected no rows"):
        assert_shared_cutoff(orphaned, panel, [1])


def test_shared_cutoff_coverage_guard_raises_when_a_lag_is_partly_nan(
    modeling_table: pd.DataFrame, panel: pd.DataFrame
) -> None:
    """Skipping rows whose probe date exists is not a smaller check, it is a hole.

    NaN'd at weeks_actualized == 0, which the MTD half excludes (no increment
    exists there), so the probe coverage guard is the assertion under test.
    """
    holed = modeling_table.copy()
    holed.loc[holed["weeks_actualized"] == 0, "lag1"] = float("nan")
    with pytest.raises(AssertionError, match=r"were skipped"):
        assert_shared_cutoff(holed, panel, [1])


def test_shared_cutoff_returns_early_when_no_lags_are_declared(
    modeling_table: pd.DataFrame, panel: pd.DataFrame
) -> None:
    """No declared lags means nothing observable to verify the offset against."""
    assert assert_shared_cutoff(modeling_table.drop(columns="lag1"), panel, []) is None


# ================================================
# assert_no_future_leakage
# ================================================


def _clean_rebuild(origin_target_table: pd.DataFrame, schema: ModelingTableSchema):
    """A leak-free rebuild: features depend only on weeks at or before the origin.

    Both features the caller declares y-derived are actually derived from the panel
    here. An earlier version copied mtd_revenue straight from the fixture, which made
    it inert under a scramble — the positive control then passed on lag1 alone while
    a feature declared y-derived could not have moved at all.
    """

    def rebuild(p: pd.DataFrame) -> pd.DataFrame:
        observed = p.merge(
            origin_target_table[["unique_id", "forecast_origin_date"]],
            left_on=["unique_id", "ds"],
            right_on=["unique_id", "forecast_origin_date"],
        )[["unique_id", "forecast_origin_date", "y"]]
        out = origin_target_table.merge(
            observed, on=["unique_id", "forecast_origin_date"], how="left"
        )
        out["lag1"] = out["y"].fillna(0.0)

        # month-to-date: the target month's weeks at or before the origin, and only
        # those — which is what makes it leak-free as well as panel-derived
        month = p[p["ds"].isin(FEB)][["unique_id", "ds", "y"]]
        spanned = out[["unique_id", "forecast_origin_date"]].merge(
            month, on="unique_id"
        )
        spanned = spanned[spanned["ds"] <= spanned["forecast_origin_date"]]
        mtd = (
            spanned.groupby(["unique_id", "forecast_origin_date"], as_index=False)["y"]
            .sum()
            .rename(columns={"y": "mtd_revenue"})
        )
        out = out.drop(columns="mtd_revenue").merge(
            mtd, on=["unique_id", "forecast_origin_date"], how="left"
        )
        out["mtd_revenue"] = out["mtd_revenue"].fillna(0.0)

        out["feature_row_ds"] = out["forecast_origin_date"]
        return schema.select(out)

    return rebuild


LEAK_KWARGS = {
    "y_derived_features": ["mtd_revenue", "lag1"],
    "calendar_derived_features": [],
}


def test_leakage_passes_when_no_feature_depends_on_the_future(
    panel: pd.DataFrame, origin_target_table: pd.DataFrame, schema: ModelingTableSchema
) -> None:
    """Origins at or before the cutoff are invariant to scrambled future weeks."""
    assert (
        assert_no_future_leakage(
            panel,
            _clean_rebuild(origin_target_table, schema),
            schema,
            FEB[1],
            **LEAK_KWARGS,
        )
        is None
    )


def test_leakage_raises_on_a_deliberately_leaky_rebuild(
    panel: pd.DataFrame, origin_target_table: pd.DataFrame, schema: ModelingTableSchema
) -> None:
    """§8.4's named case: a post-cutoff value reaching a pre-cutoff feature."""

    def leaky(p: pd.DataFrame) -> pd.DataFrame:
        out = origin_target_table.copy()
        # every origin sees the LAST week of the panel, cutoff or not
        last = p.sort_values("ds").groupby("unique_id")["y"].last()
        out["lag1"] = out["unique_id"].map(last).astype(float)
        out["feature_row_ds"] = out["forecast_origin_date"]
        return schema.select(out)

    with pytest.raises(AssertionError, match=r"LEAK"):
        assert_no_future_leakage(panel, leaky, schema, FEB[1], **LEAK_KWARGS)


def test_leakage_raises_when_the_partition_misses_a_feature(
    panel: pd.DataFrame, origin_target_table: pd.DataFrame, schema: ModelingTableSchema
) -> None:
    """A partition, not a subset: adding a feature without classifying it fails."""
    with pytest.raises(AssertionError, match=r"does not cover"):
        assert_no_future_leakage(
            panel,
            _clean_rebuild(origin_target_table, schema),
            schema,
            FEB[1],
            y_derived_features=["mtd_revenue"],
            calendar_derived_features=[],
        )


def test_leakage_vacuity_guard_raises_when_nothing_is_perturbed(
    panel: pd.DataFrame, origin_target_table: pd.DataFrame, schema: ModelingTableSchema
) -> None:
    """A cutoff after every panel row scrambles nothing, so nothing is proven."""
    with pytest.raises(AssertionError, match=r"nothing perturbed"):
        assert_no_future_leakage(
            panel,
            _clean_rebuild(origin_target_table, schema),
            schema,
            pd.Timestamp("2099-01-03"),
            **LEAK_KWARGS,
        )


def test_leakage_positive_control_raises_when_the_rebuild_ignores_the_panel(
    panel: pd.DataFrame, origin_target_table: pd.DataFrame, schema: ModelingTableSchema
) -> None:
    """Nothing derived: a rebuild that ignores its argument compares a frame to itself.

    The main comparison passes trivially, so only the control — which requires a
    y-derived feature to MOVE after the cutoff — can tell that the gate is inert.
    """

    def constant(p: pd.DataFrame) -> pd.DataFrame:
        out = origin_target_table.copy()
        out["lag1"] = 1.0
        out["feature_row_ds"] = out["forecast_origin_date"]
        return schema.select(out)

    with pytest.raises(AssertionError, match=r"positive control failed"):
        assert_no_future_leakage(panel, constant, schema, FEB[1], **LEAK_KWARGS)


# ================================================
# assert_fold_is_populated
# ================================================


def test_fold_is_populated_passes_on_a_usable_fold(
    modeling_table: pd.DataFrame,
) -> None:
    """Both sides have rows and the validation month is present."""
    table = pd.concat(
        [modeling_table, modeling_table.assign(target_month=202501)], ignore_index=True
    )
    assert assert_fold_is_populated(table, 202502) is None


def test_fold_is_populated_raises_when_the_val_month_is_absent(
    modeling_table: pd.DataFrame,
) -> None:
    """The defect the old disjointness check waved through: val selects nothing."""
    with pytest.raises(AssertionError, match=r"absent from the modeling table"):
        assert_fold_is_populated(modeling_table, 209901)


def test_fold_is_populated_raises_when_there_is_nothing_to_train_on(
    modeling_table: pd.DataFrame,
) -> None:
    """The earliest month has no prior months, so it cannot be a fold."""
    with pytest.raises(AssertionError, match=r"no training rows"):
        assert_fold_is_populated(modeling_table, 202502)


# ================================================
# assert_all_horizon_1
# ================================================


def test_all_horizon_1_passes_on_a_current_month_framing(
    monthly_series: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """Every row predicts the month its origin sits in, or just ended."""
    assert assert_all_horizon_1(monthly_series, calendar_df) is None


def test_all_horizon_1_raises_on_a_horizon_2_row(
    monthly_series: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """A month further out is a different framing, not a variation of this one."""
    bad = monthly_series.copy()
    bad.loc[bad.index[0], "predicted_fiscal_year_month"] = 202503
    with pytest.raises(AssertionError, match=r"not horizon_1"):
        assert_all_horizon_1(bad, calendar_df)


def test_all_horizon_1_vacuity_guard_raises_on_an_empty_frame(
    monthly_series: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """All-of-nothing is true; the guard is what makes the check mean something."""
    with pytest.raises(AssertionError, match=r"labeled no rows"):
        assert_all_horizon_1(monthly_series.head(0), calendar_df)


# ================================================
# assert_tier_categorical
# ================================================


def test_tier_passes_on_a_complete_ladder(monthly_series: pd.DataFrame) -> None:
    """Five categories present, whatever they are called."""
    assert assert_tier_categorical(monthly_series) is None


def test_tier_raises_when_folds_disagree_and_the_dtype_collapses(
    monthly_series: pd.DataFrame,
) -> None:
    """Categoricals with mismatched categories concat to object."""
    with pytest.raises(AssertionError, match=r"must be a categorical dtype"):
        assert_tier_categorical(monthly_series.assign(tier=["a", "b", "c", "d"]))


def test_tier_raises_when_every_fold_truncated_the_same_way(
    monthly_series: pd.DataFrame,
) -> None:
    """Stays categorical and short — the case the dtype check cannot see."""
    short = monthly_series.assign(
        tier=pd.Categorical(["low"] * 4, categories=["very_low", "low", "middle"])
    )
    with pytest.raises(AssertionError, match=r"3 categories, expected 5"):
        assert_tier_categorical(short)


def test_tier_is_agnostic_to_the_label_names(monthly_series: pd.DataFrame) -> None:
    """Nothing downstream requires particular names, so renaming must not fail."""
    renamed = monthly_series.assign(
        tier=pd.Categorical(
            ["big"] * 4, categories=["tiny", "small", "mid", "big", "huge"]
        )
    )
    assert assert_tier_categorical(renamed) is None


# ================================================
# assert_benchmark_key_parity
# ================================================


@pytest.fixture
def counts() -> dict[str, int]:
    """Two origins x two series over one month, matching the sidecar fixture."""
    return {"target_months": 1, "origins": 2, "series": 2, "events": 4}


def test_key_parity_passes_and_returns_the_counts(
    monthly_series: pd.DataFrame, counts: dict[str, int]
) -> None:
    """Returns its counts so the audit the framing printed stays printable."""
    assert (
        assert_benchmark_key_parity(
            monthly_series, monthly_series, expected_counts=counts
        )
        == counts
    )


def test_key_parity_raises_when_the_key_sets_differ(
    monthly_series: pd.DataFrame, counts: dict[str, int]
) -> None:
    """The check proper: challenger and benchmark must cover identical keys."""
    bench = monthly_series.copy()
    bench.loc[bench.index[0], "unique_id"] = 99
    with pytest.raises(AssertionError, match=r"key sets differ"):
        assert_benchmark_key_parity(monthly_series, bench, expected_counts=counts)


def test_key_parity_raises_on_a_common_mode_shortfall(
    monthly_series: pd.DataFrame, counts: dict[str, int]
) -> None:
    """What key parity structurally cannot see: both sides wrong the same way.

    Two artifacts compared against each other catch divergence only. The counts
    come from upstream of both, which is the only thing that sees this.
    """
    short = monthly_series[monthly_series["unique_id"] != 20]
    with pytest.raises(AssertionError, match=r"count mismatch"):
        assert_benchmark_key_parity(short, short, expected_counts=counts)


def test_key_parity_raises_on_duplicated_keys_before_comparing_sets(
    monthly_series: pd.DataFrame, counts: dict[str, int]
) -> None:
    """Sets collapse duplicates, so two differently-duplicated frames compare equal."""
    duplicated = pd.concat([monthly_series, monthly_series.head(1)], ignore_index=True)
    with pytest.raises(AssertionError, match=r"repeats"):
        assert_benchmark_key_parity(
            duplicated,
            monthly_series,
            expected_counts={**counts, "events": 5},
        )


def test_key_parity_raises_on_a_sidecar_missing_its_value_columns(
    monthly_series: pd.DataFrame, counts: dict[str, int]
) -> None:
    """Required, not compared-if-present: skipping is how a stripped frame passes."""
    keys = ["forecast_origin_date", "predicted_fiscal_year_month", "unique_id"]
    with pytest.raises(ValueError, match=r"missing required columns"):
        assert_benchmark_key_parity(
            monthly_series[keys], monthly_series[keys], expected_counts=counts
        )


def test_key_parity_raises_when_a_shared_value_disagrees(
    monthly_series: pd.DataFrame, counts: dict[str, int]
) -> None:
    """Both frames derive actual_monthly_total from the same panel and calendar."""
    bench = monthly_series.assign(
        actual_monthly_total=monthly_series["actual_monthly_total"] * 2
    )
    with pytest.raises(AssertionError, match=r"actual_monthly_total differs"):
        assert_benchmark_key_parity(monthly_series, bench, expected_counts=counts)


# ================================================
# assert_forecast_reconstruction  (§6.4 — no caller in this PR)
# ================================================


@pytest.fixture
def components_df(monthly_series: pd.DataFrame) -> pd.DataFrame:
    """mtd + predicted_remaining reconstructs monthly_forecast exactly."""
    mtd = [100.0, 0.0, 10.0, 0.0]
    return monthly_series[
        ["forecast_origin_date", "predicted_fiscal_year_month", "unique_id"]
    ].assign(
        mtd_revenue=mtd,
        predicted_remaining=monthly_series["monthly_forecast"].to_numpy() - mtd,
    )


def test_reconstruction_passes_on_aligned_files(
    monthly_series: pd.DataFrame, components_df: pd.DataFrame
) -> None:
    """Every key appears in both files and the components sum to the forecast."""
    assert assert_forecast_reconstruction(monthly_series, components_df) is None


def test_reconstruction_raises_when_a_key_is_in_only_one_file(
    monthly_series: pd.DataFrame, components_df: pd.DataFrame
) -> None:
    """The row-alignment half, which is what can actually fail today."""
    with pytest.raises(AssertionError, match=r"appear in only one file"):
        assert_forecast_reconstruction(monthly_series, components_df.head(3))


def test_reconstruction_raises_when_the_components_do_not_sum(
    monthly_series: pd.DataFrame, components_df: pd.DataFrame
) -> None:
    """The provenance canary: free today, live the moment the sum is recomputed."""
    bad = components_df.assign(mtd_revenue=components_df["mtd_revenue"] + 1)
    with pytest.raises(AssertionError, match=r"!= monthly_forecast"):
        assert_forecast_reconstruction(monthly_series, bad)


def test_reconstruction_vacuity_guard_raises_on_empty_files(
    monthly_series: pd.DataFrame, components_df: pd.DataFrame
) -> None:
    """An all-both assertion over zero rows would otherwise pass."""
    with pytest.raises(AssertionError, match=r"checked no rows"):
        assert_forecast_reconstruction(monthly_series.head(0), components_df.head(0))


# ================================================
# Shared prep helpers
# ================================================


def test_weekly_actuals_keys_by_fiscal_position_without_aggregating(
    weekly_actuals: pd.DataFrame, panel: pd.DataFrame
) -> None:
    """One row per panel row: the identity gates compare against per-week values."""
    assert list(weekly_actuals.columns) == [
        "unique_id",
        "target_month",
        "week_of_month",
        "weekly_actual",
    ]
    assert len(weekly_actuals) == len(panel)


def test_weekly_actuals_raises_when_the_calendar_misses_a_panel_date(
    panel: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """An unlabeled row would silently vanish from every consuming gate."""
    with pytest.raises(ValueError, match=r"does not cover"):
        weekly_actuals_by_fiscal_week(panel, calendar_df.iloc[1:])


def test_weekly_actuals_raises_on_a_duplicated_calendar_ds(
    panel: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """A repeated ds fans the panel out here and again in every consuming gate."""
    duplicated = pd.concat([calendar_df, calendar_df.iloc[[0]]], ignore_index=True)
    with pytest.raises(pd.errors.MergeError):
        weekly_actuals_by_fiscal_week(panel, duplicated)


def test_expected_counts_derive_from_pipeline_inputs_only(
    origin_spine: pd.DataFrame, panel: pd.DataFrame, actual_monthly_df: pd.DataFrame
) -> None:
    """Four origins x two active series in 202502 = 8 events, hand-checked."""
    assert expected_sidecar_counts(
        origin_spine, panel, actual_monthly_df, [202502]
    ) == {
        "target_months": 1,
        "origins": 4,
        "series": 2,
        "events": 8,
    }


def test_expected_counts_use_active_pairs_not_the_panel_for_series(
    origin_spine: pd.DataFrame, panel: pd.DataFrame, actual_monthly_df: pd.DataFrame
) -> None:
    """A series with history but no evaluated month produces no sidecar row."""
    inactive = actual_monthly_df[
        ~(
            (actual_monthly_df["unique_id"] == 20)
            & (actual_monthly_df["fiscal_year_month"] == 202502)
        )
    ]
    counts = expected_sidecar_counts(origin_spine, panel, inactive, [202502])
    assert counts["series"] == 1
    assert panel["unique_id"].nunique() == 2
