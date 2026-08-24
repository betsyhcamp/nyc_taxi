from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fcstnyctaxi.lib.fold_metrics import (
    _FOLD_KEYS,
    _JOIN_KEYS,
    _PROGRESS_SKILL_COLS,
    compute_hero_metric,
    compute_signed_bias_per_series,
    compute_signed_bias_pooled,
    compute_wape,
    compute_weighted_signed_bias,
    compute_wrmae_by_progress,
    compute_wrmae_per_series,
    compute_wrmae_pooled,
)
from fcstnyctaxi.lib.metrics import (
    _SMALL_NUM_BOUND,
    signed_bias_per_series,
    signed_bias_pooled,
    wrmae_per_series,
    wrmae_pooled,
)

# ================================================
# Shared timestamps
# ================================================

_ORIGIN_1 = pd.Timestamp("2024-01-07")
_ORIGIN_2 = pd.Timestamp("2024-02-04")
_MONTH_1 = pd.Timestamp("2024-01-01")
_MONTH_2 = pd.Timestamp("2024-02-01")

# ================================================
# Fixtures
# ================================================


@pytest.fixture
def base_dfs():
    """2 fold, 2 series fixture; all ch/bm error ratios = 0.5, overforecast by 10%.

    Challenger errors: A->10, B->20 (fold 1); A->5, B->10 (fold 2).
    Benchmark errors:  A->20, B->40 (fold 1); A->10, B->20 (fold 2).
    All error ratios = 0.5 -> WRMAE metrics = 0.5.
    Overforecast bias ≈ 10% of actual -> signed bias metrics = 0.075.
    WAPE: sum(|errors|)/sum(|actuals|) per fold -> 30/300=0.1, 15/300=0.05 -> avg=0.075.
    """
    origins = [_ORIGIN_1, _ORIGIN_1, _ORIGIN_2, _ORIGIN_2]
    months = [_MONTH_1, _MONTH_1, _MONTH_2, _MONTH_2]
    uids = ["A", "B", "A", "B"]

    challenger = pd.DataFrame(
        {
            "forecast_origin_date": origins,
            "predicted_fiscal_year_month": months,
            "unique_id": uids,
            "monthly_forecast": [110.0, 220.0, 105.0, 210.0],
            "actual_monthly_total": [100.0, 200.0, 100.0, 200.0],
            "series_weight": [1.0, 1.0, 1.0, 1.0],
            "tier": ["top", "bottom", "top", "bottom"],
            "horizon": ["horizon_1"] * 4,
        }
    )

    benchmark = pd.DataFrame(
        {
            "forecast_origin_date": origins,
            "predicted_fiscal_year_month": months,
            "unique_id": uids,
            "monthly_forecast": [120.0, 240.0, 110.0, 220.0],
            "tier": ["top", "bottom", "top", "bottom"],
            "horizon": ["horizon_1"] * 4,
        }
    )

    return challenger, benchmark


@pytest.fixture
def tier_dfs():
    """2 fold, 2 tier fixture where error ratio differs by tier.

    Series A (top):    ch_err=10, bm_err=20 per fold -> ratio=0.5.
    Series B (bottom): ch_err=40, bm_err=20 per fold -> ratio=2.0.
    All tiers pooled:  weighted_ch=50, weighted_bm=40 per fold -> ratio=1.25.
    """
    origins = [_ORIGIN_1, _ORIGIN_1, _ORIGIN_2, _ORIGIN_2]
    months = [_MONTH_1, _MONTH_1, _MONTH_2, _MONTH_2]
    uids = ["A", "B", "A", "B"]

    challenger = pd.DataFrame(
        {
            "forecast_origin_date": origins,
            "predicted_fiscal_year_month": months,
            "unique_id": uids,
            # A: actual=100, forecast=110 -> ch_err=10
            # B: actual=200, forecast=240 -> ch_err=40
            "monthly_forecast": [110.0, 240.0, 110.0, 240.0],
            "actual_monthly_total": [100.0, 200.0, 100.0, 200.0],
            "series_weight": [1.0, 1.0, 1.0, 1.0],
            "tier": ["top", "bottom", "top", "bottom"],
            "horizon": ["horizon_1"] * 4,
        }
    )

    benchmark = pd.DataFrame(
        {
            "forecast_origin_date": origins,
            "predicted_fiscal_year_month": months,
            "unique_id": uids,
            # A: actual=100, bm_forecast=120 -> bm_err=20
            # B: actual=200, bm_forecast=220 -> bm_err=20
            "monthly_forecast": [120.0, 220.0, 120.0, 220.0],
            "tier": ["top", "bottom", "top", "bottom"],
            "horizon": ["horizon_1"] * 4,
        }
    )

    return challenger, benchmark


# ================================================
# compute_wrmae_pooled
# ================================================


def test_compute_wrmae_pooled_known_value(base_dfs):
    """2 fold, 2 series: all ch/bm error ratios = 0.5 -> result = 0.5."""
    challenger, benchmark = base_dfs
    result = compute_wrmae_pooled(challenger, benchmark)
    np.testing.assert_allclose(result, 0.5, rtol=1e-7)


def test_compute_wrmae_pooled_tier_filter(tier_dfs):
    """Tier filter restricts to matching series; each tier yields a distinct ratio."""
    challenger, benchmark = tier_dfs
    # top only:    ratio=0.5; bottom only: ratio=2.0; all: ratio=1.25
    np.testing.assert_allclose(
        compute_wrmae_pooled(challenger, benchmark, tier="top"), 0.5, rtol=1e-7
    )
    np.testing.assert_allclose(
        compute_wrmae_pooled(challenger, benchmark, tier="bottom"), 2.0, rtol=1e-7
    )
    np.testing.assert_allclose(
        compute_wrmae_pooled(challenger, benchmark), 1.25, rtol=1e-7
    )


def test_compute_wrmae_pooled_empty_after_tier_filter_returns_nan(base_dfs):
    """Tier filter that matches no rows returns nan."""
    challenger, benchmark = base_dfs
    result = compute_wrmae_pooled(challenger, benchmark, tier="nonexistent")
    assert np.isnan(result)


def test_compute_wrmae_pooled_empty_merge_returns_nan(base_dfs):
    """No matching join keys between challenger and benchmark returns nan."""
    challenger, benchmark = base_dfs
    benchmark = benchmark.copy()
    benchmark["unique_id"] = benchmark["unique_id"] + "_X"
    result = compute_wrmae_pooled(challenger, benchmark)
    assert np.isnan(result)


def test_compute_wrmae_pooled_near_zero_benchmark_fold_excluded(base_dfs):
    """Fold with near zero weighted benchmark sum excluded; other folds counted."""
    challenger, benchmark = base_dfs
    # Force fold 1 benchmark to match actuals -> benchmark errors = 0 -> fold excluded
    benchmark = benchmark.copy()
    fold1 = benchmark["forecast_origin_date"] == _ORIGIN_1
    benchmark.loc[fold1 & (benchmark["unique_id"] == "A"), "monthly_forecast"] = 100.0
    benchmark.loc[fold1 & (benchmark["unique_id"] == "B"), "monthly_forecast"] = 200.0
    # Only fold 2 contributes: weighted_ch=15, weighted_bm=30 -> 0.5
    result = compute_wrmae_pooled(challenger, benchmark)
    np.testing.assert_allclose(result, 0.5, rtol=1e-7)


# ================================================
# compute_wrmae_per_series
# ================================================


def test_compute_wrmae_per_series_known_value(base_dfs):
    """2 fold, 2 series: all per-series ratios = 0.5 -> avg = 0.5."""
    challenger, benchmark = base_dfs
    result = compute_wrmae_per_series(challenger, benchmark)
    np.testing.assert_allclose(result, 0.5, rtol=1e-7)


def test_compute_wrmae_per_series_near_zero_benchmark_row_excluded():
    """Near zero benchmark error row excluded; weight discarded, not renormalized."""
    # 1 fold, 2 series: A has near-zero benchmark -> excluded; only B used.
    # Weight of A (=100) is dropped after exclusion; B w_norm = 1.0.
    # ch_err_B = |220-200| = 20; bm_err_B = |240-200| = 40 -> ratio = 0.5
    tiny = _SMALL_NUM_BOUND * 0.5
    origins = [_ORIGIN_1, _ORIGIN_1]
    months = [_MONTH_1, _MONTH_1]

    challenger = pd.DataFrame(
        {
            "forecast_origin_date": origins,
            "predicted_fiscal_year_month": months,
            "unique_id": ["A", "B"],
            "monthly_forecast": [110.0, 220.0],
            "actual_monthly_total": [100.0, 200.0],
            "series_weight": [100.0, 1.0],
            "tier": ["top", "bottom"],
            "horizon": ["horizon_1", "horizon_1"],
        }
    )

    benchmark = pd.DataFrame(
        {
            "forecast_origin_date": origins,
            "predicted_fiscal_year_month": months,
            "unique_id": ["A", "B"],
            # A: actual=100, bm_forecast=100+tiny -> bm_err=tiny (excluded)
            "monthly_forecast": [100.0 + tiny, 240.0],
            "tier": ["top", "bottom"],
            "horizon": ["horizon_1", "horizon_1"],
        }
    )

    result = compute_wrmae_per_series(challenger, benchmark)
    np.testing.assert_allclose(result, 0.5, rtol=1e-7)


# ================================================
# compute_signed_bias_pooled
# ================================================


def test_compute_signed_bias_pooled_known_value(base_dfs):
    """Overforecast: fold 1 bias=0.1, fold 2 bias=0.05 -> average=0.075."""
    # Fold 1: numerator=10+20=30, denominator=1*100+1*200=300 -> 0.1
    # Fold 2: numerator=5+10=15,  denominator=300 -> 0.05
    challenger, _ = base_dfs
    result = compute_signed_bias_pooled(challenger)
    np.testing.assert_allclose(result, 0.075, rtol=1e-7)


def test_compute_signed_bias_pooled_empty_returns_nan(base_dfs):
    """Empty DataFrame -> nan via early guard."""
    challenger, _ = base_dfs
    result = compute_signed_bias_pooled(challenger.iloc[0:0])
    assert np.isnan(result)


# ================================================
# compute_signed_bias_per_series
# ================================================


def test_compute_signed_bias_per_series_known_value(base_dfs):
    """Per series overforecast: fold 1=0.1, fold 2=0.05 -> avg=0.075."""
    # Fold 1, A: (110-100)/100=0.1; B: (220-200)/200=0.1; w_norm=[0.5,0.5] -> 0.1
    # Fold 2, A: (105-100)/100=0.05; B: (210-200)/200=0.05; w_norm=[0.5,0.5] -> 0.05
    challenger, _ = base_dfs
    result = compute_signed_bias_per_series(challenger)
    np.testing.assert_allclose(result, 0.075, rtol=1e-7)


def test_compute_signed_bias_per_series_near_zero_actual_excluded():
    """Near zero actual row excluded; high weight exclusion does not distort result."""
    # 1 fold: A has near-zero actual (excluded, weight=100); B is normal.
    # Only B survives: w_norm=1.0; bias=(210-200)/200=0.05
    tiny = _SMALL_NUM_BOUND * 0.5
    origins = [_ORIGIN_1, _ORIGIN_1]
    months = [_MONTH_1, _MONTH_1]

    challenger = pd.DataFrame(
        {
            "forecast_origin_date": origins,
            "predicted_fiscal_year_month": months,
            "unique_id": ["A", "B"],
            "monthly_forecast": [1.0, 210.0],
            "actual_monthly_total": [tiny, 200.0],
            "series_weight": [100.0, 1.0],
            "tier": ["top", "bottom"],
            "horizon": ["horizon_1", "horizon_1"],
        }
    )

    result = compute_signed_bias_per_series(challenger)
    np.testing.assert_allclose(result, 0.05, rtol=1e-7)


# ================================================
# compute_wape
# ================================================


def test_compute_wape_known_value(base_dfs):
    """Fold averaged WAPE: fold 1=0.1, fold 2=0.05 -> avg=0.075."""
    # Fold 1: sum|errors|=30, sum|actuals|=300 -> 0.1
    # Fold 2: sum|errors|=15, sum|actuals|=300 -> 0.05
    challenger, _ = base_dfs
    result = compute_wape(challenger)
    np.testing.assert_allclose(result, 0.075, rtol=1e-7)


def test_compute_wape_empty_returns_nan(base_dfs):
    """Empty DataFrame -> nan via early guard."""
    challenger, _ = base_dfs
    result = compute_wape(challenger.iloc[0:0])
    assert np.isnan(result)


# ================================================
# compute_weighted_signed_bias
# ================================================


def test_compute_weighted_signed_bias_known_value(base_dfs):
    """Overforecast: fold 1=0.1, fold 2=0.05 -> avg=0.075."""
    # Fold 1: sum(signed_errors)=30, sum(|actuals|)=300 -> 0.1
    # Fold 2: sum(signed_errors)=15, sum(|actuals|)=300 -> 0.05
    challenger, _ = base_dfs
    result = compute_weighted_signed_bias(challenger)
    np.testing.assert_allclose(result, 0.075, rtol=1e-7)


def test_compute_weighted_signed_bias_negative_for_under_forecast():
    """Underforecast returns negative signed bias."""
    # Example: forecast=90, actual=100 -> signed_error=-10, |actual|=100 -> -0.1
    challenger = pd.DataFrame(
        {
            "forecast_origin_date": [_ORIGIN_1],
            "predicted_fiscal_year_month": [_MONTH_1],
            "unique_id": ["A"],
            "monthly_forecast": [90.0],
            "actual_monthly_total": [100.0],
            "series_weight": [1.0],
            "tier": ["top"],
            "horizon": ["horizon_1"],
        }
    )
    result = compute_weighted_signed_bias(challenger)
    np.testing.assert_allclose(result, -0.1, rtol=1e-7)


# ================================================
# compute_hero_metric
# ================================================


def test_compute_hero_metric_unknown_metric_raises():
    """Unknown metric_name -> ValueError naming the invalid key."""
    with pytest.raises(ValueError, match="bad_metric"):
        compute_hero_metric(pd.DataFrame(), None, "bad_metric", "horizon_1")


def test_compute_hero_metric_relative_metric_without_benchmark_raises(base_dfs):
    """Relative metric with benchmark_df=None -> ValueError."""
    challenger, _ = base_dfs
    with pytest.raises(ValueError, match="wrmae_pooled"):
        compute_hero_metric(challenger, None, "wrmae_pooled", "horizon_1")
    with pytest.raises(ValueError, match="wrmae_per_series"):
        compute_hero_metric(challenger, None, "wrmae_per_series", "horizon_1")


def test_compute_hero_metric_wape_without_benchmark_is_valid(base_dfs):
    """WAPE does not require benchmark_df=None must not raise."""
    challenger, _ = base_dfs
    result = compute_hero_metric(challenger, None, "wape", "horizon_1")
    np.testing.assert_allclose(result, 0.075, rtol=1e-7)


def test_compute_hero_metric_horizon_filter_applied(base_dfs):
    """Only rows matching target horizon passed to metric; other horizons ignored."""
    challenger, benchmark = base_dfs

    # Append horizon_2 rows with extreme errors. If not filtered, result would change
    ch_extra = challenger.copy()
    ch_extra["horizon"] = "horizon_2"
    ch_extra["monthly_forecast"] = 9999.0

    bm_extra = benchmark.copy()
    bm_extra["horizon"] = "horizon_2"
    bm_extra["monthly_forecast"] = 1.0

    full_ch = pd.concat([challenger, ch_extra], ignore_index=True)
    full_bm = pd.concat([benchmark, bm_extra], ignore_index=True)

    # Correct filtering to horizon_1 -> same result as base fixture
    result = compute_hero_metric(full_ch, full_bm, "wrmae_pooled", "horizon_1")
    np.testing.assert_allclose(result, 0.5, rtol=1e-7)


def test_compute_hero_metric_no_matching_horizon_returns_nan(base_dfs):
    """target that matches no rows -> underlying function returns nan."""
    challenger, benchmark = base_dfs
    result = compute_hero_metric(challenger, benchmark, "wrmae_pooled", "horizon_99")
    assert np.isnan(result)


def test_compute_hero_metric_dispatches_wrmae_pooled(base_dfs):
    """Dispatch result matches direct compute_wrmae_pooled call on same data."""
    challenger, benchmark = base_dfs
    direct = compute_wrmae_pooled(challenger, benchmark)
    via_hero = compute_hero_metric(challenger, benchmark, "wrmae_pooled", "horizon_1")
    np.testing.assert_allclose(via_hero, direct, rtol=1e-7)


def test_compute_hero_metric_dispatches_wrmae_per_series(base_dfs):
    """Dispatch result matches direct compute_wrmae_per_series call on same data."""
    challenger, benchmark = base_dfs
    direct = compute_wrmae_per_series(challenger, benchmark)
    via_hero = compute_hero_metric(
        challenger, benchmark, "wrmae_per_series", "horizon_1"
    )
    np.testing.assert_allclose(via_hero, direct, rtol=1e-7)


def test_compute_hero_metric_dispatches_wape(base_dfs):
    """Dispatch matches direct compute_wape; adapter ignores benchmark_df."""
    challenger, benchmark = base_dfs
    direct = compute_wape(challenger)
    via_hero = compute_hero_metric(challenger, benchmark, "wape", "horizon_1")
    np.testing.assert_allclose(via_hero, direct, rtol=1e-7)


# ================================================
# Consistency tests: orchestration path vs. primitive path
#
# Each test runs two independent implementations on the same fixture:
#   Path 1 (orchestration): vectorized groupby in fold_metrics.py
#   Path 2 (primitive):     per-fold loop calling metrics.py scalar functions
# Agreement within rtol=1e-7 guards against formula drift between the layers.
# ================================================

_CH_COLS = _JOIN_KEYS + ["monthly_forecast", "actual_monthly_total", "series_weight"]
_BM_COLS = _JOIN_KEYS + ["monthly_forecast"]


def test_wrmae_pooled_orchestration_agrees_with_primitive(tier_dfs):
    """compute_wrmae_pooled fold average matches per fold wrmae_pooled primitive."""
    challenger, benchmark = tier_dfs

    orch = compute_wrmae_pooled(challenger, benchmark)

    merged = challenger[_CH_COLS].merge(
        benchmark[_BM_COLS], on=_JOIN_KEYS, suffixes=("_ch", "_bm"), how="inner"
    )
    fold_vals = []
    for _, grp in merged.groupby(_FOLD_KEYS):
        ch_err = (grp["monthly_forecast_ch"] - grp["actual_monthly_total"]).abs()
        bm_err = (grp["monthly_forecast_bm"] - grp["actual_monthly_total"]).abs()
        fold_vals.append(
            wrmae_pooled(
                ch_err.to_numpy(dtype=np.float64),
                bm_err.to_numpy(dtype=np.float64),
                grp["series_weight"].to_numpy(dtype=np.float64),
            )
        )
    np.testing.assert_allclose(orch, np.nanmean(fold_vals), rtol=1e-7)


def test_wrmae_per_series_orchestration_agrees_with_primitive(tier_dfs):
    """compute_wrmae_per_series fold avg matches per fold wrmae_per_series primitive."""
    challenger, benchmark = tier_dfs

    orch = compute_wrmae_per_series(challenger, benchmark)

    merged = challenger[_CH_COLS].merge(
        benchmark[_BM_COLS], on=_JOIN_KEYS, suffixes=("_ch", "_bm"), how="inner"
    )
    fold_vals = []
    for _, grp in merged.groupby(_FOLD_KEYS):
        ch_err = (grp["monthly_forecast_ch"] - grp["actual_monthly_total"]).abs()
        bm_err = (grp["monthly_forecast_bm"] - grp["actual_monthly_total"]).abs()
        fold_vals.append(
            wrmae_per_series(
                ch_err.to_numpy(dtype=np.float64),
                bm_err.to_numpy(dtype=np.float64),
                grp["series_weight"].to_numpy(dtype=np.float64),
            )
        )
    np.testing.assert_allclose(orch, np.nanmean(fold_vals), rtol=1e-7)


def test_signed_bias_pooled_orchestration_agrees_with_primitive(tier_dfs):
    """compute_signed_bias_pooled agrees with signed_bias_pooled primitive per fold."""
    challenger, _ = tier_dfs

    orch = compute_signed_bias_pooled(challenger)

    fold_vals = []
    for _, grp in challenger.groupby(_FOLD_KEYS):
        fold_vals.append(
            signed_bias_pooled(
                grp["actual_monthly_total"].to_numpy(dtype=np.float64),
                grp["monthly_forecast"].to_numpy(dtype=np.float64),
                grp["series_weight"].to_numpy(dtype=np.float64),
            )
        )
    np.testing.assert_allclose(orch, np.nanmean(fold_vals), rtol=1e-7)


def test_signed_bias_per_series_orchestration_agrees_with_primitive(tier_dfs):
    """compute_signed_bias_per_series agrees with signed_bias_per_series per fold."""
    challenger, _ = tier_dfs

    orch = compute_signed_bias_per_series(challenger)

    fold_vals = []
    for _, grp in challenger.groupby(_FOLD_KEYS):
        fold_vals.append(
            signed_bias_per_series(
                grp["actual_monthly_total"].to_numpy(dtype=np.float64),
                grp["monthly_forecast"].to_numpy(dtype=np.float64),
                grp["series_weight"].to_numpy(dtype=np.float64),
            )
        )
    np.testing.assert_allclose(orch, np.nanmean(fold_vals), rtol=1e-7)


# ================================================
# compute_wrmae_by_progress
# ================================================


@pytest.fixture
def progress_dfs():
    """2 origins x 2 series, each origin its own cohort, with DIFFERENT skill.

    Cohort (4, 1) from origin 1: challenger errors 10/20 against benchmark
    20/40 -> 30/60 = 0.5. Cohort (5, 2) from origin 2: both sides err 20/40
    -> 60/60 = 1.0. The cohorts must differ, or a bug that pools every row
    into one number is indistinguishable from correct segmentation.

    Challenger rows are ordered origin_2 first, so the cohorts arrive as
    [(5, 2), (4, 1)] and the closing sort has real work to do. Ordering them
    the other way would make the sort assertion vacuous.
    """
    challenger = pd.DataFrame(
        {
            "forecast_origin_date": [_ORIGIN_2, _ORIGIN_2, _ORIGIN_1, _ORIGIN_1],
            "predicted_fiscal_year_month": [_MONTH_2, _MONTH_2, _MONTH_1, _MONTH_1],
            "unique_id": ["A", "B", "A", "B"],
            "monthly_forecast": [120.0, 240.0, 110.0, 220.0],
            "actual_monthly_total": [100.0, 200.0, 100.0, 200.0],
            "series_weight": [1.0, 1.0, 1.0, 1.0],
        }
    )
    benchmark = challenger[_JOIN_KEYS].assign(
        monthly_forecast=[120.0, 240.0, 120.0, 240.0]
    )
    spine = pd.DataFrame(
        {
            "forecast_origin_date": [_ORIGIN_1, _ORIGIN_2],
            "target_month": [_MONTH_1, _MONTH_2],
            "weeks_in_month": [4, 5],
            "weeks_actualized": [1, 2],
        }
    )
    return challenger, benchmark, spine


def test_compute_wrmae_by_progress_returns_expected_cohort_table(progress_dfs):
    """Pins values, n_events, row sort and column schema in one comparison.

    assert_frame_equal covers all four at once, so the expected frame is an
    executable statement of the return contract rather than four assertions
    that each check one column.
    """
    challenger, benchmark, spine = progress_dfs

    out = compute_wrmae_by_progress(challenger, benchmark, spine)

    expected = pd.DataFrame(
        {
            "weeks_in_month": [4, 5],
            "weeks_actualized": [1, 2],
            "n_events": [2, 2],
            "wrmae_vs_benchmark": [0.5, 1.0],
        }
    )
    pd.testing.assert_frame_equal(out, expected)


def test_compute_wrmae_by_progress_raises_when_the_benchmark_loses_rows(progress_dfs):
    """Both sides are checked, not just the challenger.

    The natural bug is validating the challenger and copy-pasting past the
    benchmark: it drops rows silently and shifts the skill ratio rather than
    failing, so nothing else on this list would catch it.
    """
    challenger, benchmark, spine = progress_dfs
    orphan = benchmark.head(1).assign(forecast_origin_date=pd.Timestamp("2024-03-03"))
    benchmark = pd.concat([benchmark, orphan], ignore_index=True)

    with pytest.raises(ValueError, match=r"rows have no origin_spine entry"):
        compute_wrmae_by_progress(challenger, benchmark, spine)


def test_compute_wrmae_by_progress_returns_declared_columns_when_empty(progress_dfs):
    """An empty challenger yields an empty frame, not a KeyError from the sort.

    Regression guard: the columns= argument that supplies the schema reads as
    mere column ordering, so a later tidy-up could drop it and no other test
    would notice.
    """
    _, benchmark, spine = progress_dfs

    out = compute_wrmae_by_progress(pd.DataFrame(columns=_JOIN_KEYS), benchmark, spine)

    assert out.empty
    assert list(out.columns) == _PROGRESS_SKILL_COLS


def test_compute_wrmae_by_progress_takes_cohorts_from_the_spine(progress_dfs):
    """The same frames under a different spine produce different cohorts.

    This is the one property that distinguishes this implementation from the
    inline loop it replaces, which derived progress from the challenger's own
    columns. Asserting the cohorts against a single spine would be satisfied
    by either version.
    """
    challenger, benchmark, spine = progress_dfs
    relabelled = spine.assign(weeks_in_month=[5, 4], weeks_actualized=[3, 0])

    out = compute_wrmae_by_progress(challenger, benchmark, relabelled)

    cohorts = set(
        out[["weeks_in_month", "weeks_actualized"]].itertuples(index=False, name=None)
    )
    assert cohorts == {(5, 3), (4, 0)}


def test_compute_wrmae_by_progress_raises_on_a_duplicated_spine_key(progress_dfs):
    """A fanned spine is named as a spine defect before either merge runs."""
    challenger, benchmark, spine = progress_dfs
    fanned = pd.concat([spine, spine.head(1)], ignore_index=True)

    with pytest.raises(ValueError, match=r"origin_spine must be unique"):
        compute_wrmae_by_progress(challenger, benchmark, fanned)


@pytest.mark.parametrize(
    ("frame_name", "missing"),
    [
        ("challenger_df", "forecast_origin_date"),
        ("challenger_df", "predicted_fiscal_year_month"),
        ("benchmark_df", "forecast_origin_date"),
        ("benchmark_df", "predicted_fiscal_year_month"),
        ("origin_spine", "target_month"),
        ("origin_spine", "weeks_in_month"),
    ],
)
def test_compute_wrmae_by_progress_raises_when_a_frame_lacks_a_column(
    progress_dfs, frame_name, missing
):
    """Every frame is checked, and the message names which one is at fault."""
    frames = dict(
        zip(
            ("challenger_df", "benchmark_df", "origin_spine"), progress_dfs, strict=True
        )
    )
    frames[frame_name] = frames[frame_name].drop(columns=[missing])

    with pytest.raises(ValueError, match=rf"{frame_name} is missing required columns"):
        compute_wrmae_by_progress(**frames)


def test_compute_wrmae_by_progress_raises_when_a_frame_carries_progress_columns(
    progress_dfs,
):
    """A pre-merged frame must fail loudly rather than merge to _x/_y suffixes.

    The suffixed merge leaves the row count intact, so the row-loss check
    cannot see it and the cohort loop would raise KeyError somewhere unrelated.
    """
    challenger, benchmark, spine = progress_dfs
    challenger = challenger.assign(weeks_in_month=4, weeks_actualized=1)

    with pytest.raises(ValueError, match=r"name collision"):
        compute_wrmae_by_progress(challenger, benchmark, spine)
