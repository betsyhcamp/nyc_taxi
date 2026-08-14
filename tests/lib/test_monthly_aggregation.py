from __future__ import annotations

import pandas as pd
import pytest

from fcstnyctaxi.lib.monthly_aggregation import (
    attach_tier_and_weight,
    build_monthly_forecast_vs_actual,
    combine_monthly_forecast,
    compute_actual_monthly_totals,
    compute_mtd_actuals,
    compute_predicted_remaining,
)

# ================================================
# Fixtures
#
# 8 weeks, W-SUN, spanning two fiscal months: 202501 (weeks 0-3) and 202502
# (weeks 4-7). The fold origin sits at week 5 (second week of 202502), so
# weeks 4-5 are already observed (MTD actuals) and weeks 6-7 are still ahead
# (predicted remaining) — this is the MTD-actuals-vs-full-month boundary
# noted in spec_lightgbm_n_estimators_calibration_v1.md §8.4.
#
# Known values, worked by hand:
#   id=10: mtd_actuals=100+150=250, predicted_remaining=80+90=170
#          -> monthly_forecast=420; actual_monthly_total=450 (ground truth)
#   id=20: mtd_actuals=50+70=120,  predicted_remaining=30+40=70
#          -> monthly_forecast=190; actual_monthly_total=200 (ground truth)
# ================================================


@pytest.fixture
def calendar_df() -> pd.DataFrame:
    dates = pd.date_range("2025-01-05", periods=8, freq="W-SUN")
    return pd.DataFrame(
        {
            "ds": dates,
            "fiscal_year_month": [
                202501,
                202501,
                202501,
                202501,
                202502,
                202502,
                202502,
                202502,
            ],
        }
    )


@pytest.fixture
def train_df(calendar_df: pd.DataFrame) -> pd.DataFrame:
    """Observed weeks 0-5 (through the fold origin) for two series.

    Weeks 0-3 (202501) are filler, deliberately large/distinct from the
    202502 values so a test that fails to filter by target_fiscal_months
    would be caught. Weeks 4-5 (202502) are the MTD actuals.
    """
    weeks = calendar_df["ds"].iloc[:6].tolist()
    return pd.DataFrame(
        {
            "unique_id": [10] * 6 + [20] * 6,
            "ds": weeks + weeks,
            "y": [1, 1, 1, 1, 100, 150, 2, 2, 2, 2, 50, 70],
        }
    )


@pytest.fixture
def forecast_df(calendar_df: pd.DataFrame) -> pd.DataFrame:
    """Forecasted weeks 6-7 (the remainder of 202502) for two series."""
    weeks = calendar_df["ds"].iloc[6:8].tolist()
    return pd.DataFrame(
        {
            "unique_id": [10, 10, 20, 20],
            "ds": weeks + weeks,
            "ypred": [80.0, 90.0, 30.0, 40.0],
        }
    )


@pytest.fixture
def actual_monthly_df() -> pd.DataFrame:
    """Ground-truth realized monthly totals, deliberately different from
    monthly_forecast so a copy-paste bug swapping the two would be caught."""
    return pd.DataFrame(
        {
            "unique_id": [10, 20],
            "fiscal_year_month": [202502, 202502],
            "actual_monthly_total": [450, 200],
        }
    )


@pytest.fixture
def tier_df() -> pd.DataFrame:
    return pd.DataFrame({"unique_id": [10, 20], "tier": ["high", "low"]})


@pytest.fixture
def weight_df() -> pd.DataFrame:
    return pd.DataFrame({"unique_id": [10, 20], "series_weight": [5.0, 2.0]})


@pytest.fixture
def fold_origin(calendar_df: pd.DataFrame) -> pd.Timestamp:
    """The fold origin is the last observed week: index 5."""
    return calendar_df["ds"].iloc[5]


# ================================================
# compute_predicted_remaining
# ================================================


def test_compute_predicted_remaining_known_value(
    forecast_df: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """Sums ypred per (unique_id, fiscal_year_month) for the forecasted weeks."""
    result = compute_predicted_remaining(forecast_df, calendar_df)
    by_id = result.set_index("unique_id")["ypred"]
    assert by_id[10] == 170.0
    assert by_id[20] == 70.0


def test_compute_predicted_remaining_single_target_month(
    forecast_df: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """All forecasted weeks fall in 202502 -> exactly one period per series."""
    result = compute_predicted_remaining(forecast_df, calendar_df)
    assert set(result["fiscal_year_month"].unique()) == {202502}
    assert len(result) == 2


# ================================================
# compute_mtd_actuals
# ================================================


def test_compute_mtd_actuals_known_value(
    train_df: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """Sums y within target_fiscal_months only — 202501 filler weeks excluded."""
    result = compute_mtd_actuals(train_df, calendar_df, target_fiscal_months=[202502])
    by_id = result.set_index("unique_id")["mtd_actuals"]
    assert by_id[10] == 250
    assert by_id[20] == 120


def test_compute_mtd_actuals_empty_when_target_month_not_yet_observed(
    calendar_df: pd.DataFrame,
) -> None:
    """fold_0 case: target month is ahead of the origin, no weeks observed yet."""
    early_train_df = pd.DataFrame(
        {
            "unique_id": [10, 10, 10, 10],
            "ds": calendar_df["ds"].iloc[:4].tolist(),
            "y": [1, 1, 1, 1],
        }
    )
    result = compute_mtd_actuals(
        early_train_df, calendar_df, target_fiscal_months=[202502]
    )
    assert result.empty


# ================================================
# combine_monthly_forecast
# ================================================


def test_combine_monthly_forecast_known_value() -> None:
    """monthly_forecast = mtd_actuals + predicted_remaining."""
    mtd_actuals_df = pd.DataFrame(
        {
            "unique_id": [10, 20],
            "fiscal_year_month": [202502, 202502],
            "mtd_actuals": [250.0, 120.0],
        }
    )
    predicted_remaining_df = pd.DataFrame(
        {
            "unique_id": [10, 20],
            "fiscal_year_month": [202502, 202502],
            "ypred": [170.0, 70.0],
        }
    )
    result = combine_monthly_forecast(mtd_actuals_df, predicted_remaining_df)
    by_id = result.set_index("unique_id")["monthly_forecast"]
    assert by_id[10] == 420.0
    assert by_id[20] == 190.0


def test_combine_monthly_forecast_outer_join_fills_missing_side_with_zero() -> None:
    """A series present in only one input isn't dropped — missing side is 0."""
    mtd_actuals_df = pd.DataFrame(
        {"unique_id": [10], "fiscal_year_month": [202502], "mtd_actuals": [250.0]}
    )
    predicted_remaining_df = pd.DataFrame(
        {"unique_id": [20], "fiscal_year_month": [202502], "ypred": [70.0]}
    )
    result = combine_monthly_forecast(mtd_actuals_df, predicted_remaining_df)
    by_id = result.set_index("unique_id")["monthly_forecast"]
    assert by_id[10] == 250.0
    assert by_id[20] == 70.0


def test_combine_monthly_forecast_materializes_zero_mtd_when_input_empty(
    calendar_df: pd.DataFrame,
) -> None:
    """mtd_revenue is 0, not missing, when no target-month week is observed yet.

    compute_mtd_actuals returns no rows at all for this population, so the
    surfaced mtd_revenue is manufactured by the outer join's fillna(0) rather
    than read off mtd_actuals_df. Sourcing it from mtd_actuals_df directly
    would drop these rows entirely — and they are not an edge case: every
    next-month (horizon_2) row of Framing A takes this path, since no week of
    the next fiscal month is ever observed at the origin.

    The empty frame is derived through compute_mtd_actuals rather than
    hand-built so it carries the dtypes production actually emits (int64 keys),
    not the float64/object columns an empty literal would produce.
    """
    weeks = calendar_df["ds"].iloc[:4].tolist()
    pre_target_train_df = pd.DataFrame(
        {
            "unique_id": [10] * 4 + [20] * 4,
            "ds": weeks + weeks,
            "y": [1, 1, 1, 1, 2, 2, 2, 2],
        }
    )
    empty_mtd_actuals_df = compute_mtd_actuals(
        pre_target_train_df, calendar_df, target_fiscal_months=[202502]
    )
    assert empty_mtd_actuals_df.empty  # guards this test's premise

    predicted_remaining_df = pd.DataFrame(
        {
            "unique_id": [10, 20],
            "fiscal_year_month": [202502, 202502],
            "ypred": [170.0, 70.0],
        }
    )

    result = combine_monthly_forecast(empty_mtd_actuals_df, predicted_remaining_df)

    # presence before value: a dropped row fails here rather than as a KeyError
    assert set(result["unique_id"]) == {10, 20}

    by_id = result.set_index("unique_id")
    assert by_id["mtd_revenue"].notna().all()
    assert by_id.loc[10, "mtd_revenue"] == 0
    assert by_id.loc[20, "mtd_revenue"] == 0
    assert by_id.loc[10, "monthly_forecast"] == 170.0
    assert by_id.loc[20, "monthly_forecast"] == 70.0


# ================================================
# build_monthly_forecast_vs_actual
# ================================================


def test_build_monthly_forecast_vs_actual_known_value(
    forecast_df: pd.DataFrame,
    train_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    actual_monthly_df: pd.DataFrame,
) -> None:
    result = build_monthly_forecast_vs_actual(
        forecast_df=forecast_df,
        train_df=train_df,
        calendar_df=calendar_df,
        actual_monthly_df=actual_monthly_df,
    )
    row_10 = result.set_index("unique_id").loc[10].to_dict()
    assert row_10["monthly_forecast"] == 420
    assert row_10["actual_monthly_total"] == 450

    row_20 = result.set_index("unique_id").loc[20].to_dict()
    assert row_20["monthly_forecast"] == 190
    assert row_20["actual_monthly_total"] == 200


def test_build_monthly_forecast_vs_actual_returns_one_row_per_series(
    forecast_df: pd.DataFrame,
    train_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    actual_monthly_df: pd.DataFrame,
) -> None:
    result = build_monthly_forecast_vs_actual(
        forecast_df=forecast_df,
        train_df=train_df,
        calendar_df=calendar_df,
        actual_monthly_df=actual_monthly_df,
    )

    assert len(result) == 2


def test_build_monthly_forecast_vs_actual_column_schema(
    forecast_df: pd.DataFrame,
    train_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    actual_monthly_df: pd.DataFrame,
) -> None:
    result = build_monthly_forecast_vs_actual(
        forecast_df=forecast_df,
        train_df=train_df,
        calendar_df=calendar_df,
        actual_monthly_df=actual_monthly_df,
    )

    assert set(result.columns) == {
        "unique_id",
        "fiscal_year_month",
        "mtd_revenue",
        "predicted_remaining",
        "monthly_forecast",
        "actual_monthly_total",
    }


def test_build_monthly_forecast_vs_actual_surfaces_add_back_components(
    forecast_df: pd.DataFrame,
    train_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    actual_monthly_df: pd.DataFrame,
) -> None:
    """Both add-back components reach the caller, and monthly_forecast is their sum.

    The sum assertion alone would pass if the two columns were swapped, so the
    hand-worked values are pinned individually as well. This is the invariant
    the sidecar's components file is checked against on read.
    """
    result = build_monthly_forecast_vs_actual(
        forecast_df=forecast_df,
        train_df=train_df,
        calendar_df=calendar_df,
        actual_monthly_df=actual_monthly_df,
    )

    by_id = result.set_index("unique_id")
    assert by_id.loc[10, "mtd_revenue"] == 250
    assert by_id.loc[10, "predicted_remaining"] == 170.0
    assert by_id.loc[20, "mtd_revenue"] == 120
    assert by_id.loc[20, "predicted_remaining"] == 70.0

    reconstructed = result["mtd_revenue"] + result["predicted_remaining"]
    assert (reconstructed == result["monthly_forecast"]).all()


# ================================================
# attach_tier_and_weight
# ================================================


def test_attach_tier_and_weight_known_value(
    tier_df: pd.DataFrame, weight_df: pd.DataFrame, fold_origin: pd.Timestamp
) -> None:
    """Merges tier/weight, attaches fold context, renames period_col, and
    selects the final leaderboard-ready column set and order."""
    monthly_rows_df = pd.DataFrame(
        {
            "unique_id": [10, 20],
            "fiscal_year_month": [202502, 202502],
            "monthly_forecast": [420.0, 190.0],
            "actual_monthly_total": [450, 200],
        }
    )
    result = attach_tier_and_weight(
        monthly_rows_df=monthly_rows_df,
        tier_df=tier_df,
        weight_df=weight_df,
        fold_origin=fold_origin,
        origin_month_fraction_elapsed=0.5,
    )

    assert list(result.columns) == [
        "forecast_origin_date",
        "predicted_fiscal_year_month",
        "unique_id",
        "tier",
        "monthly_forecast",
        "actual_monthly_total",
        "series_weight",
        "origin_month_fraction_elapsed",
    ]

    row_10 = result.set_index("unique_id").loc[10].to_dict()
    assert row_10["tier"] == "high"
    assert row_10["series_weight"] == 5.0
    assert row_10["forecast_origin_date"] == fold_origin
    assert row_10["origin_month_fraction_elapsed"] == 0.5
    assert row_10["predicted_fiscal_year_month"] == 202502


# ================================================
# compute_actual_monthly_totals
# ================================================


@pytest.fixture
def full_ts_df(calendar_df: pd.DataFrame) -> pd.DataFrame:
    """Full historical panel spanning both fiscal months completely —
    unlike train_df (fixture above), which stops mid-202502.
    compute_actual_monthly_totals() sums the whole panel, not a fold-scoped
    subset, so every week of both months must be present here to exercise
    that distinction from compute_mtd_actuals().

    Known values, worked by hand:
      id=10: 202501 (weeks 0-3) = 1+1+1+1 = 4; 202502 (weeks 4-7) = 10+20+30+40 = 100
      id=20: 202501 (weeks 0-3) = 2+2+2+2 = 8; 202502 (weeks 4-7) = 5+10+15+20 = 50
    """
    weeks = calendar_df["ds"].tolist()
    return pd.DataFrame(
        {
            "unique_id": [10] * 8 + [20] * 8,
            "ds": weeks + weeks,
            "y": [1, 1, 1, 1, 10, 20, 30, 40] + [2, 2, 2, 2, 5, 10, 15, 20],
        }
    )


def test_compute_actual_monthly_totals_known_value(
    full_ts_df: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """Sums y per (unique_id, fiscal_year_month) across the full panel."""
    result = compute_actual_monthly_totals(full_ts_df, calendar_df)

    row = result[(result["unique_id"] == 10) & (result["fiscal_year_month"] == 202502)]
    assert row["actual_monthly_total"].item() == 100

    row = result[(result["unique_id"] == 20) & (result["fiscal_year_month"] == 202502)]
    assert row["actual_monthly_total"].item() == 50


def test_compute_actual_monthly_totals_sums_all_months_not_just_one(
    full_ts_df: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """Sums every fiscal month present in ts_df, not just one target month —
    the key difference from compute_mtd_actuals()'s target_fiscal_months
    filtering. All 2 ids x 2 months are present, and id=10's other month
    (202501, not covered by the known-value test above) is spot-checked."""

    result = compute_actual_monthly_totals(full_ts_df, calendar_df)

    row = result[(result["unique_id"] == 10) & (result["fiscal_year_month"] == 202501)]
    assert row["actual_monthly_total"].item() == 4

    assert len(result) == 4
