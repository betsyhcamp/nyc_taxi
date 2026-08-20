"""Runtime invariant checks for origin-level modeling tables and their sidecars.

Gates raise AssertionError and return None. The single exception is
assert_benchmark_key_parity, which returns its counts for display.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from fcstnyctaxi.lib.origin_modeling_table._checks import require_columns
from fcstnyctaxi.lib.origin_modeling_table.column_roles import ModelingTableSchema
from fcstnyctaxi.lib.period_utils import label_horizon

IDENTITY_RTOL = 1e-9

SIDECAR_KEYS = ["forecast_origin_date", "predicted_fiscal_year_month", "unique_id"]

# Columns a challenger and its benchmark must agree on, being derived from the same
# panel and calendar by the same helpers. Required rather than compared-if-present:
# a sidecar missing them is malformed, and skipping is how a stripped frame passes.
SIDECAR_PARITY_NUMERIC = ["actual_monthly_total", "series_weight"]
SIDECAR_PARITY_EXACT = ["tier", "origin_month_fraction_elapsed"]


# ================================================
# Shared prep
# ================================================


def _sample_rows(df: pd.DataFrame, cols: list[str], n: int = 5) -> str:
    """Format up to n violating rows for an assertion message.

    Private to the gates to provide human readable info for when a violation occurs.

    Absent columns are skipped rather than raised on, because this runs *inside* the
    f-string that builds an assertion message. A raise here replaces the
    AssertionError with a KeyError and the real failure never surfaces — verified.
    No current call site depends on the leniency; it exists so that a future one
    cannot silently cost a diagnosis.
    """
    present = [c for c in cols if c in df.columns]
    return df[present].head(n).to_string(index=False)


def weekly_actuals_by_fiscal_week(
    panel: pd.DataFrame, calendar_df: pd.DataFrame
) -> pd.DataFrame:
    """Rekey the panel from dates to fiscal positions, keeping per-week values.

    The identity gates compare against raw per-week actuals, so this deliberately
    does not aggregate — the name says weekly_actual rather than a shape-only name
    to keep a summed frame from being passed in its place.

    Args:
        panel: Weekly panel; requires (unique_id, ds, y).
        calendar_df: Requires (ds, fiscal_year_month, fiscal_week_of_month).

    Returns:
        DataFrame with (unique_id, target_month, week_of_month, weekly_actual),
        one row per panel row.

    Raises:
        ValueError: If either frame lacks a required column, or if the calendar
            does not cover every panel date.
        MergeError: If calendar_df repeats a ds, which would fan the panel out.
    """
    require_columns(panel, ["unique_id", "ds", "y"], "panel")
    require_columns(
        calendar_df, ["ds", "fiscal_year_month", "fiscal_week_of_month"], "calendar_df"
    )

    uncovered = panel.loc[~panel["ds"].isin(calendar_df["ds"]), "ds"].unique()
    if len(uncovered):
        raise ValueError(
            f"calendar_df does not cover {len(uncovered)} panel date(s); "
            f"first few: {sorted(uncovered)[:5]}"
        )

    # validate="many_to_one" asserts the CALENDAR side is unique on ds. That is the
    # only way this function can manufacture a row that was not in its input: a
    # repeated ds fans the panel out, and the extra rows then merge-fan again in
    # every consuming gate. It deliberately says nothing about duplicate panel rows
    # — those arrive that way, and data-prep hygiene is not this module's job.
    return panel.merge(
        calendar_df[["ds", "fiscal_year_month", "fiscal_week_of_month"]],
        on="ds",
        how="left",
        validate="many_to_one",
    ).rename(
        columns={
            "fiscal_year_month": "target_month",
            "fiscal_week_of_month": "week_of_month",
            "y": "weekly_actual",
        }
    )[["unique_id", "target_month", "week_of_month", "weekly_actual"]]


def expected_sidecar_counts(
    origin_spine: pd.DataFrame,
    panel: pd.DataFrame,
    actual_monthly_df: pd.DataFrame,
    target_months: Sequence[int],
) -> dict[str, int]:
    """Derive what a sidecar should contain, from pipeline inputs only.

    Every count comes from the calendar spine, the panel, and the realized actuals &
    never from the sidecar being checked. That is what makes the comparison in
    assert_benchmark_key_parity a real one rather than a restatement, and it is why
    this takes the spine as a parameter rather than reaching for a framing's
    artifacts. Deriving rather than hardcoding also keeps the expectation correct when
    the evaluation window or the series count changes, which are routine.

    Args:
        origin_spine: Output of enumerate_origins; requires (target_month,
            forecast_origin_date).
        panel: The weekly panel; requires unique_id.
        actual_monthly_df: Requires (unique_id, fiscal_year_month); supplies the
            active (series, month) pairs.
        target_months: The fiscal months being evaluated.

    Returns:
        Dict with keys target_months, origins, series, events.
    """
    require_columns(
        origin_spine, ["target_month", "forecast_origin_date"], "origin_spine"
    )
    require_columns(panel, ["unique_id"], "panel")
    require_columns(
        actual_monthly_df, ["unique_id", "fiscal_year_month"], "actual_monthly_df"
    )

    months = list(target_months)
    origins = origin_spine[origin_spine["target_month"].isin(months)]
    active = actual_monthly_df[actual_monthly_df["fiscal_year_month"].isin(months)]
    # events is one row per (origin, active pair sharing that origin's month) which
    # the is the grid's own rule. This is an implementation check rather than an
    # independent authority. It still spans every step between panel and sidecar.
    events = origins[["target_month", "forecast_origin_date"]].merge(
        active.rename(columns={"fiscal_year_month": "target_month"}),
        on="target_month",
    )
    return {
        "target_months": len(months),
        "origins": int(origins["forecast_origin_date"].nunique()),
        # series comes from the ACTIVE pairs, not the panel: a series with history
        # but absent from every evaluated month legitimately produces no sidecar row,
        # and a panel-based count would demand one the grid cannot build (§6.2.2).
        "series": int(active["unique_id"].nunique()),
        "events": int(len(events)),
    }


def _active_pairs(
    actual_monthly_df: pd.DataFrame, target_months: Sequence[int]
) -> set[tuple]:
    """The (series, month) pairs a grid should contain, from the realized actuals.

    The row universe is active pairs, not the series x months product: a series
    absent from a month legitimately has no row, so a global product would fail on
    exactly the short history series the grid is built to handle.
    """
    restricted = actual_monthly_df[
        actual_monthly_df["fiscal_year_month"].isin(list(target_months))
    ]
    return set(
        zip(restricted["unique_id"], restricted["fiscal_year_month"], strict=False)
    )


# ================================================
# Feature-side gates
# ================================================


def assert_preprocess_feature_drift(
    weekly_features: pd.DataFrame,
    mlf_features: Sequence[str],
    *,
    lags: Sequence[int],
    lag_transforms: dict[int, list],
) -> None:
    """Assert preprocess emitted exactly the native names the framing declares.

    Emitted-versus-declared is the load-bearing check: it compares the declaration
    against what preprocess actually produced, so a mistyped native transform name
    fails here. Native names are neither obvious nor stable across MLForecast
    versions, which is what makes it worth having.Emptying all three consistently
    passes both, and that is correct rather than a hole: a framing may legitimately
    model on calendar and monthly features alone.

    Args:
        weekly_features: Output of build_weekly_features.
        mlf_features: The native MLForecast names the framing declares.
        lags: The lags passed to build_weekly_features.
        lag_transforms: The lag_transforms passed to build_weekly_features.
    """
    require_columns(
        weekly_features, ["unique_id", "feature_row_ds", "y"], "weekly_features"
    )

    emitted = set(weekly_features.columns) - {"unique_id", "feature_row_ds", "y"}
    declared = set(mlf_features)
    assert emitted == declared, (
        f"feature drift: preprocess emitted {sorted(emitted)}, "
        f"declaration has {sorted(declared)}; "
        f"only emitted: {sorted(emitted - declared)}, "
        f"only declared: {sorted(declared - emitted)}"
    )

    derived_count = len(lags) + sum(len(t) for t in lag_transforms.values())
    assert len(mlf_features) == derived_count, (
        f"declaration drift: {len(mlf_features)} names declared but lags and "
        f"lag_transforms imply {derived_count}"
    )


def assert_lag_alignment(weekly_features: pd.DataFrame, lags: Sequence[int]) -> None:
    """Assert each declared lag is y from that many periods earlier, on every row.

    This is the only check on MLForecast's time-index convention, which the whole
    feature_row_ds = origin + 1 period join rests on. It compares against the panel
    itself via a groupby shift, so a changed convention or a wrong freq fails here.
    An empty lags list returns early. A framing may legitimately declare no lags and
    model on calendar and monthly features alone, and then there is no lag
    convention to verify and guard below still works here.

    Args:
        weekly_features: Output of build_weekly_features; requires a lag{k} column
            for each declared lag.
        lags: The lags passed to build_weekly_features.
    """
    if not lags:
        return

    lag_cols = [f"lag{lag}" for lag in lags]
    require_columns(
        weekly_features,
        ["unique_id", "feature_row_ds", "y", *lag_cols],
        "weekly_features",
    )

    feats = weekly_features.sort_values(["unique_id", "feature_row_ds"])
    rows_per_series = feats.groupby("unique_id").size()

    for lag in lags:
        col = f"lag{lag}"
        prior_y = feats.groupby("unique_id")["y"].shift(lag)
        observed = feats[col].notna()

        expected_observed = int((rows_per_series - lag).clip(lower=0).sum())
        assert int(observed.sum()) == expected_observed, (
            f"{col} alignment checked {int(observed.sum())} rows, expected "
            f"{expected_observed} (per series, rows beyond the first {lag})"
        )

        violations = feats.loc[observed & (feats[col] != prior_y)]
        assert violations.empty, (
            f"{col} does not equal y from {lag} period(s) earlier for "
            f"{len(violations)} row(s):\n"
            f"{_sample_rows(violations, ['unique_id', 'feature_row_ds', 'y', col])}"
        )


# ================================================
# MTD-identity gates
# ================================================


def assert_mtd_construction(
    origin_target_table: pd.DataFrame,
    weekly_actuals: pd.DataFrame,
    actual_monthly_df: pd.DataFrame,
    *,
    rtol: float = IDENTITY_RTOL,
) -> None:
    """Assert MTD starts at zero and grows by exactly that week's raw actual.

    Compares with a tolerance rather than exactly: on integer proxy data every
    partial sum is exact, but on float net revenue the two summation paths need not
    bit-agree, so an exact comparison passes forever here and fails on work data for
    reasons unrelated to correctness.

    Args:
        origin_target_table: Requires (unique_id, target_month, weeks_actualized,
            mtd_revenue).
        weekly_actuals: Output of weekly_actuals_by_fiscal_week.
        actual_monthly_df: Requires (unique_id, fiscal_year_month); supplies the
            active-pair universe the coverage assert checks against.
        rtol: Relative tolerance for the increment comparison.
    """
    require_columns(
        origin_target_table,
        ["unique_id", "target_month", "weeks_actualized", "mtd_revenue"],
        "origin_target_table",
    )
    require_columns(
        weekly_actuals,
        ["unique_id", "target_month", "week_of_month", "weekly_actual"],
        "weekly_actuals",
    )
    require_columns(
        actual_monthly_df, ["unique_id", "fiscal_year_month"], "actual_monthly_df"
    )

    zero_rows = origin_target_table[origin_target_table["weeks_actualized"] == 0]
    covered = set(zip(zero_rows["unique_id"], zero_rows["target_month"], strict=False))
    expected = _active_pairs(
        actual_monthly_df, origin_target_table["target_month"].unique()
    )
    assert covered == expected, (
        f"MTD=0 coverage mismatch: {len(covered)} rows against "
        f"{len(expected)} active (series, month) pairs; "
        f"missing: {sorted(expected - covered)[:5]}, "
        f"unexpected: {sorted(covered - expected)[:5]}"
    )

    nonzero = zero_rows[zero_rows["mtd_revenue"] != 0]
    assert nonzero.empty, (
        f"{len(nonzero)} MTD=0 origin(s) carry nonzero mtd_revenue:\n"
        f"{_sample_rows(nonzero, ['unique_id', 'target_month', 'mtd_revenue'])}"
    )

    ordered = origin_target_table.sort_values(
        ["unique_id", "target_month", "weeks_actualized"]
    )
    ordered = ordered.assign(
        mtd_increment=ordered.groupby(["unique_id", "target_month"])[
            "mtd_revenue"
        ].diff()
    )
    increments = ordered[ordered["weeks_actualized"] >= 1].merge(
        weekly_actuals,
        left_on=["unique_id", "target_month", "weeks_actualized"],
        right_on=["unique_id", "target_month", "week_of_month"],
        how="left",
    )
    assert not increments.empty, (
        "MTD increment check selected no rows; every origin has weeks_actualized 0"
    )

    unmatched = increments[increments["weekly_actual"].isna()]
    assert unmatched.empty, (
        f"{len(unmatched)} origin(s) found no weekly actual to compare against:\n"
        f"{_sample_rows(unmatched, ['unique_id', 'target_month', 'weeks_actualized'])}"
    )

    close = np.isclose(
        increments["mtd_increment"], increments["weekly_actual"], rtol=rtol
    )
    violations = increments[~close]
    max_abs_diff = float(
        (increments["mtd_increment"] - increments["weekly_actual"]).abs().max()
    )
    report = ["unique_id", "target_month", "weeks_actualized"]
    report += ["mtd_increment", "weekly_actual"]
    assert violations.empty, (
        f"MTD increment != that week's actual for {len(violations)} row(s), "
        f"max abs diff {max_abs_diff:.6g} (rtol={rtol}):\n"
        f"{_sample_rows(violations, report)}"
    )


def assert_month_total_reconciliation(
    origin_target_table: pd.DataFrame,
    weekly_actuals: pd.DataFrame,
    actual_monthly_df: pd.DataFrame,
    *,
    rtol: float = IDENTITY_RTOL,
) -> None:
    """Assert the month total equals MTD at the last origin plus the final week.

    The month total is read from actual_monthly_df rather than from a column on the
    framing's table: Direct end of month framing names that quantity
    target_month_total_revenue and predicting remaining month framing names it
    actual_monthly_total, and a shared gate cannot hardcode either.

    Args:
        origin_target_table: Requires (unique_id, target_month, weeks_actualized,
            weeks_in_month, mtd_revenue).
        weekly_actuals: Output of weekly_actuals_by_fiscal_week.
        actual_monthly_df: Requires (unique_id, fiscal_year_month,
            actual_monthly_total); the authority for both the total and the
            active-pair universe.
        rtol: Relative tolerance for the reconciliation comparison.
    """
    require_columns(
        origin_target_table,
        [
            "unique_id",
            "target_month",
            "weeks_actualized",
            "weeks_in_month",
            "mtd_revenue",
        ],
        "origin_target_table",
    )
    require_columns(
        weekly_actuals,
        ["unique_id", "target_month", "week_of_month", "weekly_actual"],
        "weekly_actuals",
    )
    require_columns(
        actual_monthly_df,
        ["unique_id", "fiscal_year_month", "actual_monthly_total"],
        "actual_monthly_df",
    )

    last_origin = origin_target_table[
        origin_target_table["weeks_actualized"]
        == origin_target_table["weeks_in_month"] - 1
    ]
    covered = set(
        zip(last_origin["unique_id"], last_origin["target_month"], strict=False)
    )
    expected = _active_pairs(
        actual_monthly_df, origin_target_table["target_month"].unique()
    )
    assert covered == expected, (
        f"last-origin coverage mismatch: {len(covered)} pairs against "
        f"{len(expected)} active (series, month) pairs; "
        f"missing: {sorted(expected - covered)[:5]}, "
        f"unexpected: {sorted(covered - expected)[:5]}"
    )
    assert len(last_origin) == len(covered), (
        f"expected exactly one last-origin row per active pair, got "
        f"{len(last_origin)} rows for {len(covered)} pairs"
    )

    reconciliation = last_origin.merge(
        weekly_actuals,
        left_on=["unique_id", "target_month", "weeks_in_month"],
        right_on=["unique_id", "target_month", "week_of_month"],
        how="left",
    ).merge(
        actual_monthly_df[["unique_id", "fiscal_year_month", "actual_monthly_total"]],
        left_on=["unique_id", "target_month"],
        right_on=["unique_id", "fiscal_year_month"],
        how="left",
    )

    unmatched = reconciliation[
        reconciliation["weekly_actual"].isna()
        | reconciliation["actual_monthly_total"].isna()
    ]
    assert unmatched.empty, (
        f"{len(unmatched)} last-origin row(s) lack a final week or a month total:\n"
        f"{_sample_rows(unmatched, ['unique_id', 'target_month', 'weeks_in_month'])}"
    )

    reconstructed = reconciliation["mtd_revenue"] + reconciliation["weekly_actual"]
    close = np.isclose(reconstructed, reconciliation["actual_monthly_total"], rtol=rtol)
    violations = reconciliation[~close]
    max_abs_diff = float(
        (reconstructed - reconciliation["actual_monthly_total"]).abs().max()
    )
    report = ["unique_id", "target_month", "mtd_revenue"]
    report += ["weekly_actual", "actual_monthly_total"]
    assert violations.empty, (
        f"month total != mtd + final week for {len(violations)} row(s), "
        f"max abs diff {max_abs_diff:.6g} (rtol={rtol}):\n"
        f"{_sample_rows(violations, report)}"
    )


def assert_remaining_target_identity(
    origin_target_table: pd.DataFrame,
    weekly_actuals: pd.DataFrame,
    *,
    rtol: float = IDENTITY_RTOL,
) -> None:
    """Assert remaining_month_revenue equals the sum of its post-origin weeks.

    The independent derivation traces to raw per-week actuals, never to
    actual_monthly_total - mtd_revenue. Deriving it from the subtraction would
    compare the target to itself and pass structurally, which is the one vacuity
    mode no runtime guard can detect.

    Args:
        origin_target_table: Requires (unique_id, target_month, weeks_actualized,
            remaining_month_revenue).
        weekly_actuals: Output of weekly_actuals_by_fiscal_week.
        rtol: Relative tolerance for the comparison.
    """
    require_columns(
        origin_target_table,
        [
            "unique_id",
            "target_month",
            "weeks_actualized",
            "weeks_in_month",
            "remaining_month_revenue",
        ],
        "origin_target_table",
    )
    require_columns(
        weekly_actuals,
        ["unique_id", "target_month", "week_of_month", "weekly_actual"],
        "weekly_actuals",
    )

    assert not origin_target_table.empty, (
        "remaining-target identity checked no rows: origin_target_table is empty"
    )

    origin_keys = ["unique_id", "target_month", "weeks_actualized"]
    paired = origin_target_table[[*origin_keys, "weeks_in_month"]].merge(
        weekly_actuals, on=["unique_id", "target_month"], how="left"
    )
    post_origin = paired[paired["week_of_month"] > paired["weeks_actualized"]]
    derived = (
        post_origin.groupby(origin_keys)["weekly_actual"]
        .agg(derived_remaining="sum", weeks_found="size")
        .reset_index()
    )

    compared = origin_target_table.merge(derived, on=origin_keys, how="left")
    assert len(compared) == len(origin_target_table), (
        f"derivation changed the row count: {len(compared)} against "
        f"{len(origin_target_table)}"
    )

    # No fillna. Every origin has at least one post-origin week — weeks_actualized
    # runs 0..weeks_in_month-1, so the last origin still has exactly one — and the
    # count is weeks_in_month - weeks_actualized exactly (verified on real data).
    # Defaulting a missing derivation to 0.0 would make "the weekly actuals had no
    # rows for this pair" indistinguishable from a legitimate zero, and the gate
    # would pass vacuously wherever remaining_month_revenue happened to be 0.
    expected_weeks = compared["weeks_in_month"] - compared["weeks_actualized"]
    short = compared.loc[
        compared["weeks_found"].isna() | (compared["weeks_found"] != expected_weeks)
    ]
    report = [*origin_keys, "weeks_in_month", "weeks_found"]
    assert short.empty, (
        f"{len(short)} origin(s) lack the post-origin weeks their derivation needs "
        f"(expected weeks_in_month - weeks_actualized):\n"
        f"{_sample_rows(short, report)}"
    )

    close = np.isclose(
        compared["remaining_month_revenue"], compared["derived_remaining"], rtol=rtol
    )
    violations = compared[~close]
    max_abs_diff = float(
        (compared["remaining_month_revenue"] - compared["derived_remaining"])
        .abs()
        .max()
    )
    report = ["unique_id", "target_month", "weeks_actualized"]
    report += ["remaining_month_revenue", "derived_remaining"]
    assert violations.empty, (
        f"remaining_month_revenue != sum of post-origin weeks for "
        f"{len(violations)} row(s), max abs diff {max_abs_diff:.6g} (rtol={rtol}):\n"
        f"{_sample_rows(violations, report)}"
    )


# ================================================
# Join and cutoff gates
# ================================================


def assert_join_integrity(
    modeling_table: pd.DataFrame,
    origin_target_table: pd.DataFrame,
    weekly_features: pd.DataFrame,
) -> None:
    """Assert the feature join kept every origin exactly once and matched them all.

    Checks the merge key dtypes on both sides first: a dtype mismatch drops every
    row, and reporting that as a cardinality shortfall sends the reader looking in
    the wrong place.

    Args:
        modeling_table: The joined frame.
        origin_target_table: The frame joined from; requires (unique_id,
            forecast_origin_date).
        weekly_features: The frame joined to; requires (unique_id, feature_row_ds).
    """
    require_columns(modeling_table, ["unique_id", "feature_row_ds"], "modeling_table")
    require_columns(
        origin_target_table,
        ["unique_id", "forecast_origin_date"],
        "origin_target_table",
    )
    require_columns(weekly_features, ["unique_id", "feature_row_ds"], "weekly_features")

    assert not origin_target_table.empty, (
        "join integrity checked no rows: origin_target_table is empty"
    )
    assert modeling_table["unique_id"].dtype == weekly_features["unique_id"].dtype, (
        f"unique_id dtype differs across the join: "
        f"{modeling_table['unique_id'].dtype} against "
        f"{weekly_features['unique_id'].dtype}"
    )
    assert (
        modeling_table["feature_row_ds"].dtype
        == weekly_features["feature_row_ds"].dtype
    ), (
        f"feature_row_ds dtype differs across the join: "
        f"{modeling_table['feature_row_ds'].dtype} against "
        f"{weekly_features['feature_row_ds'].dtype}"
    )

    feature_keys = weekly_features[["unique_id", "feature_row_ds"]]
    assert not feature_keys.duplicated().any(), (
        f"weekly_features repeats {int(feature_keys.duplicated().sum())} "
        "(unique_id, feature_row_ds) key(s); the join would fan origins out"
    )

    keyed = origin_target_table[["unique_id", "forecast_origin_date"]].copy()
    keyed["feature_row_ds"] = keyed["forecast_origin_date"] + pd.Timedelta(weeks=1)
    antijoin = keyed.merge(
        feature_keys, on=["unique_id", "feature_row_ds"], how="left", indicator=True
    )
    unmatched = antijoin[antijoin["_merge"] == "left_only"]
    report = ["unique_id", "forecast_origin_date", "feature_row_ds"]
    assert unmatched.empty, (
        f"{len(unmatched)} origin(s) found no feature row:\n"
        f"{_sample_rows(unmatched, report)}"
    )

    assert len(modeling_table) == len(origin_target_table), (
        f"join changed the row count: {len(modeling_table)} against "
        f"{len(origin_target_table)}"
    )


def assert_shared_cutoff(
    modeling_table: pd.DataFrame,
    panel: pd.DataFrame,
    lags: Sequence[int],
    *,
    rtol: float = IDENTITY_RTOL,
) -> None:
    """Assert the joined lag features carry the same observed-through-W cutoff as MTD.

    The join-offset check runs for every declared lag: at an origin of W, lag k must
    equal the panel's y at W - (k-1) periods, because feature_row_ds is W + 1 and
    MLForecast emits lag k at ds as y[ds - k]. This is the only check on that offset.

    The MTD cross-check runs only when lag 1 is declared, because it is arithmetically
    wrong otherwise: the MTD increment at an interior origin is that week's y, which
    equals lag 1 and not lag 2 - verified, lag 2 is a week staler.

    An empty lags list returns early. A framing may legitimately declare no lags

    Known gap, named rather than assumed covered: a framing declaring only
    lag_transforms and no plain lags gets no verification of the W + 1 offset.


    Args:
        modeling_table: Requires (unique_id, forecast_origin_date, target_month,
            weeks_actualized, mtd_revenue) and a lag{k} column per declared lag.
        panel: The weekly panel the features were built from; requires
            (unique_id, ds, y).
        lags: The lags passed to build_weekly_features.
        rtol: Relative tolerance for both comparisons.
    """
    if not lags:
        return

    lag_cols = [f"lag{lag}" for lag in lags]
    require_columns(
        modeling_table,
        [
            "unique_id",
            "forecast_origin_date",
            "target_month",
            "weeks_actualized",
            "mtd_revenue",
            *lag_cols,
        ],
        "modeling_table",
    )
    require_columns(panel, ["unique_id", "ds", "y"], "panel")

    if 1 in lags:
        ordered = modeling_table.sort_values(
            ["unique_id", "target_month", "weeks_actualized"]
        )
        ordered = ordered.assign(
            mtd_increment=ordered.groupby(["unique_id", "target_month"])[
                "mtd_revenue"
            ].diff()
        )
        interior = ordered["weeks_actualized"] >= 1
        assert int(interior.sum()) > 0, (
            "MTD-versus-lag1 check selected no rows; every origin has "
            "weeks_actualized 0"
        )
        increment_violations = ordered.loc[
            interior & ~np.isclose(ordered["mtd_increment"], ordered["lag1"], rtol=rtol)
        ]
        report = ["unique_id", "forecast_origin_date", "mtd_increment", "lag1"]
        assert increment_violations.empty, (
            f"MTD increment != lag1 for {len(increment_violations)} row(s):\n"
            f"{_sample_rows(increment_violations, report)}"
        )

    for lag in lags:
        col = f"lag{lag}"
        probed = modeling_table.assign(
            probe_ds=modeling_table["forecast_origin_date"]
            - pd.Timedelta(weeks=lag - 1)
        )
        check = probed.merge(
            panel.rename(columns={"ds": "probe_ds", "y": "y_probe"})[
                ["unique_id", "probe_ds", "y_probe"]
            ],
            on=["unique_id", "probe_ds"],
            how="left",
        )
        observed = check["y_probe"].notna() & check[col].notna()

        in_panel = int(
            probed.merge(
                panel[["unique_id", "ds"]].rename(columns={"ds": "probe_ds"}),
                on=["unique_id", "probe_ds"],
                how="inner",
            ).shape[0]
        )
        assert in_panel > 0, (
            f"{col}-versus-panel check selected no rows; no origin probe date appears "
            "in the panel, which a merge-key dtype mismatch would also produce"
        )
        # Equality, not > 0. A row whose probe date is in the panel must also carry a
        # non-NaN lag: lag k at feature_row_ds W+1 is y[W-(k-1)], the very value the
        # probe fetches. They can only diverge on a gapped panel, which cannot reach
        # this gate because MLForecast's preprocess rejects one outright. Asserting
        # > 0 would let the gate compare a single row and skip thousands.
        keys_and_probe = ["unique_id", "forecast_origin_date", "probe_ds"]
        skipped = check.loc[check["y_probe"].notna() & check[col].isna()]
        assert int(observed.sum()) == in_panel, (
            f"{col}-versus-panel check compared {int(observed.sum())} rows but "
            f"{in_panel} origins have a probe date in the panel; "
            f"{len(skipped)} row(s) were skipped for a NaN {col}:\n"
            f"{_sample_rows(skipped, [*keys_and_probe, col])}"
        )

        violations = check.loc[
            observed & ~np.isclose(check[col], check["y_probe"], rtol=rtol)
        ]
        report = ["unique_id", "forecast_origin_date", "probe_ds", col, "y_probe"]
        assert violations.empty, (
            f"{col} != the panel's y at {lag - 1} period(s) before the origin for "
            f"{len(violations)} row(s):\n{_sample_rows(violations, report)}"
        )


def assert_no_future_leakage(
    panel: pd.DataFrame,
    rebuild_fn: Callable[[pd.DataFrame], pd.DataFrame],
    schema: ModelingTableSchema,
    cutoff: pd.Timestamp,
    *,
    y_derived_features: Sequence[str],
    calendar_derived_features: Sequence[str],
    seed: int = 0,
    scale: float = 1e6,
) -> None:
    """Assert no feature at an origin at or before the cutoff moves when future y does.

    Scrambles every y after the cutoff, rebuilds through the caller's rebuild_fn, and
    compares. rebuild_fn is injected because it must re-derive the framing's own
    target, which this module does not know how to build.

    The comparison set is schema.feature_cols exactly, not everything but the target:
    a framing may carry a non-feature passthrough that legitimately moves.

    Args:
        panel: The weekly panel; requires (unique_id, ds, y).
        rebuild_fn: Rebuilds the modeling table from a panel.
        schema: Declares which columns are the model matrix.
        cutoff: Origins at or before this date must not move.
        y_derived_features: Features that must move under a y-scramble.
        calendar_derived_features: Features that cannot move under a y-scramble.
            Together with y_derived_features this must partition schema.feature_cols
            exactly, so adding a feature without classifying it fails here. A framing
            may declare every feature calendar-derived: the positive control then has
            no subject, and in its place the gate asserts that no feature moves after
            the cutoff either, which is what "nothing depends on y" claims.
        seed: Seed for the scramble.
        scale: Scramble magnitude; large so a leak moves loudly.
    """
    require_columns(panel, ["unique_id", "ds", "y"], "panel")

    y_set, cal_set = set(y_derived_features), set(calendar_derived_features)
    features = set(schema.feature_cols)
    overlap = y_set & cal_set
    assert not overlap, (
        f"a feature is declared both y- and calendar-derived: {sorted(overlap)}"
    )
    assert y_set | cal_set == features, (
        f"the feature partition does not cover schema.feature_cols; "
        f"unclassified: {sorted(features - (y_set | cal_set))}, "
        f"not a feature: {sorted((y_set | cal_set) - features)}"
    )

    scrambled_panel = panel.copy()
    scrambled_panel["y"] = scrambled_panel["y"].astype(float)
    future = scrambled_panel["ds"] > cutoff
    assert int(future.sum()) > 0, (
        f"nothing perturbed: no panel row falls after the cutoff {cutoff}"
    )
    scrambled_panel.loc[future, "y"] = (
        np.random.default_rng(seed).random(int(future.sum())) * scale
    )

    base = rebuild_fn(panel)
    scrambled = rebuild_fn(scrambled_panel)
    keys = ["unique_id", "forecast_origin_date"]

    def _at(frame: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
        return schema.select_features(frame[mask].sort_values(keys)).reset_index(
            drop=True
        )

    past_base = _at(base, base["forecast_origin_date"] <= cutoff)
    past_scrambled = _at(scrambled, scrambled["forecast_origin_date"] <= cutoff)
    assert len(past_base) > 0, (
        f"nothing checked: no origin falls at or before the cutoff {cutoff}"
    )

    same = (past_base == past_scrambled) | (past_base.isna() & past_scrambled.isna())
    moved = [c for c in same.columns if not same[c].all()]
    assert not moved, (
        f"LEAK: future y changed {len(moved)} feature(s) at origins at or before "
        f"the cutoff {cutoff}: {moved}"
    )

    future_base = _at(base, base["forecast_origin_date"] > cutoff)
    future_scrambled = _at(scrambled, scrambled["forecast_origin_date"] > cutoff)
    assert len(future_base) > 0, (
        f"positive control cannot run: no origin falls after the cutoff {cutoff}"
    )
    if y_set:
        control_cols = sorted(y_set)
        control_same = (future_base[control_cols] == future_scrambled[control_cols]) | (
            future_base[control_cols].isna() & future_scrambled[control_cols].isna()
        )
        assert not control_same.all().all(), (
            "positive control failed: scrambling future y moved no y-derived feature "
            "at any origin after the cutoff, so this gate cannot detect a leak"
        )
    else:
        # Nothing is declared y-derived, so the positive control has no subject. That
        # declaration is itself checkable and stronger than skipping: if no feature
        # depends on y, then no feature may move anywhere, not merely at origins
        # at or before the cutoff.
        future_same = (future_base == future_scrambled) | (
            future_base.isna() & future_scrambled.isna()
        )
        moved_after = [c for c in future_same.columns if not future_same[c].all()]
        assert not moved_after, (
            "the feature partition declares no y-derived features, but scrambling "
            f"future y moved {len(moved_after)} feature(s) after the cutoff: "
            f"{moved_after}. Either the declaration is wrong or the feature is."
        )


def assert_fold_is_populated(modeling_table: pd.DataFrame, val_month: int) -> None:
    """Assert the fold has a validation month present and rows on both sides.

    Checking presence and both side counts catches issues at the fold level.

    Args:
        modeling_table: Requires a target_month column.
        val_month: The fiscal month held out for this fold.
    """
    require_columns(modeling_table, ["target_month"], "modeling_table")

    months = set(modeling_table["target_month"])
    assert val_month in months, (
        f"val_month {val_month} is absent from the modeling table; "
        f"present months: {sorted(months)}"
    )

    n_train = int((modeling_table["target_month"] < val_month).sum())
    n_val = int((modeling_table["target_month"] == val_month).sum())
    assert n_train > 0, f"no training rows before val_month {val_month}"
    assert n_val > 0, f"no validation rows at val_month {val_month}"


# ================================================
# Sidecar gates
# ================================================


def assert_all_horizon_1(
    monthly_series: pd.DataFrame, calendar_df: pd.DataFrame
) -> None:
    """Assert every labeled row is horizon_1, for a current-month-only framing.

    Args:
        monthly_series: The assembled sidecar frame.
        calendar_df: Fiscal calendar, for the origin-to-fiscal-month map.
    """
    horizon = label_horizon(monthly_series, calendar_df)
    assert len(horizon) > 0, "horizon check labeled no rows; monthly_series is empty"

    violations = monthly_series.loc[horizon != "horizon_1"].assign(
        horizon=horizon[horizon != "horizon_1"]
    )
    report = ["forecast_origin_date", "predicted_fiscal_year_month", "horizon"]
    assert violations.empty, (
        f"{len(violations)} row(s) are not horizon_1:\n"
        f"{_sample_rows(violations, report)}"
    )


def assert_tier_categorical(monthly_series: pd.DataFrame, num_tiers: int = 5) -> None:
    """Assert tier is categorical and carries the full set of tiers, not a subset.

    Label-agnostic on purpose: nothing downstream requires particular tier names.
    leaderboard.py reads tier.cat.categories as the tier universe and iterates it, so
    what matters is that the set is complete, not what it is called. Only the count
    is declared, so renaming a tier needs no change here.

    Two assertions, and the dtype one is a precondition rather than a peer.
    .cat.categories exists only on a categorical, so checking the count alone would
    surface a non-categorical as "Can only use .cat accessor with a 'category'
    dtype" — an AttributeError naming pandas rather than the defect, and the wrong
    exception type for a suite that expects AssertionError.

    They also catch different failures. assign_tiers computes
    effective_tiers = min(num_tiers, mean_pos.nunique()) and slices the labels, so a
    fold with too few distinct means truncates. Folds whose ladders disagree concat
    to object dtype — the first assertion. Every fold truncating identically stays
    categorical and short — the second.

    Args:
        monthly_series: Requires a tier column.
        num_tiers: How many tiers assign_tiers was asked for. Defaults to its own
            default, so a framing using standard tiering states nothing.
    """
    require_columns(monthly_series, ["tier"], "monthly_series")

    dtype = monthly_series["tier"].dtype
    assert isinstance(dtype, pd.CategoricalDtype), (
        f"tier must be a categorical dtype, got {dtype}; folds whose tier ladders "
        "disagree concat to object, which is what a partially-truncated run looks like"
    )

    categories = list(dtype.categories)
    assert len(categories) == num_tiers, (
        f"tier carries {len(categories)} categories, expected {num_tiers}: "
        f"{categories}. Every fold truncated to the same short ladder, which the "
        "dtype check cannot see and the leaderboard would read as the complete set"
    )


def assert_benchmark_key_parity(
    monthly_series: pd.DataFrame,
    benchmark_h1: pd.DataFrame,
    *,
    expected_counts: dict[str, int],
    rtol: float = IDENTITY_RTOL,
) -> dict[str, int]:
    """Assert the challenger and benchmark cover identical keys and shared values.

    The key-set comparison is the check: the benchmark is an independently produced
    registered artifact, so any row the challenger gains or loses shows up by name.
    The counts are the common-mode guard, and they are not redundant with it. Two
    artifacts compared against each other catch divergence.

    Args:
        monthly_series: The challenger sidecar frame.
        benchmark_h1: The benchmark frame, already filtered to horizon_1 and the
            evaluated months.
        expected_counts: From expected_sidecar_counts. Required keys target_months,
            origins, series, events.
        rtol: Relative tolerance for the shared numeric columns.

    Returns:
        The observed counts, same keys as expected_counts, so the audit C printed
        for a human stays printable.
    """
    required = [*SIDECAR_KEYS, *SIDECAR_PARITY_NUMERIC, *SIDECAR_PARITY_EXACT]
    for frame, name in (
        (monthly_series, "monthly_series"),
        (benchmark_h1, "benchmark_h1"),
    ):
        require_columns(frame, required, name)
        # Before any set comparison: duplicates collapse into a set, so two frames
        # duplicated differently can compare equal on keys while differing in rows.
        dupes = int(frame.duplicated(subset=SIDECAR_KEYS).sum())
        assert dupes == 0, (
            f"{name} repeats {dupes} (forecast_origin_date, "
            "predicted_fiscal_year_month, unique_id) key(s); the key-set comparison "
            "below would collapse them and pass"
        )

    counts = {
        "target_months": int(monthly_series["predicted_fiscal_year_month"].nunique()),
        "origins": int(monthly_series["forecast_origin_date"].nunique()),
        "series": int(monthly_series["unique_id"].nunique()),
        "events": int(len(monthly_series)),
    }
    missing_keys = [k for k in counts if k not in expected_counts]
    assert not missing_keys, f"expected_counts is missing keys: {missing_keys}"
    mismatched = {
        k: (counts[k], expected_counts[k])
        for k in counts
        if counts[k] != expected_counts[k]
    }
    assert not mismatched, f"count mismatch (observed, expected): {mismatched}"

    challenger_keys = set(zip(*(monthly_series[c] for c in SIDECAR_KEYS), strict=False))
    bench_keys = set(zip(*(benchmark_h1[c] for c in SIDECAR_KEYS), strict=False))
    assert challenger_keys == bench_keys, (
        f"key sets differ by {len(challenger_keys ^ bench_keys)} key(s); "
        f"first few: {sorted(challenger_keys ^ bench_keys)[:5]}"
    )

    parity = monthly_series.merge(
        benchmark_h1, on=SIDECAR_KEYS, suffixes=("_c", "_bench")
    )
    assert len(parity) == len(monthly_series), (
        f"parity merge changed the row count: {len(parity)} against "
        f"{len(monthly_series)}"
    )

    for column in SIDECAR_PARITY_NUMERIC:
        left, right = f"{column}_c", f"{column}_bench"
        violations = parity[~np.isclose(parity[left], parity[right], rtol=rtol)]
        assert violations.empty, (
            f"{column} differs between challenger and benchmark for "
            f"{len(violations)} row(s):\n"
            f"{_sample_rows(violations, SIDECAR_KEYS + [left, right])}"
        )

    for column in SIDECAR_PARITY_EXACT:
        left, right = f"{column}_c", f"{column}_bench"
        violations = parity[parity[left] != parity[right]]
        assert violations.empty, (
            f"{column} differs between challenger and benchmark for "
            f"{len(violations)} row(s):\n"
            f"{_sample_rows(violations, SIDECAR_KEYS + [left, right])}"
        )

    return counts


def assert_forecast_reconstruction(
    monthly_series: pd.DataFrame, components_df: pd.DataFrame
) -> None:
    """Assert every monthly_series row has a component row that rebuilds its forecast.

    A row-alignment check across sidecar assembly: the outer
    merge with an indicator is what catches a row present in one file and absent
    from the other, and that is the half that can fail today.

    Args:
        monthly_series: Requires the three sidecar keys and monthly_forecast.
        components_df: Requires the three sidecar keys, mtd_revenue and
            predicted_remaining.
    """
    require_columns(
        monthly_series, [*SIDECAR_KEYS, "monthly_forecast"], "monthly_series"
    )
    require_columns(
        components_df,
        [*SIDECAR_KEYS, "mtd_revenue", "predicted_remaining"],
        "components_df",
    )

    merged = monthly_series[[*SIDECAR_KEYS, "monthly_forecast"]].merge(
        components_df[[*SIDECAR_KEYS, "mtd_revenue", "predicted_remaining"]],
        on=SIDECAR_KEYS,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    assert not merged.empty, "forecast reconstruction checked no rows"
    assert len(merged) == len(monthly_series), (
        f"components do not align one-to-one with monthly_series: merged to "
        f"{len(merged)} rows against {len(monthly_series)}"
    )

    unaligned = merged[merged["_merge"] != "both"]
    assert unaligned.empty, (
        f"{len(unaligned)} key(s) appear in only one file:\n"
        f"{_sample_rows(unaligned, [*SIDECAR_KEYS, '_merge'])}"
    )

    reconstructed = merged["mtd_revenue"] + merged["predicted_remaining"]
    violations = merged[reconstructed != merged["monthly_forecast"]]
    report = [*SIDECAR_KEYS, "mtd_revenue", "predicted_remaining", "monthly_forecast"]
    assert violations.empty, (
        f"mtd + predicted_remaining != monthly_forecast for "
        f"{len(violations)} row(s):\n"
        f"{_sample_rows(violations, report)}"
    )
