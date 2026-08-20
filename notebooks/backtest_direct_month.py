# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: nyc_taxi (3.12.9.final.0)
#     language: python
#     name: python3
# ---

# %%
import json
import sys

import pandas as pd
import yaml
from lightgbm import LGBMRegressor
from mlforecast.lag_transforms import RollingMean
from tsbricks.blocks.metadata import get_git_hash, get_uv_lock_info

from fcstnyctaxi.lib.config_utils import save_config
from fcstnyctaxi.lib.fold_metrics import compute_wrmae_pooled
from fcstnyctaxi.lib.io import write_text_to_gcs
from fcstnyctaxi.lib.monthly_aggregation import (
    attach_tier_and_weight,
    compute_actual_monthly_totals,
)
from fcstnyctaxi.lib.origin_modeling_table.builders import (
    attach_mtd_revenue,
    attach_weekly_features,
    attach_workday_progress,
    build_origin_series_grid,
    build_weekly_features,
    enumerate_origins,
    trim_incomplete_series_months,
)
from fcstnyctaxi.lib.origin_modeling_table.column_roles import ModelingTableSchema
from fcstnyctaxi.lib.origin_modeling_table.gates import (
    assert_all_horizon_1,
    assert_benchmark_key_parity,
    assert_fold_is_populated,
    assert_join_integrity,
    assert_lag_alignment,
    assert_month_total_reconciliation,
    assert_mtd_construction,
    assert_no_future_leakage,
    assert_preprocess_feature_drift,
    assert_shared_cutoff,
    assert_tier_categorical,
    expected_sidecar_counts,
    weekly_actuals_by_fiscal_week,
)
from fcstnyctaxi.lib.period_utils import (
    assign_tiers,
    compute_series_weights,
    label_horizon,
)
from fcstnyctaxi.lib.utils import generate_run_id, get_project_root_dir

# %%
project_root = get_project_root_dir()
sys.path.insert(0, str(project_root))

run_config_path = project_root / "notebooks" / "backtest_configs" / "run_config.yaml"
run_cfg = yaml.safe_load(run_config_path.read_text())

bucket = run_cfg["project"]["gcs_bucket"]
timeseries_uri = f"{bucket}/{run_cfg['project']['time_series_uri']}"
calendar_uri = f"{bucket}/{run_cfg['project']['fiscal_calendar_uri']}"

ts_df = pd.read_parquet(timeseries_uri)
calendar_df = pd.read_parquet(calendar_uri)

# %%
# TODO: refactor to remove hardcoding &, instead, use parameter file for these items
# Hardcoded for now; registry lookup (resolve_sidecar_uri) is deferred
BENCHMARK_SIDECAR_URI = (
    "gs://nyc-taxi-ehc--modeling/dev/backtests/backtest_weekly/20260803T230103657210Z/"
)

FREQ = "W-SUN"
MLF_LAGS = [1]
MLF_LAG_TRANSFORMS = {1: [RollingMean(window_size=4)]}

# Model-matrix allowlist (fail-closed): nothing is a feature unless named here.
# Custom features (G1/G3) hand-declared (leakage-prone surface). MLForecast
# native names live once in MLF_FEATURES and are drift-gated.
G1_FEATURES = [
    "mtd_revenue",
    "workdays_elapsed",
    "workdays_remaining",
    "number_workdays",
]
MLF_FEATURES = ["lag1", "rolling_mean_lag1_window_size4"]
G3_FEATURES = ["last_completed_month_revenue"]
FEATURE_COLUMNS = G1_FEATURES + MLF_FEATURES + G3_FEATURES

TARGET_MONTHS = [202504, 202505, 202506, 202507, 202508, 202509]
# Inputs to the future-scramble leakage gate. cutoff_W must be an origin with
# weeks_actualized == 1 in a 5-week month, so the scrambled region sits inside the
# target month and MTD is exercised; it stays hardcoded because it is an operator's
# choice of scenario rather than a fact about the data (spec §12). The partition must
# cover FEATURE_COLUMNS exactly — adding a feature without classifying it fails the gate.
cutoff_W = pd.Timestamp("2025-05-25")
Y_DERIVED_FEATURE_COLS = ["mtd_revenue", *MLF_FEATURES, *G3_FEATURES]
CALENDAR_DERIVED_FEATURE_COLS = [
    "workdays_elapsed",
    "workdays_remaining",
    "number_workdays",
]


# Column roles for the modeling table. Declared here because which columns are
# features is this framing's decision; the library only applies the declaration.
MODELING_SCHEMA = ModelingTableSchema(
    key_cols=("unique_id", "forecast_origin_date", "target_month"),
    feature_cols=tuple(FEATURE_COLUMNS),
    target_col="target_month_total_revenue",
    passthrough_cols=("feature_row_ds",),
    progress_cols=("weeks_actualized", "weeks_in_month"),
)

HYPERPARAMS = {
    "objective": "regression_l1",
    "n_estimators": 400,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "min_data_in_leaf": 125,
    "n_jobs": 1,
    "random_state": 0,
    "verbosity": -1,
}

sidecar_uri = f"{bucket}/dev/backtests/backtest_direct_month/{generate_run_id()}/"


# %%
# ============ Framing specific cleaning: Trim incomplete months ============
# If target used in training is full month, need all months to have all weeks present
# this cleaning makes sense to move into data_prep.py/SQL script if this direct month
# problem framing has better forecast error than weekly forecast then aggregate framing
# Until make final determination on which problem framing is best, leave this
# framing-specific cleaning here.

ts_df, dropped_pairs = trim_incomplete_series_months(
    panel_df=ts_df,
    calendar_df=calendar_df,
)
# %%
dropped_pairs

# %%
actual_monthly_df = compute_actual_monthly_totals(
    ts_df=ts_df,
    calendar_df=calendar_df,
    period_col="fiscal_year_month",
    time_col="ds",
    id_col="unique_id",
    target_col="y",
)

# %%
weekly_features = build_weekly_features(
    ts_df, FREQ, lags=MLF_LAGS, lag_transforms=MLF_LAG_TRANSFORMS
)

assert_preprocess_feature_drift(
    weekly_features, MLF_FEATURES, lags=MLF_LAGS, lag_transforms=MLF_LAG_TRANSFORMS
)

# %%
weekly_features = weekly_features.sort_values(
    by=["unique_id", "feature_row_ds"]
).reset_index(drop=True)
assert_lag_alignment(weekly_features, MLF_LAGS)

# %%
weekly_features


# %%
def build_origin_target_table(panel, calendar_df, origin_spine, actual_monthly_df):
    """Compose the shared builders into this notebook's origin/target table.

    Everything after the builder calls is specific to the direct-month framing 
    including its target column name and its last_completed_month_revenue feature.
    """
    grid_df = build_origin_series_grid(origin_spine, actual_monthly_df)
    table = attach_mtd_revenue(grid_df, panel, calendar_df)
    table = table.rename(columns={"actual_monthly_total": "target_month_total_revenue"})

    # last-completed (M-1): NaN for a series' FIRST active month (no prior month)
    months_sorted = sorted(calendar_df["fiscal_year_month"].unique())
    prev_month_lookup = pd.Series(months_sorted, index=months_sorted).shift(1)
    # casting as int problematic if NaNs allowed in prev_month column
    table["prev_month"] = table["target_month"].map(prev_month_lookup).astype(int)
    table = table.merge(
        actual_monthly_df.rename(
            columns={
                "fiscal_year_month": "prev_month",
                "actual_monthly_total": "last_completed_month_revenue",
            }
        ),
        on=["unique_id", "prev_month"],
        how="left",
    )

    cols = [
        "unique_id",
        "forecast_origin_date",
        "target_month",
        "mtd_revenue",
        "workdays_elapsed",
        "workdays_remaining",
        "number_workdays",
        "weeks_actualized",
        "weeks_in_month",
        "last_completed_month_revenue",
        "target_month_total_revenue",
    ]
    return table[cols].reset_index(drop=True)


# %%
origin_spine = enumerate_origins(calendar_df)
origin_spine = attach_workday_progress(origin_spine, calendar_df)
# %%
origin_spine.tail(30)

# %%
origin_target_table = build_origin_target_table(
    ts_df, calendar_df, origin_spine, actual_monthly_df
)

# %%
# raw per-week actuals, the independent source both MTD-identity gates compare against
weekly_actuals = weekly_actuals_by_fiscal_week(ts_df, calendar_df)

assert_mtd_construction(origin_target_table, weekly_actuals, actual_monthly_df)
assert_month_total_reconciliation(origin_target_table, weekly_actuals, actual_monthly_df)


# %%
origin_target_table.head()


# %%
modeling_table = attach_weekly_features(origin_target_table, weekly_features)
modeling_table = MODELING_SCHEMA.select(modeling_table)
MODELING_SCHEMA.validate(modeling_table)

# %%
assert_join_integrity(modeling_table, origin_target_table, weekly_features)


# %%
modeling_table.head()

# %%
assert_shared_cutoff(modeling_table, ts_df, MLF_LAGS)

# %%
# ====== Gate: general leakage smoke test (future-scramble) ===========

def rebuild_all(panel):
    actual_monthly = compute_actual_monthly_totals(
        ts_df=panel,
        calendar_df=calendar_df,
        period_col="fiscal_year_month",
        time_col="ds",
        id_col="unique_id",
        target_col="y",
    )
    features = build_weekly_features(
        panel, FREQ, lags=MLF_LAGS, lag_transforms=MLF_LAG_TRANSFORMS
    )
    ot = build_origin_target_table(panel, calendar_df, origin_spine, actual_monthly)
    return MODELING_SCHEMA.select(attach_weekly_features(ot, features))


assert_no_future_leakage(
    ts_df,
    rebuild_all,
    MODELING_SCHEMA,
    cutoff_W,
    y_derived_features=Y_DERIVED_FEATURE_COLS,
    calendar_derived_features=CALENDAR_DERIVED_FEATURE_COLS,
)


# %%
def run_fold(modeling_table, val_month):
    is_train = modeling_table["target_month"] < val_month
    is_val = modeling_table["target_month"] == val_month
    assert_fold_is_populated(modeling_table, val_month)

    X = MODELING_SCHEMA.select_features(modeling_table)
    y = modeling_table[MODELING_SCHEMA.target_col]
    X_train, X_val = X[is_train], X[is_val]
    y_train = y[is_train]

    model = LGBMRegressor(**HYPERPARAMS).fit(X_train, y_train)
    preds = model.predict(X_val)

    return modeling_table.loc[
        is_val,
        list(MODELING_SCHEMA.key_cols)
        + list(MODELING_SCHEMA.progress_cols)
        + [MODELING_SCHEMA.target_col],
    ].assign(prediction=preds)


# %%
# produce backtest forecasts
forecasts_df = pd.concat(
    [run_fold(modeling_table, val_month) for val_month in TARGET_MONTHS],
    ignore_index=True,
)

# %%
len(forecasts_df)

# %%
forecasts_df.head()

# %%
# assemble forecasts w/ tier, weight, added items needed later for scoring backtests
fraction_by_origin = calendar_df.set_index("ds")["origin_month_fraction_elapsed"]

monthly_rows_rename_dict = {
    "prediction": "monthly_forecast",
    "target_month_total_revenue": "actual_monthly_total",
    "target_month": "fiscal_year_month",
}
monthly_rows_col_keep = [
    "unique_id",
    "fiscal_year_month",
    "monthly_forecast",
    "actual_monthly_total",
]
rows = []
for origin_date, origin_group in forecasts_df.groupby("forecast_origin_date"):
    monthly_rows_df = origin_group.rename(columns=monthly_rows_rename_dict)[
        monthly_rows_col_keep
    ]
    tier_df = assign_tiers(
        train_df=ts_df, origin_date=origin_date, calendar_df=calendar_df
    )

    weight_df = compute_series_weights(
        train_df=ts_df, origin_date=origin_date, calendar_df=calendar_df
    )
    origin_group_built = attach_tier_and_weight(
        monthly_rows_df=monthly_rows_df,
        tier_df=tier_df,
        weight_df=weight_df,
        fold_origin=origin_date,
        origin_month_fraction_elapsed=fraction_by_origin[origin_date],
    )
    rows.append(origin_group_built)

monthly_series = pd.concat(rows, ignore_index=True)

# %%
len(monthly_series)

# %%
monthly_series.head()

# %%
# Gate: since we only predict the current month, every row must be horizon_1
assert_all_horizon_1(monthly_series, calendar_df)

# %%
# prep sidecar: per_series_mtd file
val_months = modeling_table["target_month"].isin(TARGET_MONTHS)

per_series_mtd = (
    modeling_table.loc[
        val_months, ["unique_id", "forecast_origin_date", "target_month", "mtd_revenue"]
    ]
    .rename(columns={"target_month": "predicted_fiscal_year_month"})
    .reset_index(drop=True)
)


# %%
per_series_mtd.head(2)

# %%
monthly_series.head(2)

# %%
# prep sidecar: metrics file
keys = ["forecast_origin_date", "predicted_fiscal_year_month", "unique_id"]
per_series = monthly_series.merge(per_series_mtd, on=keys)

per_series["pred_below"] = per_series["monthly_forecast"] < per_series["mtd_revenue"]
per_series["actual_below"] = (
    per_series["actual_monthly_total"] < per_series["mtd_revenue"]
)
per_series["pred_violation"] = (
    per_series["mtd_revenue"] - per_series["monthly_forecast"]
).where(per_series["pred_below"])
per_series["actual_violation"] = (
    per_series["mtd_revenue"] - per_series["actual_monthly_total"]
).where(per_series["actual_below"])

metrics = per_series.groupby(
    ["forecast_origin_date", "predicted_fiscal_year_month"], as_index=False
).agg(
    n_series=("unique_id", "size"),
    frac_pred_below_mtd=("pred_below", "mean"),  # mean of bool = fraction
    mean_pred_below_mtd_violation=("pred_violation", "mean"),  # skips NaNs
    max_pred_below_mtd_violation=("pred_violation", "max"),
    frac_actual_below_mtd=("actual_below", "mean"),
    mean_actual_below_mtd=("actual_violation", "mean"),
    max_actual_below_mtd=("actual_violation", "max"),
)

# %%
progress_cols = [
    "forecast_origin_date",
    "target_month",
    "weeks_actualized",
    "weeks_in_month",
]
origin_progress = (
    forecasts_df[progress_cols]
    .drop_duplicates()
    .rename(columns={"target_month": "predicted_fiscal_year_month"})
)
metrics = metrics.merge(
    origin_progress, on=["forecast_origin_date", "predicted_fiscal_year_month"]
)

# %%
# prep composed config for sidecar
# TODO: When we refac to use an input config, this becomes unnecessary
composed_cfg = {
    "model": {
        "estimator": "LGBMRegressor",
        "label": "lightgbm_direct",
        "framing": "direct_month",
        "hyperparameters": HYPERPARAMS,
        "features": list(MODELING_SCHEMA.feature_cols),
        "target": MODELING_SCHEMA.target_col,
        "target_handling": "C1 direct total: predict the full-month total on raw y, no clipping (vs C2 residual-to-baseline).",
    },
    "evaluation": {
        "target_months": TARGET_MONTHS,
        "fold_rule": "expanding month-block; split by target_month",
        "benchmark_sidecar_uri": BENCHMARK_SIDECAR_URI,
    },
    "data": {
        "freq": FREQ,
        "period_col": "fiscal_year_month",
        "id_col": "unique_id",
        "target_col": "y",
        "completeness_trim": "incomplete (series, month) pairs trimmed from the panel before feature-building (§5.0); the time_series_snapshot is the trimmed panel.",
    },
}

# %%
# ===============================================
# Gates to check contents before writing sidecar
# ===============================================

# %%
# Gate: tier must be categorical AND carry the full ordered category list, since
# leaderboard.py reads dtype.categories as the complete tier set
assert_tier_categorical(monthly_series)

# %%
# Setup for gate: load the benchmark monthly_series, label horizon, keep horizon_1 in TARGET_MONTHS
bench_monthly_series = pd.read_parquet(f"{BENCHMARK_SIDECAR_URI}monthly_series.parquet")

bench_monthly_series["horizon"] = label_horizon(bench_monthly_series, calendar_df)

mask = (bench_monthly_series["horizon"] == "horizon_1") & (
    bench_monthly_series["predicted_fiscal_year_month"].isin(TARGET_MONTHS)
)
benchmark_h1 = bench_monthly_series[mask].reset_index(drop=True)

# %%
# Gate: challenger and benchmark must cover identical keys and agree on the values
# both derive from the same panel and calendar. The expected counts are derived from
# the spine, panel and realized actuals — never from a sidecar — so they survive a
# change to TARGET_MONTHS or the series count without an edit here.
sidecar_counts = assert_benchmark_key_parity(
    monthly_series,
    benchmark_h1,
    expected_counts=expected_sidecar_counts(
        origin_spine, ts_df, actual_monthly_df, TARGET_MONTHS
    ),
)
print(
    f"target months: {sidecar_counts['target_months']}\n"
    f"origins: {sidecar_counts['origins']}\n"
    f"series: {sidecar_counts['series']}\n"
    f"horizon_1 events: {sidecar_counts['events']}"
)

# %%
# Write the sidecar (only reached if every gate above passed)
# composed config (hand-built dict -> YAML)
save_config(composed_cfg, f"{sidecar_uri}composed_config.yaml")

# lineage / run metadata w/ same helpers the benchmark uses
run_metadata = {
    "git_hash": get_git_hash(),
    "uv_lock_info": get_uv_lock_info(),
    "ts_data_uri": timeseries_uri,
}
write_text_to_gcs(json.dumps(run_metadata, indent=2), f"{sidecar_uri}run_metadata.json")

# input snapshots (self-contained debugging)
calendar_df.to_parquet(f"{sidecar_uri}fiscal_calendar.parquet", index=False)
ts_df.to_parquet(f"{sidecar_uri}time_series_snapshot.parquet", index=False)

# output tables
# leaderboard reads monthly_series/metrics/composed_config/fiscal_calendar
# per_series_mtd is unique to this script
monthly_series.to_parquet(f"{sidecar_uri}monthly_series.parquet", index=False)
metrics.to_parquet(f"{sidecar_uri}metrics.parquet", index=False)
per_series_mtd.to_parquet(f"{sidecar_uri}per_series_mtd.parquet", index=False)

print(f"Sidecar written: {sidecar_uri}")

# %%
join_keys = ["forecast_origin_date", "predicted_fiscal_year_month"]
ch = monthly_series.merge(origin_progress, on=join_keys)
bm = benchmark_h1.merge(origin_progress, on=join_keys)

rows = []
for _, wim, wa in (
    ch[["weeks_in_month", "weeks_actualized"]].drop_duplicates().itertuples()
):
    ch_cohort = ch[(ch["weeks_in_month"] == wim) & (ch["weeks_actualized"] == wa)]
    bm_cohort = bm[(bm["weeks_in_month"] == wim) & (bm["weeks_actualized"] == wa)]
    rows.append(
        {
            "weeks_in_month": wim,
            "weeks_actualized": wa,
            "n_events": len(ch_cohort),
            "wrmae_vs_benchmark": compute_wrmae_pooled(ch_cohort, bm_cohort),
        }
    )

progress_skill = (
    pd.DataFrame(rows)
    .sort_values(["weeks_in_month", "weeks_actualized"])
    .reset_index(drop=True)
)
progress_skill


# %%
# below-MTD: where does the model predict below already-booked MTD?
pred_below_cols = [
    "forecast_origin_date",
    "predicted_fiscal_year_month",
    "weeks_actualized",
    "weeks_in_month",
    "frac_pred_below_mtd",
    "mean_pred_below_mtd_violation",
    "max_pred_below_mtd_violation",
]
below_mtd_summary = metrics.loc[
    metrics["frac_pred_below_mtd"] > 0, pred_below_cols
].reset_index(drop=True)

print(
    f"{(metrics['frac_pred_below_mtd'] > 0).sum()} of {len(metrics)} origins predict below MTD "
    f"| worst violation = {metrics['max_pred_below_mtd_violation'].max():.0f} "
    f"| actual-side base rate (max frac_actual_below_mtd) = {metrics['frac_actual_below_mtd'].max()}"
)
below_mtd_summary

# %%
metrics

# %% [markdown]
# Effective-sample caveat. This eval is 6 target months / 26 origins / 1,768 horizon_1 events — but not 1,768 independent observations. Within a target month, all origins share one fitted model and one final actual, and there are only 6 months, so the effective independent sample is small. Adequate for a plumbing smoke test, not high-confidence model selection — and because the feature set is deliberately minimal (§5.4), a loss to the benchmark here is weak evidence against Framing C, not a verdict. Read trends across cohorts, not any single thin cell.
