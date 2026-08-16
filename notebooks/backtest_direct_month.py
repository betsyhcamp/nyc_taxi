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

import numpy as np
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
    attach_workday_progress,
    build_weekly_features,
    enumerate_origins,
    trim_incomplete_series_months,
)
from fcstnyctaxi.lib.period_utils import (
    assign_tiers,
    compute_series_weights,
    derive_horizon_label,
)
from fcstnyctaxi.lib.utils import generate_run_id, get_project_root_dir

# %%
project_root = get_project_root_dir()
sys.path.insert(0, str(project_root))

run_config_path = project_root/ "notebooks" / "backtest_configs" / "run_config.yaml"
run_cfg = yaml.safe_load(run_config_path.read_text())

bucket = run_cfg["project"]["gcs_bucket"]
timeseries_uri = f"{bucket}/{run_cfg['project']['time_series_uri']}"
calendar_uri = f"{bucket}/{run_cfg['project']['fiscal_calendar_uri']}"

ts_df = pd.read_parquet(timeseries_uri)
calendar_df = pd.read_parquet(calendar_uri)

# %%
TARGET_MONTHS = [202504, 202505, 202506, 202507, 202508, 202509]

# TODO: refactor to remove hardcoding &, instead, use parameter file for these items
# Hardcoded for now; registry lookup (resolve_sidecar_uri) is deferred
BENCHMARK_SIDECAR_URI = (
    "gs://nyc-taxi-ehc--modeling/dev/backtests/backtest_weekly/"
    "20260803T230103657210Z/"
)

FREQ = "W-SUN"
MLF_LAGS = [1]
MLF_LAG_TRANSFORMS = {1: [RollingMean(window_size=4)]}

# Model-matrix allowlist (fail-closed): nothing is a feature unless named here.
# Custom features (G1/G3) hand-declared (leakage-prone surface). MLForecast
# native names live once in MLF_FEATURES and are drift-gated.
G1_FEATURES = ["mtd_revenue", "workdays_elapsed", "workdays_remaining", "number_workdays"]
MLF_FEATURES = ["lag1", "rolling_mean_lag1_window_size4"]
G3_FEATURES = ["last_completed_month_revenue"]
FEATURE_COLUMNS = G1_FEATURES + MLF_FEATURES + G3_FEATURES

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
    panel_df = ts_df,
    calendar_df = calendar_df,
)
# %%
dropped_pairs

# %%
actual_monthly_df = compute_actual_monthly_totals(
    ts_df=ts_df,
    calendar_df=calendar_df,
    period_col= "fiscal_year_month",
    time_col= "ds",
    id_col= "unique_id",
    target_col= "y"
)

# %%
weekly_features =  build_weekly_features(
    ts_df,
    FREQ,
    lags=MLF_LAGS,
    lag_transforms=MLF_LAG_TRANSFORMS
    )

# drift-guard gate: preprocess must emit exactly the native names MLF_FEATURES declares
emitted = [c for c in weekly_features.columns if c not in {"unique_id", "feature_ds", "y"}]
assert set(emitted) == set(MLF_FEATURES), \
    f"drift: preprocess emitted {sorted(emitted)}, MLF_FEATURES declares {sorted(MLF_FEATURES)}"

# %%
# alignment sanity check, then display `weekly_features`
weekly_features = weekly_features.sort_values(by=["unique_id", "feature_ds"]).reset_index(drop=True)
prior_week_y = weekly_features.groupby("unique_id")["y"].shift(1)
check_rows = ~weekly_features["lag1"].isna()
assert (weekly_features["lag1"]==prior_week_y).loc[check_rows].all()

# %%
weekly_features

# %%
def build_origin_target_table(panel, calendar_df, origin_spine, actual_monthly_df):
    cal_df = (
        calendar_df.copy().sort_values(by=["fiscal_year_month", "fiscal_week_of_month"])
    )

    # per series MTD of y
    panel_cal = (
        panel
        .merge(cal_df[["ds", "fiscal_year_month", "fiscal_week_of_month"]], on="ds", how="left")
        .sort_values(["unique_id", "fiscal_year_month", "fiscal_week_of_month"])
    )
    panel_cal["mtd_revenue"] = panel_cal.groupby(["unique_id", "fiscal_year_month"])["y"].cumsum()
    mtd_lookup = panel_cal.rename(
        columns={"fiscal_year_month":"target_month", "fiscal_week_of_month":"weeks_actualized"}
    )[["unique_id", "target_month", "weeks_actualized", "mtd_revenue"]]

    # --- assemble: start from ACTIVE (series, target_month) pairs
    # actual_monthly_df has one row per (series, month) the series is active in, so an inner
    # join on target_month gives each origin only its active series AND attaches target var
    final_month = actual_monthly_df.rename(
        columns={"fiscal_year_month": "target_month", "actual_monthly_total": "target_month_total_revenue"})
    # origin_spine already carries number_workdays / workdays_elapsed /
    # workdays_remaining from attach_workday_progress, computed at origin grain
    table = origin_spine.merge(final_month, on="target_month", how="inner")

    table = table.merge(mtd_lookup, on=["unique_id", "target_month", "weeks_actualized"], how="left")
    table["mtd_revenue"] = table["mtd_revenue"].fillna(0)

    # last-completed (M-1): NaN for a series' FIRST active month (no prior month)
    months_sorted = sorted(cal_df["fiscal_year_month"].unique())
    prev_month_lookup = pd.Series(months_sorted, index=months_sorted).shift(1)
    table["prev_month"] = table["target_month"].map(prev_month_lookup).astype(int)
    table = table.merge(
        actual_monthly_df.rename(columns={"fiscal_year_month": "prev_month",
                                          "actual_monthly_total": "last_completed_month_revenue"}),
        on=["unique_id", "prev_month"], how="left")

    cols = ["unique_id", "forecast_origin_date", "target_month",
            "mtd_revenue", "workdays_elapsed", "workdays_remaining", "number_workdays",
            "weeks_actualized", "weeks_in_month",
            "last_completed_month_revenue", "target_month_total_revenue"]
    return table[cols].reset_index(drop=True)


# %%
origin_spine = enumerate_origins(calendar_df)
origin_spine = attach_workday_progress(origin_spine, calendar_df)
# %%
origin_spine.tail(30)

# %%
origin_target_table = build_origin_target_table(
    ts_df, 
    calendar_df, 
    origin_spine, 
    actual_monthly_df
)

# %%
# ============ MTD-identity gate (vectorized, over ALL (unique_id, target_month)) ============
# Ground truth: raw weekly y per (series, month, week-of-month), from panel
weekly_y = (
    ts_df.merge(
        calendar_df[["ds", "fiscal_year_month", "fiscal_week_of_month"]],
        on="ds", how="left",
    )
    .rename(columns={
        "fiscal_year_month": "target_month",
        "fiscal_week_of_month": "week_of_month",
        "y": "week_y",
    })
    [["unique_id", "target_month", "week_of_month", "week_y"]]
)

# ---- Gate 2: sign-safe MTD construction ----
# mtd increments within each (series, month); sort first — diff() is order-dependent.
ott = origin_target_table.sort_values(["unique_id", "target_month", "weeks_actualized"])
ott = ott.assign(
    mtd_increment=ott.groupby(["unique_id", "target_month"])["mtd_revenue"].diff()
)
# check: MTD=0 origin has zero month-to-date
bad_mtd0 = ott[(ott["weeks_actualized"] == 0) & (ott["mtd_revenue"] != 0)]
assert bad_mtd0.empty, f"MTD=0 origins with nonzero mtd:\n{bad_mtd0.head()}"

# check: each increment equals that week's raw y (weeks_actualized >= 1; may be negative)
increments = ott[ott["weeks_actualized"] >= 1].merge(
    weekly_y,
    left_on=["unique_id", "target_month", "weeks_actualized"],
    right_on=["unique_id", "target_month", "week_of_month"],
    how="left",
)
gate2_fail = increments[increments["mtd_increment"] != increments["week_y"]]
assert gate2_fail.empty, (
    f"Gate 2 (increment != weekly y) failed for {len(gate2_fail)} rows:\n"
    f"{gate2_fail[['unique_id','target_month','weeks_actualized','mtd_increment','week_y']].head()}"
)

# ---- Gate 1: reconciliation (target == mtd at last origin + final week's y) ----
last_origin = origin_target_table[
    origin_target_table["weeks_actualized"] == origin_target_table["weeks_in_month"] - 1
]
reconciliation = last_origin.merge(
    weekly_y,
    left_on=["unique_id", "target_month", "weeks_in_month"],
    right_on=["unique_id", "target_month", "week_of_month"],
    how="left",
)
reconstructed = reconciliation["mtd_revenue"] + reconciliation["week_y"]
gate1_fail = reconciliation[reconstructed != reconciliation["target_month_total_revenue"]]
assert gate1_fail.empty, (
    f"Gate 1 (reconciliation) failed for {len(gate1_fail)} rows:\n"
    f"{gate1_fail[['unique_id','target_month','mtd_revenue','week_y','target_month_total_revenue']].head()}"
)


# %%
origin_target_table.head()


# %%
def build_modeling_table(origin_target_table, weekly_features):
    keyed_origins = origin_target_table.copy()
    keyed_origins["feature_ds"] = keyed_origins["forecast_origin_date"] + pd.Timedelta(weeks=1)
    modeling = keyed_origins.merge(weekly_features, on=['unique_id', 'feature_ds'],how='left')
    modeling = modeling.drop(columns="y")
    keep = (
        ["unique_id", "forecast_origin_date", "target_month"]
        + FEATURE_COLUMNS
        + ["target_month_total_revenue"]
        + ["weeks_actualized", "weeks_in_month", "feature_ds"]
    )
    return modeling[keep].reset_index(drop=True)


# %%
modeling_table = build_modeling_table(origin_target_table, weekly_features)

# %%
# ====== Gate: join integrity ===========
keyed_origins = origin_target_table.copy()

keyed_origins["feature_ds"] = keyed_origins["forecast_origin_date"] + pd.Timedelta(weeks=1)

assert modeling_table["unique_id"].dtype == weekly_features["unique_id"].dtype
assert modeling_table["feature_ds"].dtype == weekly_features["feature_ds"].dtype

antijoin = keyed_origins.merge(weekly_features[["unique_id", "feature_ds"]], 
                    on = ["unique_id", "feature_ds"], how="left", indicator=True)
assert (antijoin["_merge"] == "left_only").sum()==0

assert len(modeling_table)==len(origin_target_table)
assert list(modeling_table[FEATURE_COLUMNS].columns)==FEATURE_COLUMNS


# %%
modeling_table.head()

# %%
# ====== Gate: mtd identity checking lag ===========
mt = modeling_table.sort_values(by=["unique_id", "target_month", "weeks_actualized"])
mt = mt.assign(
    mtd_increment=mt.groupby(["unique_id", "target_month"])["mtd_revenue"].diff()
)
interior = mt["weeks_actualized"] >=1
assert (mt.loc[interior, "mtd_increment"] == mt.loc[interior, "lag1"]).all()

panel_at_W = ts_df.rename(columns={"ds": "forecast_origin_date", "y": "y_at_W"})

check = modeling_table.merge(
    panel_at_W[["unique_id", "forecast_origin_date", "y_at_W"]],
    on=["unique_id", "forecast_origin_date"],
    how="left",
)
observed = check["y_at_W"].notna()
assert (check.loc[observed, "lag1"]==check.loc[observed, "y_at_W"]).all()

# %%
# ====== Gate: general leakage smoke test (future-scramble) ===========
# Leak-free == an origin's features are invariant to any change in weeks > W.
# Scramble all y after a cutoff, rebuild every y-derived input, and assert every
# origin at/before the cutoff has identical FEATURES. The target is excluded since it
# legitimately includes future weeks. Vectorized over all unique_ids.

# early-in-month (weeks_actualized=1) in a 5-week month leaves several same-month weeks
# in the scrambled region; wa=1 (not 0) keeps W inside the target month so mtd is exercised
cutoff_W = pd.Timestamp("2025-05-25")

def rebuild_all(panel):
    actual_monthly = compute_actual_monthly_totals(
        ts_df=panel, calendar_df=calendar_df,
        period_col="fiscal_year_month", time_col="ds",
        id_col="unique_id", target_col="y",
    )
    features = build_weekly_features(
        panel, FREQ, lags=MLF_LAGS, lag_transforms=MLF_LAG_TRANSFORMS
    )
    ot = build_origin_target_table(panel, calendar_df, origin_spine, actual_monthly)
    return build_modeling_table(ot, features)

panel_scrambled = ts_df.copy()
panel_scrambled["y"] = panel_scrambled["y"].astype(float)
future = panel_scrambled["ds"] > cutoff_W
panel_scrambled.loc[future, "y"] = np.random.default_rng(0).random(int(future.sum())) * 1e6

base = rebuild_all(ts_df)
scrambled = rebuild_all(panel_scrambled)

keys = ["unique_id", "forecast_origin_date"]
past_base = base[base["forecast_origin_date"] <= cutoff_W].sort_values(keys)[FEATURE_COLUMNS].reset_index(drop=True)
past_scr  = scrambled[scrambled["forecast_origin_date"] <= cutoff_W].sort_values(keys)[FEATURE_COLUMNS].reset_index(drop=True)

same = (past_base == past_scr) | (past_base.isna() & past_scr.isna())
assert same.all().all(), "LEAK: future y changed features of an origin at/before the cutoff"



# %%
def run_fold(modeling_table, val_month):
    train = modeling_table[modeling_table["target_month"]< val_month]
    val = modeling_table[modeling_table["target_month"]== val_month]

    # train and val target months must not overlap
    assert set(train["target_month"]).isdisjoint(set(val["target_month"]))

    X_train = train[FEATURE_COLUMNS]
    y_train = train["target_month_total_revenue"]
    model = LGBMRegressor(**HYPERPARAMS).fit(X_train, y_train)

    preds = model.predict(val[FEATURE_COLUMNS])

    keys = ["unique_id", "forecast_origin_date", "target_month"]
    month_progress_cols = ["weeks_actualized", "weeks_in_month"]
    return val[keys+month_progress_cols+["target_month_total_revenue"]].assign(prediction=preds)


# %%
# produce backtest forecasts
forecasts_df = pd.concat(
    [run_fold(modeling_table, val_month) for val_month in TARGET_MONTHS],
    ignore_index=True
)

# %%
len(forecasts_df)

# %%
forecasts_df.head()

# %%
# assemble forecasts w/ tier, weight, added items needed later for scoring backtests
fraction_by_origin = calendar_df.set_index("ds")["origin_month_fraction_elapsed"]

monthly_rows_rename_dict = {
        "prediction":"monthly_forecast",
        "target_month_total_revenue": "actual_monthly_total",
        "target_month": "fiscal_year_month",
    }
monthly_rows_col_keep = ["unique_id", "fiscal_year_month", "monthly_forecast", "actual_monthly_total"]
rows = []
for origin_date, origin_group in forecasts_df.groupby("forecast_origin_date"):
    monthly_rows_df = (
        origin_group
        .rename(columns=monthly_rows_rename_dict)
        [monthly_rows_col_keep]
    )
    tier_df   = assign_tiers(
        train_df = ts_df,
        origin_date = origin_date,
        calendar_df = calendar_df
    )

    weight_df = compute_series_weights(
        train_df = ts_df,
        origin_date=origin_date,
        calendar_df = calendar_df
    )
    origin_group_built = attach_tier_and_weight(
        monthly_rows_df = monthly_rows_df,
        tier_df = tier_df,
        weight_df = weight_df,
        fold_origin=origin_date,
        origin_month_fraction_elapsed=fraction_by_origin[origin_date]
    )
    rows.append(origin_group_built)

monthly_series = pd.concat(rows, ignore_index=True)

# %%
len(monthly_series)

# %%
monthly_series.head()

# %%
# Setup for gate check: origin's fiscal month from base calendar dataframe
origin_fiscal_month_by_ds = calendar_df.set_index("ds")["fiscal_year_month"]
origin_fiscal_month =  monthly_series["forecast_origin_date"].map(
    origin_fiscal_month_by_ds
)

horizon = derive_horizon_label(
    predicted_fiscal_year_month = monthly_series["predicted_fiscal_year_month"],
    origin_fiscal_year_month = origin_fiscal_month,
    origin_month_fraction_elapsed = monthly_series["origin_month_fraction_elapsed"]
)

# Gate: Since we are only predicting current month, all rows==horizon_1 label
assert (horizon == "horizon_1").all(), f"non-horizon_1 rows: {int((horizon != 'horizon_1').sum())}"

# %%
# prep sidecar: per_series_mtd file 
val_months = modeling_table["target_month"].isin(TARGET_MONTHS)

per_series_mtd = (
    modeling_table.loc[val_months, ["unique_id", "forecast_origin_date", "target_month", "mtd_revenue"]]
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
per_series =  monthly_series.merge(per_series_mtd, on = keys)

per_series["pred_below"] = per_series["monthly_forecast"] < per_series["mtd_revenue"]
per_series["actual_below"] = per_series["actual_monthly_total"] < per_series["mtd_revenue"]
per_series["pred_violation"] = (
    per_series["mtd_revenue"] - per_series["monthly_forecast"]).where(per_series["pred_below"])
per_series["actual_violation"] = (
    per_series["mtd_revenue"] - per_series["actual_monthly_total"]).where(per_series["actual_below"])

metrics = (
    per_series
    .groupby(["forecast_origin_date", "predicted_fiscal_year_month"], as_index=False)
    .agg(
        n_series = ("unique_id", "size"),
        frac_pred_below_mtd = ("pred_below", "mean"), # mean of bool = fraction
        mean_pred_below_mtd_violation = ("pred_violation", "mean"), # skips NaNs
        max_pred_below_mtd_violation = ("pred_violation", "max"),
        frac_actual_below_mtd = ("actual_below", "mean"),
        mean_actual_below_mtd = ("actual_violation", "mean"),
        max_actual_below_mtd = ("actual_violation", "max")
    )
)

# %%
progress_cols = [
    "forecast_origin_date", "target_month", "weeks_actualized", "weeks_in_month"
    ]
origin_progress = (
    forecasts_df[progress_cols]
    .drop_duplicates()
    .rename(columns={"target_month":"predicted_fiscal_year_month"})
)
metrics = metrics.merge(
    origin_progress, 
    on=["forecast_origin_date","predicted_fiscal_year_month"]
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
        "features": FEATURE_COLUMNS,
        "target": "target_month_total_revenue",
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
# Gate: Check `tier` is categorical datatype
assert isinstance(monthly_series["tier"].dtype, pd.CategoricalDtype)

# %%
# Setup for gate: load the benchmark monthly_series, label horizon, keep horizon_1 in TARGET_MONTHS
bench_monthly_series = pd.read_parquet(f"{BENCHMARK_SIDECAR_URI}monthly_series.parquet")

origin_fiscal_month_by_ds = calendar_df.set_index("ds")["fiscal_year_month"]
bench_origin_fiscal_month =  bench_monthly_series["forecast_origin_date"].map(
    origin_fiscal_month_by_ds
)
bench_horizon=derive_horizon_label(
    predicted_fiscal_year_month = bench_monthly_series["predicted_fiscal_year_month"],
    origin_fiscal_year_month = bench_origin_fiscal_month,
    origin_month_fraction_elapsed = bench_monthly_series["origin_month_fraction_elapsed"]
    )
bench_monthly_series["horizon"] = bench_horizon

mask = (
    (bench_monthly_series["horizon"]=="horizon_1")
    & (bench_monthly_series["predicted_fiscal_year_month"].isin(TARGET_MONTHS))
)
benchmark_h1 = bench_monthly_series[mask].reset_index(drop=True)

# %%
monthly_series_keys = set(
    zip(
        monthly_series["forecast_origin_date"], 
        monthly_series["predicted_fiscal_year_month"], 
        monthly_series["unique_id"]
        )
    )
bench_keys = set(
    zip(
        benchmark_h1["forecast_origin_date"], 
        benchmark_h1["predicted_fiscal_year_month"], 
        benchmark_h1["unique_id"]
        )
    )
assert monthly_series_keys == bench_keys, "Keys not the same " \
    f"here {monthly_series_keys.symmetric_difference(bench_keys)}"

print(
    f"target months:{monthly_series['predicted_fiscal_year_month'].nunique()} (expect 6)\n"
    f"origins: {monthly_series['forecast_origin_date'].nunique()} (expect 26)\n"
    f"series: {monthly_series['unique_id'].nunique()} (expect 68)\n"
    f"horizon_1 events: {len(monthly_series)} (expect 1768)"
)

# %%
# Run gates to check benchmark:Current work parity
rtol = 1e-9
key_cols = ["forecast_origin_date", "predicted_fiscal_year_month", "unique_id"]

parity = monthly_series.merge(benchmark_h1, on=key_cols, suffixes=("_c", "_bench"))
assert np.allclose(
    parity["actual_monthly_total_c"],
    parity["actual_monthly_total_bench"],
    rtol=rtol
)
assert np.allclose(parity["series_weight_c"], parity["series_weight_bench"], rtol=rtol)
assert (parity["tier_c"] == parity["tier_bench"]).all()
assert (
    parity["origin_month_fraction_elapsed_c"] 
    == parity["origin_month_fraction_elapsed_bench"]
    ).all()

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
for _, wim, wa in ch[["weeks_in_month", "weeks_actualized"]].drop_duplicates().itertuples():
    ch_cohort = ch[(ch["weeks_in_month"] == wim) & (ch["weeks_actualized"] == wa)]
    bm_cohort = bm[(bm["weeks_in_month"] == wim) & (bm["weeks_actualized"] == wa)]
    rows.append({
        "weeks_in_month": wim,
        "weeks_actualized": wa,
        "n_events": len(ch_cohort),
        "wrmae_vs_benchmark": compute_wrmae_pooled(ch_cohort, bm_cohort),
    })

progress_skill = (
    pd.DataFrame(rows)
    .sort_values(["weeks_in_month", "weeks_actualized"])
    .reset_index(drop=True)
)
progress_skill


# %%
# below-MTD: where does the model predict below already-booked MTD?
pred_below_cols = [
    "forecast_origin_date", "predicted_fiscal_year_month",
    "weeks_actualized", "weeks_in_month",
    "frac_pred_below_mtd", "mean_pred_below_mtd_violation", "max_pred_below_mtd_violation",
]
below_mtd_summary = metrics.loc[metrics["frac_pred_below_mtd"] > 0, pred_below_cols].reset_index(drop=True)

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
