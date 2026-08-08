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
import sys
import pandas as pd
import numpy as np

from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean

from fcstnyctaxi.lib.monthly_aggregation import compute_actual_monthly_totals
from fcstnyctaxi.lib.utils import get_project_root_dir, generate_run_id

import yaml

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

# Framing C completeness: trim incomplete (series, month) at the source.
# (C's origins assume all N weeks of a target month exist; Framing A tolerates partial
#  months — so this stays local to C for now, promote upstream only if C wins.)
#labeled       = ts_df ⋈ calendar[ds, fiscal_year_month, fiscal_week_of_month, weeks_in_month]
#weeks_present = labeled.groupby([unique_id, fiscal_year_month])[fiscal_week_of_month].transform("nunique")
#keep_row      = weeks_present == labeled["weeks_in_month"]
#ts_df         = labeled[keep_row][["unique_id", "ds", "y"]]     # back to the original schema

#dropped = labeled[~keep_row]
#print(dropped.groupby([unique_id, fiscal_year_month]).size())   # expect: (104,201805), (105,201801)
# re-check nothing partial survives:
#assert (ts_df ⋈ calendar).groupby([...]).nunique == weeks_in_month  everywhere

labeled = ts_df.merge(
    calendar_df[["ds", "fiscal_year_month", "fiscal_week_of_month", "weeks_in_month"]],
    on="ds", how="left",
)
weeks_present = labeled.groupby(["unique_id", "fiscal_year_month"])["fiscal_week_of_month"].transform("nunique")
keep_row = weeks_present == labeled["weeks_in_month"]
ts_df = labeled.loc[keep_row, ["unique_id","ds","y"]]

dropped = labeled[~keep_row]
print(dropped.groupby(["unique_id", "fiscal_year_month"]).size())

recheck = ts_df.merge(
    calendar_df[["ds", "fiscal_year_month", "fiscal_week_of_month", "weeks_in_month"]],
    on="ds", how="left",
)
weeks_present = recheck.groupby(["unique_id", "fiscal_year_month"])["fiscal_week_of_month"].transform("nunique")
assert (weeks_present == recheck["weeks_in_month"]).all(), "trimmed panel still has a partial (series, month)"

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
def build_weekly_features(panel, freq, *, lags, lag_transforms):
    """Layer 1 feature factory. Feature-defining args only (keyword-only).
    Native MLForecast feature names are kept.
    """
    mlf = MLForecast(models=[], freq=freq, lags=lags, lag_transforms=lag_transforms)
    feats = mlf.preprocess(panel, dropna=False) # unique_id, ds, y, + native feature names
    return feats.rename(columns={"ds": "feature_ds"})


# %%
layer1 =  build_weekly_features(
    ts_df,
    FREQ,
    lags=MLF_LAGS,
    lag_transforms=MLF_LAG_TRANSFORMS
    )

# drift-guard gate: preprocess must emit exactly the native names MLF_FEATURES declares
emitted = [c for c in layer1.columns if c not in {"unique_id", "feature_ds", "y"}]
assert set(emitted) == set(MLF_FEATURES), \
    f"drift: preprocess emitted {sorted(emitted)}, MLF_FEATURES declares {sorted(MLF_FEATURES)}"

# %%
# alignment sanity check, then display `layer1`
layer1 = layer1.sort_values(by=["unique_id", "feature_ds"]).reset_index(drop=True)
prior_week_y = layer1.groupby("unique_id")["y"].shift(1)
check_rows = ~layer1["lag1"].isna()
assert (layer1["lag1"]==prior_week_y).loc[check_rows].all()

# %%
layer1


# %%
def enumerate_origins(calendar_df):
    cal_df=calendar_df.copy().sort_values(by="ds").reset_index(drop=True)
    is_month_end = cal_df["origin_month_fraction_elapsed"]==1
    
    cal_df["target_month"] = np.where(
        is_month_end, 
        cal_df["fiscal_year_month"].shift(-1),
        cal_df["fiscal_year_month"]
    )
    cal_df["weeks_in_month"] = np.where(
        is_month_end,
        cal_df["weeks_in_month"].shift(-1),
        cal_df["weeks_in_month"]
    )
    cal_df["weeks_actualized"] = np.where(
        is_month_end,
        0,
        cal_df["fiscal_week_of_month"]
    )

    cols_keep = [
        "target_month",
        "forecast_origin_date",
        "weeks_actualized",
        "weeks_in_month"
    ]
    first_month = cal_df["fiscal_year_month"].min()
    return (
        cal_df[(cal_df["target_month"].notna()) & (cal_df["target_month"]!=first_month)]
        .rename(columns={"ds":"forecast_origin_date"})
        .astype({"weeks_in_month": int, "target_month": int})
        [cols_keep]
        .reset_index(drop=True)
    )



# %%
def build_origin_target_table(panel, calendar_df, origin_spine, actual_monthly_df):
    cal_df = (
        calendar_df.copy().sort_values(by=["fiscal_year_month", "fiscal_week_of_month"])
    )

    # workday lookups
    number_workdays_by_month = (
        cal_df.groupby("fiscal_year_month")["count_workdays"].sum()
    )
    workdays_elapsed_lookup = (
        cal_df
        .assign(workdays_elapsed=cal_df.groupby("fiscal_year_month")["count_workdays"].cumsum())
        .rename(columns={
            "fiscal_year_month":"target_month",
            "fiscal_week_of_month": "weeks_actualized"
            })
        [["target_month", "weeks_actualized", "workdays_elapsed"]]
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
    
    # previous fiscal month revenue
    months_sorted = sorted(cal_df["fiscal_year_month"].unique())
    prev_month = pd.Series(months_sorted, index=months_sorted).shift(1) # M-> M-1
    
    # --- assemble: start from ACTIVE (series, target_month) pairs
    # actual_monthly_df has one row per (series, month) the series is active in, so an inner
    # join on target_month gives each origin only its active series AND attaches the target.
    final_month = actual_monthly_df.rename(
        columns={"fiscal_year_month": "target_month", "actual_monthly_total": "target_month_total_revenue"})
    table = origin_spine.merge(final_month, on="target_month", how="inner")

    table["number_workdays"] = table["target_month"].map(number_workdays_by_month)
    table = table.merge(workdays_elapsed_lookup, on=["target_month", "weeks_actualized"], how="left")
    table["workdays_elapsed"] = table["workdays_elapsed"].fillna(0)
    table["workdays_remaining"] = table["number_workdays"] - table["workdays_elapsed"]

    table = table.merge(mtd_lookup, on=["unique_id", "target_month", "weeks_actualized"], how="left")
    table["mtd_revenue"] = table["mtd_revenue"].fillna(0)

    # last-completed (M-1): NaN for a series' FIRST active month (no prior month) — a real signal, keep it
    prev_month_lookup = pd.Series(months_sorted, index=months_sorted).shift(1)   # renamed to avoid shadowing
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
