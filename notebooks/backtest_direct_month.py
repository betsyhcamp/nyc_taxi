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
# ============ MTD-identity gate (build-time, after build_origin_target_table) ============
sample_pairs = [
    (uid, M) 
    for uid in [4, 12, 104] 
    for M in [202505, 202506, 202507]
]
temp_panel_cal = ts_df.merge(
    calendar_df[["ds", "fiscal_year_month", "fiscal_week_of_month"]],
    on="ds", how="left")
target_by_series_month = (
    origin_target_table[["unique_id", "target_month", "target_month_total_revenue"]]
    .drop_duplicates()
    .set_index(["unique_id", "target_month"])["target_month_total_revenue"]
)
for (uid, M) in sample_pairs:
    N = int(
        calendar_df[['fiscal_year_month', 'weeks_in_month']].drop_duplicates()
        .set_index('fiscal_year_month').loc[M, 'weeks_in_month']
    )

    y_week = (
        temp_panel_cal[(temp_panel_cal["unique_id"] == uid) & (temp_panel_cal["fiscal_year_month"] == M)]
        .set_index("fiscal_week_of_month")["y"]
        .sort_index()
    )
    mtd = (
        origin_target_table[(origin_target_table["unique_id"] == uid) & (origin_target_table["target_month"] == M)]
        .set_index("weeks_actualized")["mtd_revenue"]
        .sort_index()
    )
    target = target_by_series_month[(uid, M)]
    assert mtd[0]==0
    for k in range(1,N):
        assert mtd[k]-mtd[k-1] == y_week[k]
    assert target == mtd[N-1] + y_week[N] 

# %%

# %%
