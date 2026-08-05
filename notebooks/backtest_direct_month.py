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
