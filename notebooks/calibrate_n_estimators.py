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
import json
import yaml
import fsspec
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tsbricks.backtesting.schema import ModelConfig
from tsbricks.runner import dynamic_import

from fcstnyctaxi.lib.calibration import (
    calibrate_n_estimators,
    most_parsimonious_n_estimators
)
from fcstnyctaxi.lib.monthly_aggregation import compute_actual_monthly_totals
from fcstnyctaxi.lib.period_utils import generate_origins_for_periods
from fcstnyctaxi.lib.io import write_text_to_gcs
from fcstnyctaxi.lib.utils import get_project_root_dir, generate_run_id

# %%
project_root = get_project_root_dir()
sys.path.insert(0, str(project_root))

run_config_path = project_root / "notebooks" / "backtest_configs" / "run_config.yaml"
run_cfg = yaml.safe_load(run_config_path.read_text())
calibration_config_path = project_root / "notebooks" / "backtest_configs" / "calibration_config.yaml"
cal_cfg = yaml.safe_load(calibration_config_path.read_text())

model_cfg_path = project_root / cal_cfg["model"]
model_cfg_dict = yaml.safe_load(model_cfg_path.read_text())["model"]
model_cfg = ModelConfig(**model_cfg_dict)


sidecar_dir = generate_run_id()
sidecar_uri = (
    f"{run_cfg['project']['gcs_bucket']}/dev/backtests/calibrate_n_estimators/{sidecar_dir}/"
)

# %%
timeseries_uri = (
    f"{run_cfg['project']['gcs_bucket']}/{run_cfg['project']['time_series_uri']}"
)
ts_df = pd.read_parquet(timeseries_uri)

calendar_uri = (
    f"{run_cfg['project']['gcs_bucket']}/{run_cfg['project']['fiscal_calendar_uri']}"
)
calendar_df = pd.read_parquet(calendar_uri)

# %%
calendar_df.head()

# %%
# Expand monthly calibration_periods.start_months into the exact weekly
# {origin, horizon} list
calibration_periods = cal_cfg["calibration_periods"]
calibration_origin_pairs = generate_origins_for_periods(
    start_months=calibration_periods["start_months"],
    forecast_horizon_months=calibration_periods["forecast_horizon_months"],
    calendar_df=calendar_df,
    calendar_time_col="ds",
)
print(f"Generated {len(calibration_origin_pairs)} calibration origins")
calibration_origin_pairs

# %%
# Leakage gate 
# Get forecast horizon fiscal months of calibration origins. Require no overlap
# between horizons in calibration w/ what is used in read backtest cross-evaluation
month_by_ds = calendar_df.set_index("ds")["fiscal_year_month"]
calibration_target_months = {
    month_by_ds[week]
    for pair in calibration_origin_pairs
    for week in calendar_df.loc[
        calendar_df["ds"] > pd.Timestamp(pair["origin"]), "ds"
    ].head(pair["horizon"])
}

# Derive calibration boundary by reading the backtest's own start_months, so
# leakage gate can't become stale when the backtest window is modified 
backtest_cfg_path = project_root / run_cfg["configs"]["backtest_config"]
backtest_cfg = yaml.safe_load(backtest_cfg_path.read_text())
backtest_start_months = backtest_cfg["evaluation_periods"]["start_months"]

# Raise exception if leakage gate doesn't pass
earliest_backtest_target_month = min(backtest_start_months)
latest_calibration_target_month = max(calibration_target_months)
if latest_calibration_target_month >= earliest_backtest_target_month:
    raise ValueError(
        "Leakage gate failed: latest calibration target month "
        f"{latest_calibration_target_month} is not strictly before the earliest " 
        f"backtest target month {earliest_backtest_target_month}"
)

# %%
grid_cfg = cal_cfg["n_estimators_grid"]
n_estimators_grid = list(
    range(
        grid_cfg["step"], # grid start
        grid_cfg["ceiling"]+1, # grid end inclusive of ceiling
        grid_cfg["step"] # grid step size
    )
)

set_truncation_iteration = dynamic_import(cal_cfg["truncation_adapter"])

# %%
ts_df.head()

# %%
# actuals: ground-truth monthly totals, precomputed once (parallels evaluate_model)
#actual_monthly_df = compute_actual_monthly_totals(ts_df, calendar_df)

# SHAPE MISMATCH to fix: generate_origins_for_periods returned list-of-dicts
# ({"origin":..., "horizon":...}), but calibrate_n_estimators wants
# Sequence[tuple[origin, horizon]]. Convert:
#calibration_origins = [(p["origin"], p["horizon"]) for p in calibration_origin_pairs]

#scores_df = calibrate_n_estimators(
#    ts_df=ts_df,
#    calendar_df=calendar_df,
#    actual_monthly_df=actual_monthly_df,
#    model_config=model_cfg,
#    n_estimators_grid=n_estimators_grid,
#    calibration_origins=calibration_origins,
#    set_truncation_iteration=set_truncation_iteration,
#)
#scores_df            # eyeball: columns origin, n_estimators, horizon, score
#

# %%
data = backtest_cfg["data"]
actual_monthly_df = compute_actual_monthly_totals(
    ts_df = ts_df, 
    calendar_df = calendar_df, 
    period_col = "fiscal_year_month", 
    time_col = data["date_col"], 
    id_col = data["id_col"], 
    target_col = data["target_col"]
)

# %%
cal_cfg

# %%
calibration_origins = [
    (pair["origin"], pair["horizon"]) for pair in calibration_origin_pairs
]

# %%
scores_df = calibrate_n_estimators(
    ts_df=ts_df,
    calendar_df=calendar_df,
    actual_monthly_df=actual_monthly_df,
    model_config=model_cfg,
    n_estimators_grid=n_estimators_grid,
    calibration_origins=calibration_origins,
    set_truncation_iteration=set_truncation_iteration,
)

# %%
scores_df

# %%
scores_df.groupby("horizon")["score"].describe(),

# %%
one_origin = scores_df["origin"].iloc[0]          # or hand-pick a specific origin
curve = (
    scores_df[scores_df["origin"] == one_origin]
    .pivot(index="n_estimators", columns="horizon", values="score")
    .sort_index()
)
curve 

# %%
rows = []
for (origin, horizon), grp in scores_df.groupby(["origin", "horizon"]):
    curve = grp.set_index("n_estimators")["score"].sort_index()   # index = n_estimators
    recommended_n_estimators = most_parsimonious_n_estimators(curve) # defaults: window=3, epsilon=0.01
    rows.append(
        {
            "origin": origin,
            "horizon": horizon,
            "recommended_n_estimators": recommended_n_estimators
        }
    )
recommended_n_estimators_df = pd.DataFrame(rows)


# %%
tail_fraction = 0.10
grid = sorted(n_estimators_grid)
tail_threshold = tail_threshold = grid[-1] - tail_fraction * (grid[-1] - grid[0])
recommended_n_estimators_df["inconclusive"] = (
    recommended_n_estimators_df["recommended_n_estimators"] >= tail_threshold
)

# %%
report = (
    recommended_n_estimators_df
    .groupby("horizon")["inconclusive"]
    .agg(n_flagged="sum", n_origins="count")
)
# print "horizon_2: 5/9 inconclusive -> widen grid and re-run" only where n_flagged > 0

recommended_n_estimators_df.groupby("horizon")["recommended_n_estimators"].describe()
# tight cluster -> trustworthy; wide -> origin-sensitive

