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
import functools
import yaml
import fsspec
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tsbricks.backtesting.schema import ModelConfig
from tsbricks.runner._utils import dynamic_import # private tsbricks module; see tsbricks_improvements.md 

from fcstnyctaxi.lib.calibration import calibrate_n_estimators
from fcstnyctaxi.lib.model_invocation import invoke_predict
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
cal_cfg["n_estimators_grid"]


# %%
n_estimators_grid = list(
    range(
        cal_cfg["n_estimators_grid"]["step"], 
        cfg["n_estimators_grid"]["ceiling"]+1, 
        cal_cfg["n_estimators_grid"]["step"]
    )
)

# %%
