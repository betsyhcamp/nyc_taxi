# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: nyc_taxi (3.12.9)
#     language: python
#     name: python3
# ---

# %%
import sys
import pandas as pd
import json

from fcstnyctaxi.lib.backtest_results import build_backtest_results, build_cv_results
from fcstnyctaxi.lib.config_utils import merge_configs, save_config
from fcstnyctaxi.lib.cross_validation_utils import sorted_origin_horizon_pairs
from fcstnyctaxi.lib.io import write_text_to_gcs
from fcstnyctaxi.lib.utils import get_project_root_dir, generate_run_id

from tsbricks.backtesting import (
    evaluate_metrics,
    generate_folds, 
    aggregate_backtest
)

from tsbricks.runner import (
    apply_transforms,
    fit_transforms,
    inverse_transforms,
    invoke_model,
)

# may want to remove for prod code since imports are unused directly
import fs  # noqa: F401 - surfaces pkg_resources deprecation warning here, not mid-loop
import tqdm.auto  # noqa: F401 - surfaces TqdmWarning here, not mid-loop

# %%
project_root = get_project_root_dir()
sys.path.insert(0, str(project_root))

base_cfg_path = project_root / "notebooks" / "backtest_configs"/ "base_config.yaml"
model_cfg_path = project_root / "notebooks" / "backtest_configs"/ "model_naive.yaml"

sidecar_dir = generate_run_id()
sidecar_uri=f"gs://nyc-taxi-ehc--modeling/dev/backtests/backtest_weekly/{sidecar_dir}/"

cfg = merge_configs(base_cfg_path, model_cfg_path)

# %%
ts_df = pd.read_parquet(
    "gs://nyc-taxi-ehc--modeling/dev/backtests/data/time_series.parquet"
)

cal_df = pd.read_parquet(cfg.aggregation.calendar_source)


# %%
ts_df.info()
ts_df.head()

# %%
cal_df.info()
cal_df.head()

# %%
#def compute_mtd_actuals(fiscal_week_startdate, ts_df, cal_df):
    

# %%
#target_fiscal_months = 201801	
#temp = ts_df.merge(cal_df, how= 'left', on='ds')

# %%
#temp.head()

# %%
#temp.groupby(['unique_id', 'fiscal_year_month'])

# %%
cv_folds, _ = generate_folds(
    ts_df, 
    cfg.cross_validation,
    cfg.data
)

# %%
print(f"Generated {len(cv_folds)} folds")
for fold_id, splits in cv_folds.items():
    train_end = splits["train"]["ds"].max()
    val_end = splits["val"]["ds"].max()
    print(
        f"  {fold_id}: train ends {train_end}, val ends {val_end} "
        f"(train rows={len(splits['train'])}, val rows={len(splits['val'])})"
    )

# %%
per_fold_metrics = []
per_fold_forecasts: dict[str, pd.DataFrame] = {}

origin_horizon_pairs = sorted_origin_horizon_pairs(
    cfg.cross_validation.origin_horizon_pairs(),
    cfg.data.freq
    )

for fold_idx, (fold_id, splits) in enumerate(cv_folds.items()):
    fold_origin, fold_horizon = origin_horizon_pairs[fold_idx]
    print(f"fold origin: {fold_origin}, fold horizon: {fold_horizon}")

    train, val = splits["train"], splits["val"]

    fitted_transforms, train_t = fit_transforms(train, cfg.transforms or [])
    
    _ = apply_transforms(val, fitted_transforms) # here for consistency
    
    forecast_df, _fitted, _model_obj = invoke_model(
        train_t, cfg.model, fold_horizon
    )

    forecast_original_scale = inverse_transforms(forecast_df, fitted_transforms)
    per_fold_forecasts[fold_id] = forecast_original_scale

    fold_metrics = evaluate_metrics(
        y_true=val,
        y_pred=forecast_original_scale,
        y_train=train,
        metrics_config=cfg.evaluation.native.metrics,
        fold_id=fold_id,
    )
    fold_metrics["fold_origin"] = fold_origin
    fold_metrics["fold_horizon"] = fold_horizon

    per_fold_metrics.append(fold_metrics)

metrics = pd.concat(per_fold_metrics, ignore_index=True)

# %%
cv_results = build_cv_results(
    forecasts_per_fold=per_fold_forecasts,
    train_val_splits_per_fold=cv_folds,
    metrics = metrics,
    origin_horizon_pairs=origin_horizon_pairs
)

# %%
backtest_results = build_backtest_results(
    cv = cv_results,
    config = cfg.model_dump(by_alias=True, exclude_none=True),
    origin_horizon_pairs=origin_horizon_pairs,
    capture_lineage=True
)

# %%
metrics

# %%
# look at a forecast and it's training data as a basic check
fold_id = "fold_1"
train = cv_folds[fold_id]["train"]
forecast = per_fold_forecasts[fold_id]

sample_uid = train["unique_id"].iloc[0]
print(f"unique_id={sample_uid}, fold={fold_id}")

train_sample = train[train["unique_id"] == sample_uid].sort_values("ds")
forecast_sample = forecast[forecast["unique_id"] == sample_uid].sort_values("ds")


# %%
def compute_mtd_actuals(
    train_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    target_fiscal_months: list,
    time_col: str = "ds",
    period_col: str = "fiscal_year_month",
    id_col: str = "unique_id",
    target_col: str = "y",
) -> pd.DataFrame:
    target_cal = (
        calendar_df
        .loc[calendar_df[period_col].isin(target_fiscal_months),[time_col, period_col]]
        .drop_duplicates()
    )

    return (
        train_df 
        .merge(target_cal, on=time_col, how='inner')
        .groupby([id_col, period_col])[target_col]
        .sum()
        .reset_index()
        .rename(columns={target_col:"mtd_actuals"})
    )


# %%
fold_id = "fold_1"
#
fcst_cal = (
    per_fold_forecasts[fold_id]
    .merge(cal_df, how='left', on='ds')
    .loc[:,'fiscal_year_month']
    .unique()
)
train = cv_folds[fold_id]["train"]
train_cal_df = train.merge(cal_df, how='left', on='ds')
mtd_df = (
    train_cal_df 
    .loc[train_cal_df['fiscal_year_month'].isin(fcst_months), :]
    .groupby(['unique_id', 'fiscal_year_month'])['y']
    .sum()
    .reset_index(drop=False)
)


# %%
def combine_monthly_forecast(
    mtd_actuals_df : pd.DataFrame,
    predicted_remaining_df : pd.DataFrame,
    period_col: str = "fiscal_year_month",
    forecast_col: str = "ypred",
    mtd_actuals_col: str = "mtd_actuals",
    id_col: str = "unique_id"
):
    merged_df =(
        mtd_actuals_df[[id_col, period_col, mtd_actuals_col]]
        .merge(
            predicted_remaining_df[[id_col, period_col, forecast_col]], 
            on=[id_col, period_col], 
            how="outer"
        )
        .fillna(0)
    )
    
    merged_df = merged_df.assign(
        monthly_forecast=merged_df[mtd_actuals_col] + merged_df[forecast_col]
    )
    
    return merge_df[[id_col, period_col, "monthly_forecast"]]


# %%
train_sample.tail(10)

# %%
forecast_sample

# %%
aggregated_results = aggregate_backtest(
    results=backtest_results,
    aggregation_config= cfg.aggregation,
    evaluation_level_config=cfg.evaluation.aggregated,
    calendar_df=cal_df
)

# %%
agg_fold_0 = aggregated_results.cv_forecasts["fold_0"]
agg_fold_0

# %%
agg_fold_0[agg_fold_0["unique_id"]==4].sort_values(by="fiscal_year_month")

# %%
# create sidecar contents for this run
save_config(cfg, f"{sidecar_uri}composed_config.yaml")

# run_metadata.json contents
run_metadata = {
    "git_hash": backtest_results.git_hash,
    "uv_lock_info": backtest_results.uv_lock_info,
    "ts_data_uri": "gs://nyc-taxi-ehc--modeling/dev/backtests/data/time_series.parquet"
}

write_text_to_gcs(json.dumps(run_metadata, indent=2), f"{sidecar_uri}run_metadata.json")

# metrics
backtest_results.cv.metrics.to_parquet(f"{sidecar_uri}metrics.parquet", index=False)
aggregated_results.cv_metrics.to_parquet(
    f"{sidecar_uri}aggregated_metrics.parquet", 
    index=False
)

print(f"Sidecar written: {sidecar_uri}")
