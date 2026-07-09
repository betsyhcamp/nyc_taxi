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
import yaml

from fcstnyctaxi.lib.backtest_results import build_backtest_results, build_cv_results
from fcstnyctaxi.lib.config_utils import merge_configs, save_config
from fcstnyctaxi.lib.cross_validation_utils import sorted_origin_horizon_pairs
from fcstnyctaxi.lib.io import write_text_to_gcs
from fcstnyctaxi.lib.period_utils import (
    assign_tiers,
    compute_series_weights,
    generate_origins_for_periods
)
from fcstnyctaxi.lib.utils import get_project_root_dir, generate_run_id

from tsbricks.backtesting import (
    evaluate_metrics,
    generate_folds
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

run_config_path = project_root / "notebooks" / "backtest_configs" / "run_config.yaml"
run_cfg = yaml.safe_load(run_config_path.read_text())


backtest_cfg_path = project_root / "notebooks" / run_cfg["configs"]["backtest_config"]
model_cfg_path = project_root / "notebooks" / run_cfg["configs"]["model"]

sidecar_dir = generate_run_id()
sidecar_uri = (
    f"{run_cfg['project']['gcs_bucket']}/dev/backtests/backtest_weekly/{sidecar_dir}/"
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
raw_backtest_cfg = yaml.safe_load(backtest_cfg_path.read_text())
eval_periods = raw_backtest_cfg["evaluation_periods"]


# %%
origin_pairs = generate_origins_for_periods(
    start_months=eval_periods["start_months"],
    forecast_horizon_months=eval_periods["forecast_horizon_months"],
    calendar_df=calendar_df,
    calendar_time_col="ds"
)
print(f"Generated {len(origin_pairs)} forecast origins")

# %%
global_horizon = (raw_backtest_cfg.get("cross_validation") or {}).get("horizon")
if global_horizon:
    forecast_origins = [p["origin"] for p in origin_pairs]
else:
    forecast_origins = origin_pairs

# %%
runtime_overrides = {
    "aggregation": {"calendar_source": calendar_uri},
    "cross_validation": {"forecast_origins": forecast_origins},
}
cfg = merge_configs(backtest_cfg_path, model_cfg_path, runtime_overrides)

# %%
actual_monthly_df = (
    ts_df
    .merge(calendar_df[["ds", cfg.aggregation.period_col]].drop_duplicates(), on="ds", how='left')
    .groupby([cfg.aggregation.period_col, "unique_id"])["y"]
    .sum()
    .reset_index()
    .rename(columns={"y":"actual_monthly_total"})
)

# %%
print(yaml.dump(cfg.model_dump(by_alias=True, exclude_none=True), default_flow_style=False))


# %%
def compute_mtd_actuals(
    train_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    target_fiscal_months: list,
    period_col: str = "fiscal_year_month",
    time_col: str = "ds",
    id_col: str = "unique_id",
    target_col: str = "y",
) -> pd.DataFrame:
    """Sum observed weekly actuals within target fiscal months per (id_col, period_col).

    Args:
        train_df: Per-fold observed data (weeks up to the fold origin).
        calendar_df: Maps time_col (ds) to period_col (fiscal_year_month).
        target_fiscal_months: Fiscal months covered by the forecast horizon.
        period_col: Fiscal period column. Default fiscal_year_month.
        time_col: Join key between train_df and calendar_df. Default "ds".
        id_col: Series identifier column. Default "unique_id".
        target_col: Target value column to sum. Default "y".

    Returns:
        DataFrame with columns (id_col, period_col, "mtd_actuals").
        Empty when no target-month weeks have been observed yet (e.g. fold_0).
    """
    

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
def combine_monthly_forecast(
    mtd_actuals_df : pd.DataFrame,
    predicted_remaining_df : pd.DataFrame,
    period_col: str = "fiscal_year_month",
    forecast_col: str = "ypred",
    mtd_actuals_col: str = "mtd_actuals",
    id_col: str = "unique_id"
) -> pd.DataFrame:
    """Add MTD actuals and predicted remaining to produce a total monthly forecast.

    Args:
        mtd_actuals_df: Output of compute_mtd_actuals; columns 
            (id_col, period_col, mtd_actuals_col).
        predicted_remaining_df: Aggregated fold forecasts; columns 
            (id_col, period_col, forecast_col).
        period_col: Fiscal period column. Default "fiscal_year_month".
        forecast_col: Predicted remaining column in predicted_remaining_df. Default
            "ypred".
        mtd_actuals_col: MTD actuals column in mtd_actuals_df. Default "mtd_actuals".
        id_col: Series identifier column. Default "unique_id".

    Returns:
        DataFrame with columns (id_col, period_col, "monthly_forecast").
        Outer join with fillna(0) handles forecast origins where MTD actuals are zero.
    """

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
    
    return merged_df[[id_col, period_col, "monthly_forecast"]]


# %%
def evaluate_model(
    cfg,                           # BacktestConfig — varies per Optuna trial
    ts_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    actual_monthly_df: pd.DataFrame,
) -> dict:
    """
    Runs the composable fold loop for one model config.
    Optuna-compatible: data loaded once outside; cfg varies per trial.

    Returns dict with keys:
        backtest_results,
        monthly_series_df  # columns: "forecast_origin_date", 
                           # "predicted_fiscal_year_month","unique_id", "tier", 
                           # "monthly_forecast",
                           # "actual_monthly_total", "series_weight"
    """
    # -- 1. generate folds --------------------
    cv_folds, _ = generate_folds(
        ts_df, 
        cfg.cross_validation,
        cfg.data
    )
    # -- 2. weekly fold loop: fit -> invoke -> inverse -> evaluate ------
    per_fold_metrics = []
    per_fold_forecasts: dict[str, pd.DataFrame] = {}

    origin_horizon_pairs = sorted_origin_horizon_pairs(
        cfg.cross_validation.origin_horizon_pairs(),
        cfg.data.freq
        )
    monthly_series_rows = []
    seen_origins = set() # dedup: forecast_origin_date

    for fold_idx, (fold_id, splits) in enumerate(cv_folds.items()):
        fold_origin, fold_horizon = origin_horizon_pairs[fold_idx]
        print(f"fold origin: {fold_origin}, fold horizon: {fold_horizon}")

        if fold_origin in seen_origins: # deduplicate fcsts w/ same forecast_origin_date
            continue
        seen_origins.add(fold_origin)

        train, val = splits["train"], splits["val"]

        # tier and weight at this fold's origin
        tier_df = assign_tiers(train, fold_origin, calendar_df)
        weight_df = compute_series_weights(train, fold_origin, calendar_df)
        
        fitted_transforms, train_t = fit_transforms(train, cfg.transforms or [])
        
        _ = apply_transforms(val, fitted_transforms) # here for consistency
        
        forecast_df, _fitted, _model_obj = invoke_model(
            train_t, cfg.model, fold_horizon
        )

        forecast_original_scale = inverse_transforms(forecast_df, fitted_transforms)
        per_fold_forecasts[fold_id] = forecast_original_scale

        # predicted remaining per fiscal month which replaces aggregate_backtest output
        predicted_remaining_df = (
            forecast_original_scale
            .merge(
                calendar_df[["ds", cfg.aggregation.period_col]].drop_duplicates(),
                on="ds",
                how="left"
                )
            .groupby(["unique_id", cfg.aggregation.period_col])["ypred"].sum()
            .reset_index()
        )
        
        target_fiscal_months = (
            predicted_remaining_df[cfg.aggregation.period_col].unique().tolist()
        )

        # MTD + combine + merge produces full artifact rows for this fold
        mtd_actuals_df = compute_mtd_actuals(
            train,
            calendar_df,
            target_fiscal_months,
            period_col=cfg.aggregation.period_col
            )
        monthly_forecast_total_df = combine_monthly_forecast(
            mtd_actuals_df,
            predicted_remaining_df,
            period_col=cfg.aggregation.period_col
        )
        
        fold_rows = (
            monthly_forecast_total_df
            .merge(actual_monthly_df, on=["unique_id", cfg.aggregation.period_col])
            .merge(tier_df,   on="unique_id")
            .merge(weight_df, on="unique_id")
            .assign(forecast_origin_date=fold_origin)
            .rename(columns={
                cfg.aggregation.period_col: "predicted_fiscal_year_month",
            })
            [["forecast_origin_date", "predicted_fiscal_year_month",
              "unique_id", "tier", "monthly_forecast",
              "actual_monthly_total", "series_weight"]]
        )
        monthly_series_rows.append(fold_rows)
        
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
    monthly_series_df = pd.concat(monthly_series_rows, ignore_index=True)
    
    # -- 3. build weekly results -----------------------------
    cv_results = build_cv_results(
        forecasts_per_fold=per_fold_forecasts,
        train_val_splits_per_fold=cv_folds,
        metrics = metrics,
        origin_horizon_pairs=origin_horizon_pairs
    )
    backtest_results = build_backtest_results(
        cv = cv_results,
        config = cfg.model_dump(by_alias=True, exclude_none=True),
        origin_horizon_pairs=origin_horizon_pairs,
        capture_lineage=True
    )


    return {
        "backtest_results": backtest_results,
        "monthly_series_df": monthly_series_df,
    }

# %%
result = evaluate_model(cfg, ts_df, calendar_df, actual_monthly_df)
backtest_results   = result["backtest_results"]
monthly_series_df  = result["monthly_series_df"]

# %%
# extract the raw forecasts generated during cross-validation
cv_forecasts_df = pd.concat(
    [df.assign(fold_id=fold_id)
     for fold_id, df in backtest_results.cv.forecasts_per_fold.items()],
    ignore_index=True
)

# %%
# -- create sidecar contents for this run -------------------

# write composed config
save_config(cfg, f"{sidecar_uri}composed_config.yaml")

# create and write run_metadata.json contents
run_metadata = {
    "git_hash": backtest_results.git_hash,
    "uv_lock_info": backtest_results.uv_lock_info,
    "ts_data_uri": timeseries_uri,
}
write_text_to_gcs(json.dumps(run_metadata, indent=2), f"{sidecar_uri}run_metadata.json")

# input data snapshots for self-contained debugging 
calendar_df.to_parquet(f"{sidecar_uri}fiscal_calendar.parquet", index=False)
ts_df.to_parquet(f"{sidecar_uri}time_series_snapshot.parquet", index=False)

# write output results data to sidecar in GCS
cv_forecasts_df.to_parquet(f"{sidecar_uri}raw_cv_forecasts.parquet", index=False)
backtest_results.cv.metrics.to_parquet(f"{sidecar_uri}metrics.parquet", index=False)
monthly_series_df.to_parquet(f"{sidecar_uri}monthly_series.parquet", index=False)

print(f"Sidecar written: {sidecar_uri}")
