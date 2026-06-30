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

from fcstnyctaxi.lib.utils import get_project_root_dir
from fcstnyctaxi.lib.config_utils import merge_configs

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
ts_df = pd.read_parquet(
    "gs://nyc-taxi-ehc--modeling/dev/backtests/data/time_series.parquet"
)

cal_df = pd.read_parquet(
    "gs://nyc-taxi-ehc--modeling/dev/backtests/data/fiscal_calendar.parquet"
)

# %%
ts_df.info()
ts_df.head()

# %%
cal_df.info()
cal_df.head()

# %%
project_root = get_project_root_dir()
sys.path.insert(0, str(project_root))

base_cfg_path = project_root / "notebooks" / "backtest_configs"/ "base_config.yaml"
model_cfg_path = project_root / "notebooks" / "backtest_configs"/ "model_naive.yaml"
cfg = merge_configs(base_cfg_path, model_cfg_path)

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

origin_horizon_pairs = cfg.cross_validation.origin_horizon_pairs()

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
metrics

# %%
# look at a forecast and it's training data as a basic check
fold_id = "fold_0"
train = cv_folds[fold_id]["train"]
forecast = per_fold_forecasts[fold_id]

sample_uid = train["unique_id"].iloc[0]
print(f"unique_id={sample_uid}, fold={fold_id}")

train_sample = train[train["unique_id"] == sample_uid].sort_values("ds")
forecast_sample = forecast[forecast["unique_id"] == sample_uid].sort_values("ds")

# %%
train_sample.tail(10)

# %%
forecast_sample

# %%
