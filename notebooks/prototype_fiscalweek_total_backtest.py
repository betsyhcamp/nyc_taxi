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
from google.cloud import bigquery
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tsbricks.blocks.dataio import read_sql, query_to_dataframe
from tsbricks.blocks.plots import plot_seasonal
from fcstnyctaxi.lib.utils import get_project_root_dir


from tsbricks.backtesting import (
    evaluate_metrics,
    generate_folds,
    parse_config,
)

from tsbricks.runner import (
    apply_transforms,
    fit_transforms,
    inverse_transforms,
    invoke_model,
)
from statsforecast import StatsForecast
from statsforecast.models import Naive
from utilsforecast.losses import mae as uf_mae
from utilsforecast.evaluation import evaluate

import warnings
from tqdm import TqdmWarning
warnings.filterwarnings("ignore", category=TqdmWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="fs")

# %%
project = "nyc-taxi-ehc"
location = "us-central1"
input_sql_filename = "initial_daily_taxi_revenue.sql"

project_root = get_project_root_dir()

queries_dir = project_root / "queries"
raw_data_dir = project_root / "data"/ "raw"

# %%
sys.path.insert(0, str(project_root))

# %%
sql_str = read_sql(queries_dir / input_sql_filename)

# %%
client = bigquery.Client(project=project, location=location)
df, *_ = query_to_dataframe(sql_str, client=client)

# %%
df.info()
df.head()

# %%
dtype_map = {
    "pickup_taxi_zone_id": "int64",
    "day_of_fiscal_month": "int64",
    "fiscal_week": "int64",
    "fiscal_month": "int64",
    "fiscal_year": "int64",
    "fiscal_year_month": "int64",
    "day_of_week": "int64",
    "day_of_week_name": "object",
    "is_weekend": "bool",
    "is_holiday": "bool",
    "holiday_name": "object",
    "is_daylight_savings": "bool",
    "fiscal_year_week": "int64",
    "number_ride_pickups": "int64",
}

df = df.astype(dtype_map)

df["pickup_date"] = pd.to_datetime(df["pickup_date"])

# %%
df.loc[df['day_of_fiscal_month']==1, ["pickup_date", "day_of_fiscal_month",	"fiscal_year_week",	"fiscal_week", "fiscal_month","fiscal_year","fiscal_year_month","day_of_week",	"day_of_week_name"]].drop_duplicates().tail(15)

# %%
# need to specify forecast origins for cross-val()(    )

# %%
df['pickup_taxi_zone_id'].nunique()

# %%
df['pickup_date'] = pd.to_datetime(df['pickup_date'])

# %%
total_daily_df = (
    df.groupby('pickup_date')['number_ride_pickups']
    .sum()
    .reset_index(drop=False)
)

# %%
zone_daily_df = (
    df.groupby(['pickup_date', 'pickup_taxi_zone_id'])['number_ride_pickups']
    .sum()
    .reset_index(drop=False)
)

# %%
# visualize entire time series
fig, ax = plt.subplots(figsize=(12, 4)) 
ax.plot(total_daily_df['pickup_date'],  total_daily_df['number_ride_pickups'] )

ax.set_title("Daily Yellow Taxi Pickups in Manhattan")
ax.set_xlabel("Date")
ax.set_ylabel("Number of Pickups")
ax.grid(alpha=0.2)
fig.autofmt_xdate()

plt.tight_layout()

plt.show()

# %%
#for year in zone_daily_df["pickup_date"].dt.year.unique():
for year in [2022, 2023, 2024, 2025]:
    temp_df = zone_daily_df.loc[zone_daily_df["pickup_date"].dt.year==year, :].copy()
    temp_df['number_ride_pickups'] = temp_df['number_ride_pickups'].astype(float)
    min_ride_count=temp_df["number_ride_pickups"].min()
    max_ride_count=temp_df["number_ride_pickups"].max()
    temp_df["pickup_date"] = temp_df["pickup_date"].dt.date
    temp_pivot_df = temp_df.pivot_table(index="pickup_taxi_zone_id", columns="pickup_date", values="number_ride_pickups")
    plt.figure(figsize=(16, 16))
    sns.heatmap(temp_pivot_df, cmap='viridis', vmin=min_ride_count, vmax = max_ride_count)
    plt.title(f'Count taxi ride over year {year}')
    plt.xlabel('date')
    plt.ylabel('Pickup taxi zone ID')
    plt.show()

# %%
total_fiscalweek_df = (
    df.groupby(['fiscal_year_week', 'fiscal_year'])['number_ride_pickups']
    .sum()
    .reset_index(drop=False)
    .sort_values('fiscal_year_week')
    .reset_index(drop=True)
)
total_fiscalweek_df['week_index'] = range(len(total_fiscalweek_df))

# %%
prep_total_fiscal_week_df = total_fiscalweek_df.copy()

# %%
prep_total_fiscal_week_df = prep_total_fiscal_week_df.rename(
    columns={
        'fiscal_year_week': 'ds',
        'number_ride_pickups':'y'
        }
    ).drop(columns=['fiscal_year', 'week_index'])

# %%
prep_total_fiscal_week_df['unique_id'] = 'manhatten_total'

# %%
prep_total_fiscal_week_df =(
    prep_total_fiscal_week_df[['unique_id', 'ds', 'y']]
    .sort_values(by=['unique_id','ds'])
    .reset_index(drop=True)
)

# %%
prep_total_fiscal_week_df.info()

# %%
# visualize entire time series
fig, ax = plt.subplots(figsize=(12, 4)) 
#ax.plot(total_fiscalweek_df['fiscal_year_week'],  total_fiscalweek_df['number_ride_pickups'] )
ax.plot(
    total_fiscalweek_df["week_index"],
    total_fiscalweek_df["number_ride_pickups"]
)
ax.set_title("Weekly Yellow Taxi Pickups in Manhattan")
ax.set_xlabel("Fiscal Week")
ax.set_ylabel("Number of Pickups")
ax.grid(alpha=0.2)

# Show every 13th fiscal week label, roughly quarterly
tick_step = 13
tick_positions = total_fiscalweek_df["week_index"][::tick_step]
tick_labels = total_fiscalweek_df["fiscal_year_week"][::tick_step]

ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, rotation=90, ha="center")
#ax.set_xlim(total_fiscalweek_df["week_index"].min(), 
#            total_fiscalweek_df["week_index"].max())
plt.tight_layout()

plt.show()

# %%
plot_seasonal(
    df=total_fiscalweek_df,
    time_col='fiscal_year_week',
    value_col="number_ride_pickups",
    backend= "matplotlib",
    width= 800,
    height= 450,
    palette = "viridis",
    alpha = 0.8,
    season_col='fiscal_year',
)

# %%
plot_seasonal(
    df=total_fiscalweek_df.loc[total_fiscalweek_df['fiscal_year']>=2022, :],
    time_col='fiscal_year_week',
    value_col="number_ride_pickups",
    backend= "plotly",
    width= 800,
    height= 450,
    palette = "viridis",
    alpha = 0.8,
    season_col='fiscal_year',
)

# %%
zone_fiscalweek_df = (
    df.groupby(['fiscal_year_week','fiscal_year','pickup_taxi_zone_id'])['number_ride_pickups']
    .sum()
    .reset_index(drop=False)
)

# %%
zone_id_list = [169, 234]

# %%
#for zone in zone_id_list:
#    # visualize entire time series
#    fig, ax = plt.subplots(figsize=(12, 4)) 
#    #ax.plot(total_fiscalweek_df['fiscal_year_week'],  total_fiscalweek_df['number_ride_pickups'] )
#    ax.plot(
#        total_fiscalweek_df["week_index"],
#        total_fiscalweek_df["number_ride_pickups"]
#    )
#    ax.set_title("Weekly Yellow Taxi Pickups in Manhattan")
#    ax.set_xlabel("Fiscal Week")
#    ax.set_ylabel("Number of Pickups")
#    ax.grid(alpha=0.2)
#    
#    # Show every 13th fiscal week label, roughly quarterly
#    tick_step = 13
#    tick_positions = total_fiscalweek_df["week_index"][::tick_step]
#    tick_labels = total_fiscalweek_df["fiscal_year_week"][::tick_step]
#    
#    ax.set_xticks(tick_positions)
#    ax.set_xticklabels(tick_labels, rotation=90, ha="center")
#    #ax.set_xlim(total_fiscalweek_df["week_index"].min(), 
#    #            total_fiscalweek_df["week_index"].max())
#    plt.tight_layout()
#    
#    plt.show()

# %%
naive_cfg_path = project_root / "notebooks" / "backtest_configs"/ "backtest_fiscalweek_naive.yaml"
cfg_naive = parse_config(config_path=str(naive_cfg_path))

cfg_naive.cross_validation

# %%
cv_folds, _ = generate_folds(
    prep_total_fiscal_week_df, 
    cfg_naive.cross_validation,
    cfg_naive.data
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

origin_horizon_pairs = cfg_naive.cross_validation.origin_horizon_pairs()

for fold_idx, (fold_id, splits) in enumerate(cv_folds.items()):
    fold_origin, fold_horizon = origin_horizon_pairs[fold_idx]
    print(f"fold origin: {fold_origin}, fold horizon: {fold_horizon}")
    
    train, val = splits["train"], splits["val"]
    
    fitted_transforms, train_t = fit_transforms(train, cfg_naive.transforms or [])
    
    val_t = apply_transforms(val, fitted_transforms)
    
    forecast_df, _fitted, _model_obj = invoke_model(
        train_t, cfg_naive.model, fold_horizon
    )
    
    forecast_original_scale = inverse_transforms(forecast_df, fitted_transforms)
    per_fold_forecasts[fold_id] = forecast_original_scale
    
    fold_metrics = evaluate_metrics(
        y_true=val,
        y_pred=forecast_original_scale,
        y_train=train,
        metrics_config=cfg_naive.evaluation.native.metrics,
        fold_id=fold_id,
    )
    fold_metrics["fold_origin"] = fold_origin
    fold_metrics["fold_horizon"] = fold_horizon
    
    per_fold_metrics.append(fold_metrics)

metrics_naive = pd.concat(per_fold_metrics, ignore_index=True)

# %%
metrics_naive

# %%
nixtla_records = []

for fold_origin, fold_horizon in origin_horizon_pairs:
    train = prep_total_fiscal_week_df[prep_total_fiscal_week_df["ds"] <= fold_origin].copy()
    actual = prep_total_fiscal_week_df[
        (prep_total_fiscal_week_df["ds"] > fold_origin)
        & (prep_total_fiscal_week_df["ds"] <= fold_origin + fold_horizon)
    ].copy()

    sf = StatsForecast(models=[Naive()], freq=1)
    preds = sf.forecast(df=train, h=fold_horizon)

    if "unique_id" not in preds.columns and preds.index.name == "unique_id":
        preds = preds.reset_index()

    eval_df = actual.merge(preds[["unique_id", "ds", "Naive"]], on=["unique_id", "ds"])

    fold_eval = evaluate(eval_df, metrics=[uf_mae])
    fold_mae = fold_eval["Naive"].iloc[0]

    nixtla_records.append(
        {"fold_origin": fold_origin, "fold_horizon": fold_horizon, "mae": fold_mae}
    )

nixtla_naive_df = pd.DataFrame(nixtla_records)
nixtla_naive_df

# %%
naive_fcst_metrics_df = metrics_naive.merge(nixtla_naive_df, on=['fold_origin',	'fold_horizon'])

# %%
naive_fcst_metrics_df = naive_fcst_metrics_df.rename(columns={
        'mae': 'nixtla_mae',
        'value':'tsbricks_mae'
        }
    ).drop(columns=['scope','grouping_column_name','aggregation', 'metric_name'])

# %%
naive_fcst_metrics_df['package_mae_delta'] = (
    naive_fcst_metrics_df['nixtla_mae']
    - naive_fcst_metrics_df['tsbricks_mae']
)

# %%
naive_fcst_metrics_df

# %%
