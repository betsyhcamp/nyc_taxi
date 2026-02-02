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
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from datetime import datetime
from pathlib import Path

from coreforecast.scalers import boxcox_lambda

from statsforecast import StatsForecast
from statsforecast.models import AutoETS, AutoTBATS, SeasonalNaive
from fcstnyctaxi.lib.utils import get_project_root_dir

from utilsforecast.plotting import plot_series
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import rmse

# %% [markdown]
# ## Metric Rationale
#
# Top level (Total Manhattan) only has one time series, so scaled error metrics are not helpful.
# Scaled metrics (e.g., WAPE, RMSSE, MASE) are used when comparing multiple time series of differing scales.
# Within non-scaled metrics, can choose those that use the Absolute value or the Squared Error.
#
# For taxi zone time series, choose a scaled metric. And for hierarchical reconciliation,
# we will minimize squared error. So, it makes most sense to choose a forecast error metric
# that also minimizes squared error. Conclusion: for taxi zone time series, use RMSSE -> WRMSSE.
#
# For the top level, I will use a non-scaled metric that minimizes squared error.
# Conclusion: for top-level, use RMSE.
#
# **Evaluation approach:**
# - **Pooled RMSE**: RMSE computed across all CV folds combined
# - **Average RMSE**: RMSE computed per fold, then averaged across 5 folds


# %%
# Constants and input/output locations
PRINT_PRECISION = ".4f"
SAVE_BACKTEST_RESULTS_TO_FILE = False

# input
project_root = get_project_root_dir()
data_dir_path = project_root / "data" / "preprocessed"
filename = "daily_total_timeseries.parquet"

# output
timestamp = datetime.now().strftime("%Y%m%d")
output_dir = Path(f"../data/backtests/{timestamp}_daily_total")
output_dir.mkdir(parents=True, exist_ok=True)

# %%
# Cross-validation setup
horizon = 28  # 28 days (4 weeks) forecast horizon
n_windows = 5  # 5 CV folds
step_size = 28  # non-overlapping folds (rolling forecast origin)
prediction_interval_levels = [60, 80, 95]

print(f"Horizon: {horizon} days")
print(f"Number of CV folds: {n_windows}")
print(f"Step size: {step_size} days")
print(f"Prediction interval levels: {prediction_interval_levels}")

# %%
# Load data
total_daily_df = pd.read_parquet(data_dir_path / filename)
total_daily_df["unique_id"] = "daily_total"

# %%
total_daily_df.info()
total_daily_df.head()

# %%
# max is 10/31/2025 which is a Friday.
print(total_daily_df["pickup_date"].max())
# set max date so we are testing full weeks (just for ease of understanding)
max_date = pd.to_datetime("2025-10-25")

# %%
# Hold out last 28 days (4 weeks) as test set - not used in this script
test_cutoff_date = max_date - pd.Timedelta(days=27)
test_set = total_daily_df[total_daily_df["pickup_date"] >= test_cutoff_date].copy()
train_val_df = total_daily_df[total_daily_df["pickup_date"] < test_cutoff_date].copy()
train_cutoff_date = train_val_df["pickup_date"].max() - pd.Timedelta(
    days=step_size * n_windows
)

print(f"Total observations: {len(total_daily_df)}")
print(f"Train/Val observations: {len(train_val_df)}")
print(f"Test set observations: {len(test_set)}")
print(f"Test cutoff date: {test_cutoff_date}")
print(f"Train cutoff date: {train_cutoff_date}")

# %%
# Visualize entire time series (including test set for context)
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(total_daily_df["pickup_date"], total_daily_df["number_ride_pickups"])
ax.axvline(x=test_cutoff_date, color="red", linestyle="--", label="Test set cutoff")
ax.axvline(
    x=train_cutoff_date,
    color="orange",
    linestyle="--",
    label="Initial train set cutoff",
)

ax.set_title("Daily Yellow Taxi Pickups in Manhattan")
ax.set_xlabel("Date")
ax.set_ylabel("Number of Pickups")
ax.legend()
ax.grid(alpha=0.2)
fig.autofmt_xdate()

plt.tight_layout()
plt.show()

# %%
# Compute optimal Box-Cox lambda on training data only
bc_lambda = boxcox_lambda(
    train_val_df.loc[
        train_val_df["pickup_date"] <= train_cutoff_date, "number_ride_pickups"
    ].values,
    method="loglik",
    season_length=7,
)
bc_lower = bc_lambda - 0.001
bc_upper = bc_lambda + 0.001

print(f"Optimal Box-Cox lambda: {bc_lambda:{PRINT_PRECISION}}")
print(
    f"TBATS Box-Cox bounds: [{bc_lower:{PRINT_PRECISION}}, {bc_upper:{PRINT_PRECISION}}]"
)

# %%
train_val_df.info()
train_val_df.head()

# %%
# Prepare data to have column names expected by StatsForecast
df_long = (
    train_val_df[["unique_id", "pickup_date", "number_ride_pickups"]]
    .rename(columns={"pickup_date": "ds", "number_ride_pickups": "y"})
    .sort_values(by="ds")
    .reset_index(drop=False)
)

# %%
df_long.info()
df_long.head()

# %%
# Define models
models = [
    AutoTBATS(
        season_length=[7, 365],
        use_boxcox=True,
        alias="TBATS_bc",
    ),
    AutoTBATS(
        season_length=[7, 365],
        use_boxcox=False,
        alias="TBATS_no_bc",
    ),
    AutoETS(season_length=7, alias="AutoETS_7"),
    AutoETS(season_length=365, alias="AutoETS_365"),
    SeasonalNaive(season_length=7, alias="SeasonalNaive_7"),
]

alias_name_list = [m.alias for m in models]

# %%
# Instantiate StatsForecast
sf = StatsForecast(models=models, freq="D", n_jobs=-1)

# %%
# Run cross-validation with fitted values enabled
cv_forecasts = sf.cross_validation(
    df=df_long,
    h=horizon,
    n_windows=n_windows,
    step_size=step_size,
    refit=True,
    level=prediction_interval_levels,
    fitted=True,
)

# %%
# Retrieve in-sample fitted values
cv_fitted = sf.cross_validation_fitted_values()

# %%
print("CV Forecasts shape:", cv_forecasts.shape)
cv_forecasts.info()
cv_forecasts.head()

# %%
print("CV Fitted values shape:", cv_fitted.shape)
cv_fitted.info()
cv_fitted.head()

# %%
# Model columns for residual computation
# aliases = ["TBATS", "AutoETS_7", "AutoETS_365", "SeasonalNaive_7"]

# Compute residuals for cv_fitted (in-sample)
for col in alias_name_list:
    if col in cv_fitted.columns:
        cv_fitted[f"{col}_residual"] = cv_fitted["y"] - cv_fitted[col]

# %%
# Verify residual columns added
print("CV Forecasts columns:", cv_forecasts.columns.tolist())
print("\nCV Fitted columns:", cv_fitted.columns.tolist())

# %%
cv_forecasts.info()
cv_forecasts.head()

# %%
# List cutoff dates for reference
cutoff_list = list(cv_forecasts["cutoff"].unique())
print("CV Fold cutoff dates:")
for i, cutoff in enumerate(cutoff_list, 1):
    print(f"  Fold {i}: {cutoff}")

# %%
# Visualization: Summary plot of all CV forecasts
plot_series(df_long, cv_forecasts.drop(columns=["y", "cutoff"]), engine="plotly")

# %%


# Visualization: Per-fold plots
for cutoff_id in cv_forecasts["cutoff"].unique():
    temp_df = cv_forecasts[cv_forecasts["cutoff"] == cutoff_id].copy()
    print(f"\nFold cutoff: {cutoff_id}")
    fig = plot_series(df_long, temp_df.drop(columns=["y", "cutoff"]))
    display(fig)

# %%
# Evaluation: Pooled RMSE (across all folds combined)
pooled_eval = evaluate(
    cv_forecasts.drop(columns=["cutoff"]),
    metrics=[rmse],
)
print("Pooled RMSE (all folds combined):")
print(pooled_eval)

# %%
# Evaluation: Average RMSE (RMSE per fold, then averaged)
fold_rmses = []
for cutoff in cv_forecasts["cutoff"].unique():
    fold_df = cv_forecasts[cv_forecasts["cutoff"] == cutoff].drop(columns=["cutoff"])
    fold_eval = evaluate(fold_df, metrics=[rmse])
    fold_eval["cutoff"] = cutoff
    fold_rmses.append(fold_eval)

fold_rmses_df = pd.concat(fold_rmses, ignore_index=True)
print("RMSE per fold:")
print(fold_rmses_df)

# %%
# Compute average RMSE across folds
avg_rmse_df = (
    fold_rmses_df.drop(columns=["cutoff"])
    .groupby(["unique_id", "metric"])
    .mean()
    .reset_index()
)
avg_rmse_df["metric"] = "rmse_avg"
print("\nAverage RMSE (across 5 folds):")
print(avg_rmse_df)

# %%
# Summary comparison
print("EVALUATION SUMMARY")
print("\nPooled RMSE:")
for col in alias_name_list:
    if col in pooled_eval.columns:
        print(f"  {col}: {pooled_eval[col].values[0]:{PRINT_PRECISION}}")

print("\nAverage RMSE:")
for col in alias_name_list:
    if col in avg_rmse_df.columns:
        print(f"  {col}: {avg_rmse_df[col].values[0]:{PRINT_PRECISION}}")

# %%
# Save results dataframes to parquet
if SAVE_BACKTEST_RESULTS_TO_FILE:
    cv_fitted.to_parquet(
        output_dir / f"cv_fitted_{timestamp}_daily_total.parquet", engine="pyarrow"
    )
    cv_forecasts.to_parquet(
        output_dir / f"cv_forecasts_{timestamp}_daily_total.parquet", engine="pyarrow"
    )
    test_set.to_parquet(
        output_dir / f"test_set_{timestamp}_daily_total.parquet", engine="pyarrow"
    )

    print(f"\nResults saved to: {output_dir}")
    print(f"  - cv_fitted_{timestamp}_daily_total.parquet")
    print(f"  - cv_forecasts_{timestamp}_daily_total.parquet")
    print(f"  - test_set_{timestamp}_daily_total.parquet")
