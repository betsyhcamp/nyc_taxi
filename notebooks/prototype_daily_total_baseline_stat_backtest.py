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
from coreforecast.scalers import boxcox, inv_boxcox, boxcox_lambda

from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA,
    AutoETS,
    AutoCES,
    AutoTheta,
    TBATS,
    SeasonalNaive,
)

from utilsforecast.plotting import plot_series
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import rmse

# %% [markdown]
# Top level (Total Manhattan) only have one time series, so scaled error metric are not helpful. Scaled metrics (e.g., WAPE, RMSSE, MASE) used when we are comparing multiple time series of differing scales. Within non-scaled metrics, can choose those that use the Absolute value or the Squared Error.
#
# For taxi zone time series, choose a scaled metric. And for hiearcharchical reconilliation, we will minimize squared error. So, makes most sense to choose a forecast error metric that also minimized squared error. Conclusion: for taxi zone time series, use RMSSE -> WRMSSE.
#
# So, for the top level, I will use a non-scaled metric that minimizes squared error. Conclusion: for top-level, use RMSE.

# %%
total_daily_df = pd.read_parquet("../data/preprocessed/daily_total_timeseries.parquet")

# %%
total_daily_df.info()
total_daily_df.head()

# %%
# visualize entire time series
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(total_daily_df["pickup_date"], total_daily_df["number_ride_pickups"])

ax.set_title("Daily Yellow Taxi Pickups in Manhattan")
ax.set_xlabel("Date")
ax.set_ylabel("Number of Pickups")
ax.grid(alpha=0.2)
fig.autofmt_xdate()

plt.tight_layout()

plt.show()

# %%

# %%
bc_lambda = boxcox_lambda(
    total_daily_df["number_ride_pickups"].values,
    method="loglik",
    season_length=7,
)
bc_ts = boxcox(total_daily_df["number_ride_pickups"].values, bc_lambda)
total_daily_df["boxcox_lambda"] = bc_lambda
total_daily_df["box_cox_timeseries"] = bc_ts


df_long = total_daily_df.melt(
    id_vars="pickup_date",
    value_vars=["number_ride_pickups", "box_cox_timeseries"],
    var_name="unique_id",
    value_name="y",
)

df_long = df_long[["unique_id", "pickup_date", "y"]].rename(
    columns={"pickup_date": "ds"}
)

# %%
df_long.info()
df_long.head()

# %%
df_long["unique_id"].unique()

# %%
# Cross-validation setup
horizon = 28  # 28 days, 4 weeks for forecast horizon

n_windows = (
    6  # number of cv folds 5 for CV, 1 Test set (generalization error that you report)
)
step_size = 28  # number of observations to step per cv-fold (non-overlapping)

# %%
# models = [
#    AutoARIMA(season_length=365, alias='ARIMA_365'),
#    AutoARIMA(season_length=7, alias='ARIMA_7'),
#
#    AutoETS(season_length=365, alias='ETS_365'),
#    AutoETS(season_length=7, alias='ETS_7'),
#
#    AutoCES(season_length=365, alias='CES_365'),
#    AutoCES(season_length=7, alias='CES_7'),
#
#    AutoTheta(season_length=365, alias='Theta_365'),
#    AutoTheta(season_length=7, alias='Theta_7'),
#
#    TBATS(season_length=[7,365]),
#
#    SeasonalNaive(season_length=7)
# ]

models = [TBATS(season_length=[7, 365]), SeasonalNaive(season_length=7)]

prediction_interval_levels = [60, 80, 95]

# %%
# Instatiate StatsForecast
sf = StatsForecast(models=models, freq="D", n_jobs=-1)

# to fit a single model
# sf.fit(df)
# forecast_df = sf.predict(h=12, level=[95])# sf.predict(h=12, level=prediction_interval_levels)

# %%
cv = sf.cross_validation(
    df=df_long, h=horizon, n_windows=n_windows, step_size=step_size, refit=True
)

# %%
plot_series(df_long, cv.drop(columns=["y", "cutoff"]))

# %%
cv

# %%
plot_series(df_long, cv.drop(columns=["y", "cutoff"]), engine="plotly")

# %%
plot_series(df_long, cv.drop(columns=["y", "cutoff"]), engine="plotly")

# %%
for cutoff_id in list(cv["cutoff"].unique()):
    temp_df = cv[cv["cutoff"] == cutoff_id].copy()

    plot_series(df_long, temp_df.drop(columns=["y", "cutoff"]))


# %%
cutoff_list = list(cv["cutoff"].unique())

# %%
cv[cv["cutoff"] == cutoff_list[0]]

# %%
cv.head()

# %%
# need to do inverse_box_cox before measuring error on unique_id = "box_cox_timeseries"

# %%
cv["cutoff"].unique()

# %%
cv["cutoff"]

# %%
evaluation = evaluate(
    cv.drop(columns=["cutoff"]),
    metrics=[rmse],
)

# %%
evaluation

# %%
evaluation.describe()

# %%
evaluation.info()

# %%
