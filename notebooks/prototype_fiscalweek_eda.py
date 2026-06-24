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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from tsbricks.blocks.dataio import read_sql, query_to_dataframe
from tsbricks.blocks.plots import plot_seasonal
from fcstnyctaxi.lib.utils import get_project_root_dir
from statsforecast.models import AutoARIMA


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
    "fiscal_week_of_month": "int64",
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
df["fiscal_week_start_date"] = pd.to_datetime(df["fiscal_week_start_date"])

# %%
df.loc[df['day_of_fiscal_month']==1, ["pickup_date", "day_of_fiscal_month",	"fiscal_year_week",	"fiscal_week", "fiscal_week_of_month","fiscal_month","fiscal_year","fiscal_year_month","fiscal_week_start_date", "day_of_week",	"day_of_week_name"]].drop_duplicates().tail(15)

# %%
# need to specify forecast origins for cross-val()(    )

# %%
df['pickup_taxi_zone_id'].nunique()

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
zone_fiscalweek_df = (
    df.groupby(['fiscal_year_month', 'fiscal_year_week','fiscal_week_start_date', 'fiscal_week_of_month', 'pickup_taxi_zone_id'])['number_ride_pickups']
    .sum()
    .reset_index(drop=False)
    .sort_values('fiscal_year_week')
    .reset_index(drop=True)
)


zone_fiscalweek_df

# %%
zone_fiscalweek_df = zone_fiscalweek_df.rename(columns=
                                               {
                                                   'pickup_taxi_zone_id': 'unique_id',
                                                   'number_ride_pickups': 'y'
                                                }
)

# %%
zone_fiscalweek_df

# %%
df = zone_fiscalweek_df.copy()

# %%
# ── Constants ──────────────────────────────────────────────────────────────────
N_WEEKS = 52


df['y_pos'] = df["y"].clip(lower=0)


# ── Trailing window ────────────────────────────────────────────────────────────
max_date = df["fiscal_week_start_date"].max()
cutoff_date = max_date - pd.Timedelta(weeks=N_WEEKS - 1)
trailing_df = df[df["fiscal_week_start_date"] >= cutoff_date]


# %%
# ── Full history aggregations ──────────────────────────────────────────────────
full_stats = df.groupby("unique_id", as_index=False).agg(
    mean_net_all_weeks=("y", "mean"),
    median_net_all_weeks=("y", "median"),
    total_all_weeks=("y", "sum"),
    total_positive_all_weeks=("y_pos", "sum"),
)

pos_full_stats = (
    df[df["y"] > 0]
    .groupby("unique_id", as_index=False)
    .agg(
        mean_pos_all_weeks=("y", "mean"),
        median_pos_all_weeks=("y", "median"),
    )
)

# %%
# ── Trailing N weeks aggregations ──────────────────────────────────────────────

trailing_stats = trailing_df.groupby("unique_id", as_index=False).agg(
    mean_net_N_weeks=("y", "mean"),
    median_net_N_weeks=("y", "median"),
    total_N_weeks=("y", "sum"),
    total_positive_N_weeks=("y_pos", "sum"),
)

pos_trailing_stats = (
    trailing_df[trailing_df["y"] > 0]
    .groupby("unique_id", as_index=False)
    .agg(
        mean_pos_N_weeks=("y", "mean"),
        median_pos_N_weeks=("y", "median"),
    )
)

# %%
# ── Time series start and length ───────────────────────────────────────────────
ts_start = (
    df[df["y"].notna() & (df["y"] != 0)]
    .groupby("unique_id")["fiscal_week_start_date"]
    .min()
    .reset_index()
    .rename(columns={"fiscal_week_start_date": "time_series_start_week"})
)


# %%
# ── Merge into summary_df ──────────────────────────────────────────────────────
summary_df = (
    full_stats
    .merge(pos_full_stats, on="unique_id", how="left")
    .merge(trailing_stats, on="unique_id", how="left")
    .merge(pos_trailing_stats, on="unique_id", how="left")
    .merge(ts_start, on="unique_id", how="left")
)

summary_df["time_series_length_weeks"] = (
    (max_date - summary_df["time_series_start_week"]).dt.days / 7
).astype(int)

# %%
# ── Ranks (1 = largest) ────────────────────────────────────────────────────────
col_to_rank = [
    ("total_all_weeks",          "rank_all_weeks"),
    ("total_positive_all_weeks", "rank_positive_all_weeks"),
    ("total_N_weeks",            "rank_N_weeks"),
    ("total_positive_N_weeks",   "rank_positive_N_weeks"),
]

for col, rank_col in col_to_rank:
    summary_df[rank_col] = (
        summary_df[col].rank(ascending=False, method="min").astype("Int64")
    )

# %%
# ── Reorder columns ────────────────────────────────────────────────────────────
summary_df = summary_df[[
    "unique_id",
    "mean_net_all_weeks",      "median_net_all_weeks",
    "mean_pos_all_weeks",      "median_pos_all_weeks",
    "mean_net_N_weeks",        "median_net_N_weeks",
    "mean_pos_N_weeks",        "median_pos_N_weeks",
    "total_all_weeks",         "total_positive_all_weeks",
    "total_N_weeks",           "total_positive_N_weeks",
    "rank_all_weeks",          "rank_positive_all_weeks",
    "rank_N_weeks",            "rank_positive_N_weeks",
    "time_series_start_week",
    "time_series_length_weeks",
]]

# %%
summary_df.head()

# %%
summary_df

# %% [markdown]
# # Section 1: Mean vs. Median Positive Weekly Revenue
#

# %%
plot_df = summary_df[summary_df["mean_pos_all_weeks"].notna()].copy()

diag_min = min(
    plot_df["mean_pos_all_weeks"].min(),
    plot_df["median_pos_all_weeks"].min(),
)
diag_max = max(
    plot_df["mean_pos_all_weeks"].max(),
    plot_df["median_pos_all_weeks"].max(),
)

# %%
fig, axes = plt.subplots(1, 2, figsize=(12.5, 6))

for ax in axes:
    ax.scatter(
        plot_df["mean_pos_all_weeks"],
        plot_df["median_pos_all_weeks"],
        facecolors="none",
        edgecolors="black",
        marker="o",
        alpha=0.6,
    )
    ax.plot(
        [diag_min, diag_max],
        [diag_min, diag_max],
        linestyle="dotted",
        color="black",
    )
    ax.set_xlabel("Mean positive weekly y")
    ax.set_ylabel("Median positive weekly y")
    ax.set_aspect("equal", adjustable="box")

axes[0].set_title("Linear Scale")
axes[1].set_title("Log Scale")
axes[1].set_xscale("log")
axes[1].set_yscale("log")

fig.suptitle("Mean vs. Median Positive Weekly y")
plt.tight_layout()
plt.show()

# %%
hover_data = plot_df[["unique_id", "mean_pos_all_weeks", "median_pos_all_weeks"]].values
hovertemplate = (
    "<b>Zone: %{customdata[0]}</b><br>"
    "Mean: %{customdata[1]:.1f}<br>"
    "Median: %{customdata[2]:.1f}"
    "<extra></extra>"
)

fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=["Linear Scale", "Log Scale"],
)

for col in [1, 2]:
    fig.add_trace(
        go.Scatter(
            x=plot_df["mean_pos_all_weeks"],
            y=plot_df["median_pos_all_weeks"],
            mode="markers",
            marker=dict(
                symbol="circle-open",
                color="black",
                opacity=0.6,
            ),
            customdata=hover_data,
            hovertemplate=hovertemplate,
            showlegend=False,
        ),
        row=1,
        col=col,
    )
    fig.add_shape(
        type="line",
        x0=diag_min,
        y0=diag_min,
        x1=diag_max,
        y1=diag_max,
        line=dict(color="black", dash="dot"),
        row=1,
        col=col,
    )

fig.update_xaxes(title_text="Mean positive weekly y", row=1, col=1)
fig.update_yaxes(title_text="Median positive weekly y", scaleanchor="x", scaleratio=1, row=1, col=1)
fig.update_xaxes(title_text="Mean positive weekly y (log scale)", type="log", row=1, col=2)
fig.update_yaxes(title_text="Median positive weekly y (log scale)", type="log", scaleanchor="x2", scaleratio=1, row=1, col=2)

fig.update_layout(
    title="Mean vs. Median Positive Weekly y",
    height=550,
    width=1100,
)

fig.show()

# %%
