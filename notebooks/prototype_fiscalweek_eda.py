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
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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


# ── Global flags ──────────────────────────────────────────────────────────────────
df['y_pos'] = df["y"].clip(lower=0)
df["y_neg_flag"] = (df["y"] < 0).astype(int)

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
# revenue buckets
summary_df["revenue_bucket"] = pd.qcut(
    summary_df["mean_pos_N_weeks"].rank(method="first"),
    q=5,
    labels=["Very low", "Low", "Middle", "High", "Very high"],
)


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
    #title="Mean vs. Median Positive Weekly y",
    height=550,
    width=1100,
)

fig.show()

# %% [markdown]
# # Section 2: Revenue concentration

# %%
n = len(summary_df)

pos_full_sorted = summary_df.sort_values("total_positive_all_weeks", ascending=False)
pos_full_cum = np.concatenate([
    [0],
    pos_full_sorted["total_positive_all_weeks"].cumsum().values / pos_full_sorted["total_positive_all_weeks"].sum(),
])

pos_trail_sorted = summary_df.sort_values("total_positive_N_weeks", ascending=False)
pos_trail_cum = np.concatenate([
    [0],
    pos_trail_sorted["total_positive_N_weeks"].cumsum().values / pos_trail_sorted["total_positive_N_weeks"].sum(),
])

net_full_sorted = summary_df.sort_values("total_all_weeks", ascending=False)
net_full_cum = np.concatenate([
    [0],
    net_full_sorted["total_all_weeks"].cumsum().values / net_full_sorted["total_all_weeks"].sum(),
])

net_trail_sorted = summary_df.sort_values("total_N_weeks", ascending=False)
net_trail_cum = np.concatenate([
    [0],
    net_trail_sorted["total_N_weeks"].cumsum().values / net_trail_sorted["total_N_weeks"].sum(),
])

x_prop = np.concatenate([[0], np.arange(1, n + 1) / n])
x_count = np.concatenate([[0], np.arange(1, n + 1)])

# %% [markdown]
# ## Section 2.1 Cumulative Revenue Concentration Curve

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)

panels = [
    (pos_full_cum, pos_trail_cum, "Positive Revenue"),
    (net_full_cum, net_trail_cum, "Net Revenue"),
]

for ax, (full_cum, trail_cum, title) in zip(axes, panels):
    ax.plot(x_prop, full_cum, color="black", linestyle="solid", label="Full history", alpha=0.7)
    ax.plot(x_prop, trail_cum, color="tab:blue", linestyle="solid", label=f"Trailing {N_WEEKS} weeks", alpha=0.7)
    ax.plot([0, 1], [0, 1], color="gray", linestyle="dotted", alpha=0.6)
    ax.set_xlabel("Cumulative proportion of series, ranked descending")
    ax.set_ylabel("Cumulative revenue proportion")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

fig.suptitle("Cumulative Revenue Concentration by Series")
plt.show()

# %% [markdown]
# ## Section 2.2: Top-N Contribution Curves

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)

panels = [
    (pos_full_cum, pos_trail_cum, "Positive Revenue"),
    (net_full_cum, net_trail_cum, "Net Revenue"),
]

for ax, (full_cum, trail_cum, title) in zip(axes, panels):
    ax.plot(x_count, full_cum, color="black", linestyle="solid", label="Full history", alpha=0.7)
    ax.plot(x_count, trail_cum, color="tab:blue", linestyle="solid", label=f"Trailing {N_WEEKS} weeks", alpha=0.7)
    for ref in [0.50, 0.80, 0.90, 0.95]:
        ax.axhline(y=ref, color="gray", linestyle="dotted", alpha=0.7)
    ax.set_xlabel("Number of top-ranked series")
    ax.set_ylabel("Cumulative revenue proportion")
    ax.set_title(title)
    ax.legend()
    ax.set_box_aspect(1)
    ax.grid(alpha=0.3)

fig.suptitle("Cumulative Revenue by Top-N Series")
plt.show()

# %% [markdown]
# ## Section 2.3: Top-N Revenue Bar Chart

# %%
# ── Section 2.3: Top-N Revenue Bar Chart ──────────────────────────────────────
TOP_N = 20

top_full = summary_df.sort_values("total_all_weeks", ascending=False).head(TOP_N)
top_trail = summary_df.sort_values("total_N_weeks", ascending=False).head(TOP_N)

#y_max = max(top_full["total_all_weeks"].max(), top_trail["total_N_weeks"].max())

fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

panels = [
    (axes[0], top_full, "total_all_weeks", "Full History"),
    (axes[1], top_trail, "total_N_weeks", f"Trailing {N_WEEKS} Weeks"),
]

for ax, data, col, title in panels:
    ax.bar(range(TOP_N), data[col], color="gray")
    ax.set_xticks(range(TOP_N))
    ax.set_xticklabels(data["unique_id"].astype(str), rotation=90)
    ax.set_xlabel("Zone")
    ax.set_ylabel("Total net revenue (y)")
    ax.set_title(title)
    #ax.set_ylim(0, y_max * 1.05)

fig.suptitle(f"Top-{TOP_N} Series by Net Revenue")
plt.show()

# %% [markdown]
# # Section 3: Month-to-date signal

# %%
N_WEEKS_104 = 104

# Sort and compute mtd_y
mtd_df = df.sort_values(["unique_id", "fiscal_year_month", "fiscal_week_of_month"]).copy()
mtd_df["mtd_y"] = mtd_df.groupby(["unique_id", "fiscal_year_month"])["y"].cumsum()
mtd_df = mtd_df.rename(columns={"fiscal_week_of_month": "origin_week"})
mtd_df.head()

# %%
# final_month_y: total y per (unique_id, fiscal_year_month), merged back
final_month = (
    df.groupby(["unique_id", "fiscal_year_month"], as_index=False)["y"]
    .sum()
    .rename(columns={"y": "final_month_y"})
)
mtd_df = mtd_df.merge(final_month, on=["unique_id", "fiscal_year_month"], how="left")
mtd_df.head()

# %%
mtd_df['fiscal_month_number'] = mtd_df['fiscal_year_month'].astype(str).str[-2:].astype(int)
mtd_df['fiscal_week_number'] = mtd_df['fiscal_year_week'].astype(str).str[-2:].astype(int)

# %%
# weeks_in_month: calendar property — max origin week per fiscal_year_month
weeks_in_month = (
    df.groupby("fiscal_year_month")["fiscal_week_of_month"]
    .max()
    .rename("weeks_in_month")
    .reset_index()
)


# %%
mtd_df = mtd_df.merge(weeks_in_month, on="fiscal_year_month", how="left")

# mtd_share
mtd_df["mtd_share"] = mtd_df["mtd_y"] / mtd_df["final_month_y"]

# Join mean_pos_N_weeks and revenue_bucket from summary_df
mtd_df = mtd_df.merge(
    summary_df[["unique_id", "mean_pos_N_weeks", "revenue_bucket"]],
    on="unique_id",
    how="left",
)

# core_threshold: per-series, computed after join
mtd_df["core_threshold"] = np.maximum(1000, 0.25 * mtd_df["mean_pos_N_weeks"])

# trailing_104_months: set of fiscal_year_month values within the trailing 2-year window
trailing_104_months = set(
    df.loc[
        df["fiscal_week_start_date"] >= max_date - pd.Timedelta(weeks=103),
        "fiscal_year_month",
    ].unique()
)
#Two things worth checking after you run it:

#mtd_df.shape — you expect len(df) rows (one per original week row, now with cumsum added)
#Spot-check a single series: mtd_df[mtd_df["unique_id"] == 4][["fiscal_year_month", "origin_week", "y", "mtd_y", "final_month_y", "weeks_in_month"]].head(10) — mtd_y should increase within each month and equal final_month_y on the last week of the month.


# %%
mtd_df.head()


# %% [markdown]
# ## Section 3.1: MTD actuals vs. final month scatterplot 

# %%

def plot_mtd_scatter(data, color_col, colorbar_label, date_color=False):
    subplot_titles = ["Week 1", "Weeks 1–2", "Weeks 1–3", "Weeks 1–4"]
    row_labels = ["4-week months", "5-week months"]

    if date_color:
        c_numeric = mdates.date2num(data[color_col])
        vmin, vmax = c_numeric.min(), c_numeric.max()
    else:
        vmin = data[color_col].min()
        vmax = data[color_col].max()

    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)

    for row_idx, n_weeks in enumerate([4, 5]):
        for col_idx, origin in enumerate([1, 2, 3, 4]):
            ax = axes[row_idx, col_idx]
            mask = (
                (data["weeks_in_month"] == n_weeks)
                & (data["origin_week"] == origin)
                & (data["mtd_y"] > 0)
                & (data["final_month_y"] > 0)
            )
            sub = data[mask]
            c = mdates.date2num(sub[color_col]) if date_color else sub[color_col]
            ax.scatter(
                sub["mtd_y"],
                sub["final_month_y"],
                c=c,
                cmap=cmap,
                norm=norm,
                alpha=0.5,
                s=15,
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            if row_idx == 0:
                ax.set_title(subplot_titles[col_idx])
            if col_idx == 0:
                ax.set_ylabel(row_labels[row_idx])

    fig.supxlabel("MTD revenue")
    fig.supylabel("Final month revenue")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, label=colorbar_label)
    if date_color:
        cb.ax.yaxis.set_major_locator(mdates.YearLocator())
        cb.ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.suptitle("MTD Actuals vs. Final Month Revenue")
    plt.show()


plot_mtd_scatter(mtd_df, "fiscal_week_start_date", "Fiscal week start date", date_color=True)
plot_mtd_scatter(mtd_df, "fiscal_month_number", "Fiscal month number")
plot_mtd_scatter(mtd_df, "fiscal_week_number", "Fiscal week number")


# %% [markdown]
# # Section 4: Negative revenue diagnostics (and revenue bucket setup)

# %%
neg_counts = df.groupby("unique_id", as_index=False).agg(
    n_weeks_total=("y", "count"),
    n_negative_weeks=("y_neg_flag", "sum"),
)

neg_sum = (
    df[df["y"] < 0]
    .groupby("unique_id", as_index=False)["y"]
    .sum()
    .rename(columns={"y": "sum_negative_y"})
)

summary_df = (
    summary_df
    .merge(neg_counts, on="unique_id", how="left")
    .merge(neg_sum, on="unique_id", how="left")
)
summary_df["sum_negative_y"] = summary_df["sum_negative_y"].fillna(0)
summary_df["frac_negative_weeks"] = summary_df["n_negative_weeks"] / summary_df["n_weeks_total"]
summary_df["neg_materiality"] = summary_df["sum_negative_y"].abs() / summary_df["total_positive_all_weeks"]


# %%
# Spot-check: series with no negatives should have sum_negative_y == 0 and neg_materiality == 0
assert (summary_df["sum_negative_y"] <= 0).all(), "sum_negative_y should be ≤ 0"
assert (summary_df["neg_materiality"] >= 0).all()
assert summary_df["revenue_tier"].value_counts().shape[0] == 5  # all 5 buckets present
summary_df[["unique_id", "n_weeks_total", "n_negative_weeks", "frac_negative_weeks", "sum_negative_y", "neg_materiality", "revenue_tier"]].head(10)


# %% [markdown]
# # Section 8: Time Series Length Histogram

# %%
hist_df = summary_df.dropna(subset=["time_series_length_weeks"])

fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

ax.hist(
    hist_df["time_series_length_weeks"],
    bins=range(
        int(hist_df["time_series_length_weeks"].min()),
        int(hist_df["time_series_length_weeks"].max()) + 2,
    ),
    color="gray",
    edgecolor="gray",
)
ax.set_xlabel("Time series length (weeks)")
ax.set_ylabel("Number of series")
ax.set_title("Distribution of Time Series Lengths")

plt.show()

# %% [markdown]
#
