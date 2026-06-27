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
import uuid
from datetime import datetime, timezone
from pathlib import Path


from google.cloud import bigquery
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tsbricks.backtesting import evaluate_metrics, generate_folds, parse_config
from tsbricks.blocks.dataio import read_sql, query_to_dataframe
from tsbricks.blocks.diagnostics import plot_acf, plot_pacf
from tsbricks.blocks.plots import plot_seasonal
from tsbricks.runner import apply_transforms, fit_transforms, inverse_transforms, invoke_model
from utilsforecast.evaluation import evaluate

from fcstnyctaxi.lib.utils import get_project_root_dir

# %%
project = "nyc-taxi-ehc"
location = "us-central1"
input_sql_filename = "initial_daily_taxi_revenue.sql"

project_root = get_project_root_dir()

queries_dir = project_root / "queries"
raw_data_dir = project_root / "data"/ "raw"

# %% [markdown]
# # Section: Data ingestion

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

# %% [markdown]
# # Section: Data preparation

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
    df.groupby(['fiscal_year','fiscal_year_month', 'fiscal_year_week','fiscal_week_start_date', 'fiscal_week_of_month', 'pickup_taxi_zone_id'])['number_ride_pickups']
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
zone_fiscalweek_df[zone_fiscalweek_df['unique_id']==104].sort_values(by='fiscal_year_week')

# %%
zone_fiscalweek_df.loc[zone_fiscalweek_df['unique_id']==104, 'y'].max()

# %%
df = zone_fiscalweek_df.copy()

# %% [markdown]
# # Section: Weekly Time series lineplots/ ACF / PACF

# %%
NUMBER_LAGS = 54
SAVE_TS_FIGS = False  # either set True or False

UNIQUE_IDS=[161, 236] 

if SAVE_TS_FIGS:
    _run_ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
    _run_uuid = uuid.uuid4().hex[:5]
    _fig_dir = project_root / "notebooks"/"eda_figs" / f"eda_figures_UTC{_run_ts}_{_run_uuid}"
    _fig_dir.mkdir(parents=True, exist_ok=True)

for uid in UNIQUE_IDS:
    sub = df[df["unique_id"] == uid].sort_values("fiscal_week_start_date")

    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])

    ax_top = fig.add_subplot(gs[0, :])
    ax_acf = fig.add_subplot(gs[1, 0])
    ax_pacf = fig.add_subplot(gs[1, 1])

    ax_top.plot(sub["fiscal_week_start_date"], sub["y"], linewidth=1)
    ax_top.set_xlabel("fiscal_week_start_date")
    ax_top.set_ylabel("y")
    ax_top.grid(alpha=0.3)

    plot_acf(sub, time_col="fiscal_week_start_date", value_col="y",
             lags=NUMBER_LAGS, backend="matplotlib", ax=ax_acf)
    plot_pacf(sub, time_col="fiscal_week_start_date", value_col="y",
              lags=NUMBER_LAGS, backend="matplotlib", ax=ax_pacf)

    fig.suptitle(f"unique_id: {uid}")

    if SAVE_TS_FIGS:
        _fname = f"UTC{_run_ts}_lineplot_acf_pacf_{uid}.jpeg"
        fig.savefig(_fig_dir / _fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# %% [markdown]
# # Section: Seasonal plots

# %%
for uid in UNIQUE_IDS:
    fig = plot_seasonal(
        df=df[df["unique_id"] == uid],
        time_col="fiscal_year_week",
        value_col="y",
        backend="matplotlib",
        width=800,
        height=450,
        palette="viridis",
        alpha=0.8,
        season_col="fiscal_year",
        return_fig=True,
    )
    fig.suptitle(f"unique_id: {uid}")

    if SAVE_TS_FIGS:
        _fname = f"UTC{_run_ts}_yearlyseasonalplot_{uid}.jpeg"
        fig.savefig(_fig_dir / _fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# %%
df["weeks_in_month"] = df.groupby("fiscal_year_month")["fiscal_week_of_month"].transform("max")
df = df[df["weeks_in_month"]>=4].reset_index(drop=True)

# %% [markdown]
# # Section: Constant definitions, flags, window definition for summary stats

# %%
# Constants
N_WEEKS = 52
ADI_THRESHOLD = 1.32
CV2_THRESHOLD = 0.49
CLASS_ORDER = ["Smooth", "Erratic", "Intermittent", "Lumpy"]
BUCKETS = ["Very low", "Low", "Middle", "High", "Very high"]

# Global flags
df['y_pos'] = df["y"].clip(lower=0)
df["y_neg_flag"] = (df["y"] < 0).astype(int)

# Trailing window
max_date = df["fiscal_week_start_date"].max()
cutoff_date = max_date - pd.Timedelta(weeks=N_WEEKS - 1)
trailing_df = df[df["fiscal_week_start_date"] >= cutoff_date]


# %% [markdown]
# # Section: Summary stats

# %%
# Full history aggregations
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
# Trailing N weeks aggregations
trailing_stats = trailing_df.groupby("unique_id", as_index=False).agg(
    mean_net_short=("y", "mean"),
    median_net_short=("y", "median"),
    total_short=("y", "sum"),
    total_positive_short=("y_pos", "sum"),
)

pos_trailing_stats = (
    trailing_df[trailing_df["y"] > 0]
    .groupby("unique_id", as_index=False)
    .agg(
        mean_pos_short=("y", "mean"),
        median_pos_short=("y", "median"),
    )
)

# %%
# Time series start and length
ts_start = (
    df[df["y"].notna() & (df["y"] != 0)]
    .groupby("unique_id")["fiscal_week_start_date"]
    .min()
    .reset_index()
    .rename(columns={"fiscal_week_start_date": "time_series_start_week"})
)


# %%
# Merge into summary_df
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
# Ranks (1 = largest)
col_to_rank = [
    ("total_all_weeks",          "rank_all_weeks"),
    ("total_positive_all_weeks", "rank_positive_all_weeks"),
    ("total_short",            "rank_short"),
    ("total_positive_short",   "rank_positive_short"),
]

for col, rank_col in col_to_rank:
    summary_df[rank_col] = (
        summary_df[col].rank(ascending=False, method="min").astype("Int64")
    )

# %%
# Reorder columns
summary_df = summary_df[[
    "unique_id",
    "mean_net_all_weeks",      "median_net_all_weeks",
    "mean_pos_all_weeks",      "median_pos_all_weeks",
    "mean_net_short",        "median_net_short",
    "mean_pos_short",        "median_pos_short",
    "total_all_weeks",         "total_positive_all_weeks",
    "total_short",           "total_positive_short",
    "rank_all_weeks",          "rank_positive_all_weeks",
    "rank_short",            "rank_positive_short",
    "time_series_start_week",
    "time_series_length_weeks",
]]

# %%
# revenue buckets
summary_df["bucket"] = pd.qcut(
    summary_df["mean_pos_short"].rank(method="first"),
    q=5,
    labels=BUCKETS,
)


# %%
summary_df.head()

# %%
summary_df

# %% [markdown]
# # Section: Mean vs. Median Positive Weekly Revenue
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
# # Section: Revenue concentration

# %%
n = len(summary_df)

pos_full_sorted = summary_df.sort_values("total_positive_all_weeks", ascending=False)
pos_full_cum = np.concatenate([
    [0],
    pos_full_sorted["total_positive_all_weeks"].cumsum().values / pos_full_sorted["total_positive_all_weeks"].sum(),
])

pos_trail_sorted = summary_df.sort_values("total_positive_short", ascending=False)
pos_trail_cum = np.concatenate([
    [0],
    pos_trail_sorted["total_positive_short"].cumsum().values / pos_trail_sorted["total_positive_short"].sum(),
])

net_full_sorted = summary_df.sort_values("total_all_weeks", ascending=False)
net_full_cum = np.concatenate([
    [0],
    net_full_sorted["total_all_weeks"].cumsum().values / net_full_sorted["total_all_weeks"].sum(),
])

net_trail_sorted = summary_df.sort_values("total_short", ascending=False)
net_trail_cum = np.concatenate([
    [0],
    net_trail_sorted["total_short"].cumsum().values / net_trail_sorted["total_short"].sum(),
])

x_prop = np.concatenate([[0], np.arange(1, n + 1) / n])
x_count = np.concatenate([[0], np.arange(1, n + 1)])

# %% [markdown]
# ## Subsec: Cumulative Revenue Concentration Curve

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
# ## Subsec: Top-N Contribution Curves

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
# ## Subsec: Top-N Revenue Bar Chart

# %%
TOP_N = 30

top_full_total  = summary_df.sort_values("total_all_weeks", ascending=False).head(TOP_N)
top_short_total = summary_df.sort_values("total_short",     ascending=False).head(TOP_N)
top_full_mean   = summary_df.sort_values("mean_pos_all_weeks",   ascending=False).head(TOP_N)
top_short_mean  = summary_df.sort_values("mean_pos_short",  ascending=False).head(TOP_N)

fig, axes = plt.subplots(2, 2, figsize=(18, 8), constrained_layout=True)
axes[1, 1].sharey(axes[0, 1])

panels = [
    (axes[0, 0], top_full_total,  "total_all_weeks", "Sum $, Full History",              "Total net revenue (y)", "silver", None),
    (axes[1, 0], top_short_total, "total_short",     f"Sum $, Trailing {N_WEEKS} Weeks", "Total net revenue (y)", "silver", "//"),
    (axes[0, 1], top_full_mean,   "mean_pos_all_weeks",   "Mean weekly $, Full History",              "Mean positive weekly revenue (y)", "tab:blue", None),
    (axes[1, 1], top_short_mean,  "mean_pos_short",  f"Mean weekly $, Trailing {N_WEEKS} Weeks", "Mean positive weekly revenue (y)", "tab:blue", "//"),
]

for ax, data, col, title, ylabel, color, hatch in panels:
    ax.bar(range(TOP_N), data[col], color=color, hatch=hatch)
    ax.set_xticks(range(TOP_N))
    ax.set_xticklabels(data["unique_id"].astype(str), rotation=90)
    ax.set_xlabel("Zone")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

fig.suptitle(f"Top {TOP_N} Ranked Series by Revenue")
plt.show()


# %% [markdown]
# # Section: Month-to-date signal

# %% [markdown]
# ## Subsec: MTD data prep

# %%
N_WEEKS_104 = 104

# Sort and compute y_mtd
mtd_df = df.sort_values(["unique_id", "fiscal_year_month", "fiscal_week_of_month"]).copy()
mtd_df["y_mtd"] = mtd_df.groupby(["unique_id", "fiscal_year_month"])["y"].cumsum()
mtd_df = mtd_df.rename(columns={"fiscal_week_of_month": "origin_week"})
mtd_df.head()

# %%
# y_final_month: total y per (unique_id, fiscal_year_month), merged back
final_month = (
    df.groupby(["unique_id", "fiscal_year_month"], as_index=False)["y"]
    .sum()
    .rename(columns={"y": "y_final_month"})
)
mtd_df = mtd_df.merge(final_month, on=["unique_id", "fiscal_year_month"], how="left")
mtd_df.head()

# %%
mtd_df['fiscal_month_number'] = mtd_df['fiscal_year_month'].astype(str).str[-2:].astype(int)
mtd_df['fiscal_week_number'] = mtd_df['fiscal_year_week'].astype(str).str[-2:].astype(int)

# %%
mtd_df

# %%
# mtd_share
mtd_df["mtd_share"] = mtd_df["y_mtd"] / mtd_df["y_final_month"]

# Join mean_pos_short and bucket from summary_df
mtd_df = mtd_df.merge(
    summary_df[["unique_id", "mean_pos_short", "bucket"]],
    on="unique_id",
    how="left",
)

# core_threshold: per-series, computed after join
mtd_df["core_threshold"] = np.maximum(1000, 0.25 * mtd_df["mean_pos_short"])

# trailing_104_months: set of fiscal_year_month values within the trailing 2-year window
trailing_104_months = set(
    df.loc[
        df["fiscal_week_start_date"] >= max_date - pd.Timedelta(weeks=103),
        "fiscal_year_month",
    ].unique()
)
#Two things worth checking after you run it:

#mtd_df.shape — you expect len(df) rows (one per original week row, now with cumsum added)
#Spot-check a single series: mtd_df[mtd_df["unique_id"] == 4][["fiscal_year_month", "origin_week", "y", "y_mtd", "y_final_month", "weeks_in_month"]].head(10) — y_mtd should increase within each month and equal y_final_month on the last week of the month.


# %%
mtd_df.head()


# %% [markdown]
# ## Subsec: MTD actuals vs. final month scatterplot 

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
                & (data["y_mtd"] > 0)
                & (data["y_final_month"] > 0)
            )
            sub = data[mask]
            c = mdates.date2num(sub[color_col]) if date_color else sub[color_col]
            ax.scatter(
                sub["y_mtd"],
                sub["y_final_month"],
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
# ## Subsec: MTD share boxplots

# %%

def plot_mtd_share_boxplots(data, title):
    buckets = ["Very low", "Low", "Middle", "High", "Very high"]
    row_labels = ["4-week months", "5-week months"]

    fig, axes = plt.subplots(2, 5, figsize=(20, 8), constrained_layout=True)

    for col_idx, bucket in enumerate(buckets):
        for row_idx, n_weeks in enumerate([4, 5]):
            ax = axes[row_idx, col_idx]
            mask = (
                (data["weeks_in_month"] == n_weeks)
                & (data["bucket"] == bucket)
                & (data["y_final_month"] >= data["core_threshold"])
            )
            sub = data[mask]
            box_data = [
                sub.loc[sub["origin_week"] == k, "mtd_share"].dropna().values
                for k in [1, 2, 3, 4]
            ]
            ax.boxplot(box_data, positions=[1, 2, 3, 4])
            ax.set_xticks([1, 2, 3, 4])
            if row_idx == 0:
                ax.set_title(bucket)
            if col_idx == 0:
                ax.set_ylabel(row_labels[row_idx])

    fig.supxlabel("Forecast origin (weeks of actuals)")
    fig.supylabel("MTD share (MTD revenue / final month revenue)")
    fig.suptitle(title)
    plt.show()


# 3.2a — Full history
plot_mtd_share_boxplots(
    mtd_df,
    "MTD Share by Forecast Origin and Revenue Bucket — Full History",
)

# 3.2b — Trailing 104 weeks
trailing_mtd_df = mtd_df[mtd_df["fiscal_year_month"].isin(trailing_104_months)]
plot_mtd_share_boxplots(
    trailing_mtd_df,
    "MTD Share by Forecast Origin and Revenue Bucket — Trailing 104 Weeks",
)


# %% [markdown]
# # Section: Negative revenue diagnostics

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
assert summary_df["bucket"].value_counts().shape[0] == 5  # all 5 buckets present
summary_df[["unique_id", "n_weeks_total", "n_negative_weeks", "frac_negative_weeks", "sum_negative_y", "neg_materiality", "bucket"]].head(10)


# %%
# Plotly approach to negative quantity scatter plot
def plot_neg_scatter(y_col, y_label, title, scale_type="log"):
    """Can use either scale=log or scale=linear"""
    hover_data = summary_df[["unique_id", "mean_pos_short", y_col]].values
    hovertemplate = (
        "<b>Zone: %{customdata[0]}</b><br>"
        "Mean positive weekly revenue (52w): %{customdata[1]:.1f}<br>"
        + y_label
        + ": %{customdata[2]:.4f}<extra></extra>"
    )
    fig = go.Figure(
        go.Scatter(
            x=summary_df["mean_pos_short"],
            y=summary_df[y_col],
            mode="markers",
            marker=dict(symbol="circle-open", color="black", opacity=0.6),
            customdata=hover_data,
            hovertemplate=hovertemplate,
            showlegend=False,
        )
    )
    fig.update_xaxes(type=scale_type, title_text="Mean positive weekly revenue (trailing 52w)")
    fig.update_yaxes(title_text=y_label)
    fig.update_layout(title=title, height=500, width=750)
    fig.show()


# %% [markdown]
# ## Subsec: Fraction of negative weeks

# %%
plot_neg_scatter(
    "frac_negative_weeks",
    "Fraction negative weeks",
    "Mean Positive Weekly Revenue (trailing 52w) vs. Fraction Negative Weeks",
)

# %% [markdown]
# ## Subsec: Revenue significance of negative weeks

# %%
plot_neg_scatter(
    "neg_materiality",
    "Negative materiality",
    "Mean Positive Weekly Revenue (trailing 52w) vs. Negative Materiality",
)


# %%
# matplotlib implementation
def plot_neg_scatter_mpl(y_col, y_label, title, scale_type="log"):
    """Can use either scale=log or scale=linear"""
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.scatter(
        summary_df["mean_pos_short"],
        summary_df[y_col],
        facecolors="none",
        edgecolors="black",
        alpha=0.6,
    )
    ax.set_xscale(scale_type)
    ax.grid(alpha=0.3)
    ax.set_xlabel("Mean positive weekly revenue (trailing 52w)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.show()



plot_neg_scatter_mpl(
    "frac_negative_weeks",
    "Fraction negative weeks",
    "Mean Positive Weekly Revenue (trailing 52w) vs. Fraction Negative Weeks",
)


plot_neg_scatter_mpl(
    "neg_materiality",
    "Negative materiality",
    "Mean Positive Weekly Revenue (trailing 52w) vs. Negative Materiality",
)


# %%
# Revenue rank vs. fraction negative weeks
hover_data_rank_neg = summary_df[["unique_id", "rank_positive_short", "frac_negative_weeks"]].values
hovertemplate_rank_neg = (
    "<b>Zone: %{customdata[0]}</b><br>"
    "Revenue rank (trailing 52w): %{customdata[1]:.0f}<br>"
    "Fraction negative weeks: %{customdata[2]:.4f}"
    "<extra></extra>"
)

fig = go.Figure(
    go.Scatter(
        x=summary_df["rank_positive_short"],
        y=summary_df["frac_negative_weeks"],
        mode="markers",
        marker=dict(symbol="circle-open", color="black", opacity=0.6),
        customdata=hover_data_rank_neg,
        hovertemplate=hovertemplate_rank_neg,
        showlegend=False,
    )
)
fig.update_xaxes(title_text="Revenue rank (trailing 52w positive revenue, rank 1 = largest)")
fig.update_yaxes(title_text="Fraction negative weeks")
fig.update_layout(
    title="Revenue Rank (trailing 52w positive revenue) vs. Fraction Negative Weeks",
    height=500,
    width=750,
)
fig.show()


# %%
fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
ax.scatter(
    summary_df["rank_positive_short"],
    summary_df["frac_negative_weeks"],
    facecolors="none",
    edgecolors="black",
    alpha=0.6,
)
ax.set_xlabel("Revenue rank (trailing 52w positive revenue, rank 1 = largest)")
ax.set_ylabel("Fraction negative weeks")
ax.set_title("Revenue Rank (trailing 52w positive revenue) vs. Fraction Negative Weeks")
ax.grid(alpha=0.3)
plt.show()


# %% [markdown]
# # Section: Intermittency metrics 

# %% [markdown]
# ## Subsec: Extend summary_df with intermittency metrics

# %%
# Full-history counts (assign avoids modifying df in place)
_df_full = df.assign(_is_zero=df["y"] == 0, _is_positive=df["y"] > 0)
full_counts = _df_full.groupby("unique_id", as_index=False).agg(
    n_zero_weeks_full=("_is_zero", "sum"),
    n_positive_weeks_full=("_is_positive", "sum"),
)

full_pos_stats = (
    df[df["y"] > 0]
    .groupby("unique_id", as_index=False)
    .agg(
        mean_pos_full=("y", "mean"),
        std_pos_full=("y", "std"),
    )
)

full_metrics = (
    full_counts
    .merge(full_pos_stats, on="unique_id", how="left")
    .merge(summary_df[["unique_id", "n_weeks_total"]], on="unique_id", how="left")
)
full_metrics["frac_zero_weeks_full"] = (
    full_metrics["n_zero_weeks_full"] / full_metrics["n_weeks_total"]
)
full_metrics["cv2_full"] = (
    (full_metrics["std_pos_full"] / full_metrics["mean_pos_full"]) ** 2
)
full_metrics["adi_full"] = (
    full_metrics["n_weeks_total"]
    / full_metrics["n_positive_weeks_full"].replace(0, np.nan)
)

conditions_full = [
    (full_metrics["adi_full"] <  ADI_THRESHOLD) & (full_metrics["cv2_full"] <  CV2_THRESHOLD),
    (full_metrics["adi_full"] <  ADI_THRESHOLD) & (full_metrics["cv2_full"] >= CV2_THRESHOLD),
    (full_metrics["adi_full"] >= ADI_THRESHOLD) & (full_metrics["cv2_full"] <  CV2_THRESHOLD),
    (full_metrics["adi_full"] >= ADI_THRESHOLD) & (full_metrics["cv2_full"] >= CV2_THRESHOLD),
]
full_metrics["intermittency_class_full"] = np.select(
    conditions_full, CLASS_ORDER, default=None
)
full_metrics["intermittency_class_full"] = (
    full_metrics["intermittency_class_full"].replace({None: np.nan})
)



# %%
# Trailing-104w subset
df_long = df[df["fiscal_year_month"].isin(trailing_104_months)].copy()
_df_long = df_long.assign(_is_zero=df_long["y"] == 0, _is_positive=df_long["y"] > 0)


counts_long = _df_long.groupby("unique_id", as_index=False).agg(
    n_weeks_long=("y", "count"),
    n_zero_weeks_long=("_is_zero", "sum"),
    n_positive_weeks_long=("_is_positive", "sum"),
)

pos_stats_long = (
    df_long[df_long["y"] > 0]
    .groupby("unique_id", as_index=False)
    .agg(
        mean_pos_long=("y", "mean"),
        std_pos_long=("y", "std"),
    )
)

metrics_long = counts_long.merge(pos_stats_long, on="unique_id", how="left")
metrics_long["frac_zero_weeks_long"] = (
    metrics_long["n_zero_weeks_long"] / metrics_long["n_weeks_long"]
)
metrics_long["cv2_long"] = (
    (metrics_long["std_pos_long"] / metrics_long["mean_pos_long"]) ** 2
)
metrics_long["adi_long"] = (
    metrics_long["n_weeks_long"]
    / metrics_long["n_positive_weeks_long"].replace(0, np.nan)
)

conditions_long = [
    (metrics_long["adi_long"] <  ADI_THRESHOLD) & (metrics_long["cv2_long"] <  CV2_THRESHOLD),
    (metrics_long["adi_long"] <  ADI_THRESHOLD) & (metrics_long["cv2_long"] >= CV2_THRESHOLD),
    (metrics_long["adi_long"] >= ADI_THRESHOLD) & (metrics_long["cv2_long"] <  CV2_THRESHOLD),
    (metrics_long["adi_long"] >= ADI_THRESHOLD) & (metrics_long["cv2_long"] >= CV2_THRESHOLD),
]
metrics_long["intermittency_class_long"] = np.select(
    conditions_long, CLASS_ORDER, default=None
)

metrics_long["intermittency_class_long"] = (
    metrics_long["intermittency_class_long"].replace({None: np.nan})
)

# %%
# Merge into summary_df
full_cols = [
    "unique_id", "n_zero_weeks_full", "n_positive_weeks_full", "frac_zero_weeks_full",
    "mean_pos_full", "std_pos_full", "cv2_full", "adi_full", "intermittency_class_full",
]
cols_long = [
    "unique_id", "n_weeks_long", "n_zero_weeks_long", "n_positive_weeks_long",
    "frac_zero_weeks_long", "mean_pos_long", "std_pos_long", "cv2_long", "adi_long",
    "intermittency_class_long",
]
summary_df = (
    summary_df
    .merge(full_metrics[full_cols], on="unique_id", how="left")
    .merge(metrics_long[cols_long], on="unique_id", how="left")
)


# %%
# Shared: discrete color map and marker types for intermittency classes
CLASS_COLOR_MAP = {
    "Smooth":       "#1f77b4",
    "Erratic":      "#ff7f0e",
    "Intermittent": "#2ca02c",
    "Lumpy":        "#d62728",
}

CLASS_MARKER_MAP_MATPLOTLIB = {
    "Smooth":       "<",
    "Erratic":      "s",
    "Intermittent": "o",
    "Lumpy":        "D",
}

CLASS_MARKER_MAP_PLOTLY = {
    "Smooth":       "triangle-left",
    "Erratic":      "square",
    "Intermittent": "circle",
    "Lumpy":        "diamond",
}

# %% [markdown]
# ## Subsec: ADI vs. $CV^2$

# %%
# ADI vs. CV2
fig_adi_cv2 = make_subplots(
    rows=1, cols=2,
    subplot_titles=["Full History", "Trailing 104 Weeks"],
)

sub_full_adi_cv2 = summary_df.dropna(subset=["adi_full", "cv2_full", "mean_pos_short"])
hover_full_adi_cv2 = sub_full_adi_cv2[
    ["unique_id", "adi_full", "cv2_full", "intermittency_class_full", "mean_pos_short"]
].values

fig_adi_cv2.add_trace(
    go.Scatter(
        x=sub_full_adi_cv2["adi_full"],
        y=sub_full_adi_cv2["cv2_full"],
        mode="markers",
        marker=dict(color=sub_full_adi_cv2["mean_pos_short"], coloraxis="coloraxis", size=6),
        customdata=hover_full_adi_cv2,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "ADI: %{customdata[1]:.2f}<br>"
            "CV²: %{customdata[2]:.2f}<br>"
            "Class: %{customdata[3]}<br>"
            "Mean pos revenue (52w): %{customdata[4]:,.0f}<extra></extra>"
        ),
        showlegend=False,
    ),
    row=1, col=1,
)

sub_long_adi_cv2 = summary_df.dropna(subset=["adi_long", "cv2_long", "mean_pos_short"])
hover_long_adi_cv2 = sub_long_adi_cv2[
    ["unique_id", "adi_long", "cv2_long", "intermittency_class_long", "mean_pos_short"]
].values

fig_adi_cv2.add_trace(
    go.Scatter(
        x=sub_long_adi_cv2["adi_long"],
        y=sub_long_adi_cv2["cv2_long"],
        mode="markers",
        marker=dict(color=sub_long_adi_cv2["mean_pos_short"], coloraxis="coloraxis", size=6),
        customdata=hover_long_adi_cv2,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "ADI: %{customdata[1]:.2f}<br>"
            "CV²: %{customdata[2]:.2f}<br>"
            "Class: %{customdata[3]}<br>"
            "Mean pos revenue (52w): %{customdata[4]:,.0f}<extra></extra>"
        ),
        showlegend=False,
    ),
    row=1, col=2,
)

for xref, yref in [("x", "y"), ("x2", "y2")]:
    fig_adi_cv2.add_shape(
        type="line",
        x0=ADI_THRESHOLD, x1=ADI_THRESHOLD, y0=0, y1=1,
        xref=xref, yref=f"{yref} domain",
        line=dict(dash="dash", color="gray", width=1),
    )
    fig_adi_cv2.add_shape(
        type="line",
        x0=0, x1=1, y0=CV2_THRESHOLD, y1=CV2_THRESHOLD,
        xref=f"{xref} domain", yref=yref,
        line=dict(dash="dash", color="gray", width=1),
    )

fig_adi_cv2.update_layout(
    title="ADI vs. CV² — Intermittency Classification",
    coloraxis=dict(
        colorscale="Viridis",
        colorbar=dict(
            title=dict(
                text="Mean positive weekly revenue (trailing 52w)",
                side="right",
            )
        ),
    ),
    height=500,
)
fig_adi_cv2.update_xaxes(title_text="ADI (average inter-demand interval)")
fig_adi_cv2.update_yaxes(title_text="CV²")
fig_adi_cv2.show()


# %%
sub_full_adi_cv2 = summary_df.dropna(subset=["adi_full", "cv2_full", "mean_pos_short"])
sub_long_adi_cv2  = summary_df.dropna(subset=["adi_long",  "cv2_long",  "mean_pos_short"])

all_color_vals = pd.concat([sub_full_adi_cv2["mean_pos_short"], sub_long_adi_cv2["mean_pos_short"]])
norm = plt.Normalize(vmin=all_color_vals.min(), vmax=all_color_vals.max())

fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

subplot_data_adi_cv2 = [
    (axes[0], sub_full_adi_cv2, "adi_full", "cv2_full", "Full History"),
    (axes[1], sub_long_adi_cv2,  "adi_long",  "cv2_long",  "Trailing 104 Weeks"),
]

for ax, sub, adi_col, cv2_col, title in subplot_data_adi_cv2:
    sc = ax.scatter(
        sub[adi_col], sub[cv2_col],
        c=sub["mean_pos_short"],
        cmap="viridis", norm=norm,
        s=30, alpha=0.8, edgecolors="none",
    )
    ax.axvline(ADI_THRESHOLD, color="gray", linestyle="--", linewidth=1)
    ax.axhline(CV2_THRESHOLD, color="gray", linestyle="--", linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("ADI (average inter-demand interval)")
    ax.set_ylabel("CV²")

fig.colorbar(sc, ax=axes, label="Mean positive weekly revenue (trailing 52w)")
fig.suptitle("ADI vs. CV² — Intermittency Classification")
plt.show()


# %% [markdown]
# ## Subsec: Mean Positive Revenue vs. Fraction Zero Weeks

# %%
fig_frac_zero = make_subplots(
    rows=1, cols=2,
    subplot_titles=["Full History", "Trailing 104 Weeks"],
)

subplot_configs_frac_zero = [
    (1, "frac_zero_weeks_full", "intermittency_class_full"),
    (2, "frac_zero_weeks_long", "intermittency_class_long"),
]

for col, y_col, class_col in subplot_configs_frac_zero:
    for cls in CLASS_ORDER:
        sub = summary_df[summary_df[class_col] == cls].dropna(subset=["mean_pos_short", y_col])
        hover_data = sub[["unique_id", "mean_pos_short", y_col, class_col]].values
        fig_frac_zero.add_trace(
            go.Scatter(
                x=sub["mean_pos_short"],
                y=sub[y_col],
                mode="markers",
                name=cls,
                legendgroup=cls,
                marker=dict(color=CLASS_COLOR_MAP[cls], symbol=CLASS_MARKER_MAP_PLOTLY[cls], size=6),
                customdata=hover_data,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Mean pos revenue (52w): %{customdata[1]:,.0f}<br>"
                    "Fraction zero weeks: %{customdata[2]:.2f}<br>"
                    "Class: %{customdata[3]}<extra></extra>"
                ),
                showlegend=(col == 1),
            ),
            row=1, col=col,
        )

fig_frac_zero.update_xaxes(type="log", title_text="Mean positive weekly revenue (trailing 52w, log scale)")
fig_frac_zero.update_yaxes(title_text="Fraction zero-revenue weeks")
fig_frac_zero.update_layout(title="Mean Positive Revenue vs. Fraction Zero Weeks", height=500)
fig_frac_zero.show()


# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

subplot_configs_frac_zero = [
    (axes[0], "frac_zero_weeks_full", "intermittency_class_full", "Full History"),
    (axes[1], "frac_zero_weeks_long",  "intermittency_class_long",  "Trailing 104 Weeks"),
]

for ax, y_col, class_col, title in subplot_configs_frac_zero:
    for cls in CLASS_ORDER:
        sub = summary_df[summary_df[class_col] == cls].dropna(subset=["mean_pos_short", y_col])
        ax.scatter(
            sub["mean_pos_short"], sub[y_col],
            color=CLASS_COLOR_MAP[cls],
            marker=CLASS_MARKER_MAP_MATPLOTLIB[cls],
            label=cls,
            s=30, alpha=0.8, edgecolors="none",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Mean positive weekly revenue (trailing 52w, log scale)")
    ax.set_ylabel("Fraction zero-revenue weeks")
    ax.set_title(title)
    ax.grid(alpha=0.7)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=len(CLASS_ORDER), bbox_to_anchor=(0.5, -0.08))
fig.suptitle("Mean Positive Revenue vs. Fraction Zero Weeks")
plt.show()


# %% [markdown]
# ## Subsec: Mean Positive Revenue vs. ADI

# %%
fig_adi_vs_mean = make_subplots(
    rows=1, cols=2,
    subplot_titles=["Full History", "Trailing 104 Weeks"],
)

subplot_configs_adi = [
    (1, "adi_full", "intermittency_class_full"),
    (2, "adi_long",  "intermittency_class_long"),
]

for col, y_col, class_col in subplot_configs_adi:
    for cls in CLASS_ORDER:
        sub = summary_df[summary_df[class_col] == cls].dropna(subset=["mean_pos_short", y_col])
        hover_data = sub[["unique_id", "mean_pos_short", y_col, class_col]].values
        fig_adi_vs_mean.add_trace(
            go.Scatter(
                x=sub["mean_pos_short"],
                y=sub[y_col],
                mode="markers",
                name=cls,
                legendgroup=cls,
                marker=dict(color=CLASS_COLOR_MAP[cls], symbol=CLASS_MARKER_MAP_PLOTLY[cls], size=6),
                customdata=hover_data,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Mean pos revenue (52w): %{customdata[1]:,.0f}<br>"
                    "ADI: %{customdata[2]:.2f}<br>"
                    "Class: %{customdata[3]}<extra></extra>"
                ),
                showlegend=(col == 1),
            ),
            row=1, col=col,
        )

fig_adi_vs_mean.update_xaxes(type="log", title_text="Mean positive weekly revenue (trailing 52w, log scale)")
fig_adi_vs_mean.update_yaxes(title_text="ADI")
fig_adi_vs_mean.update_layout(title="Mean Positive Revenue vs. ADI", height=500)
fig_adi_vs_mean.show()


# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

subplot_configs_adi = [
    (axes[0], "adi_full", "intermittency_class_full", "Full History"),
    (axes[1], "adi_long",  "intermittency_class_long",  "Trailing 104 Weeks"),
]

for ax, y_col, class_col, title in subplot_configs_adi:
    for cls in CLASS_ORDER:
        sub = summary_df[summary_df[class_col] == cls].dropna(subset=["mean_pos_short", y_col])
        ax.scatter(
            sub["mean_pos_short"], sub[y_col],
            color=CLASS_COLOR_MAP[cls],
            marker=CLASS_MARKER_MAP_MATPLOTLIB[cls],
            label=cls,
            s=30, alpha=0.8, edgecolors="none",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Mean positive weekly revenue (trailing 52w, log scale)")
    ax.set_ylabel("ADI")
    ax.set_title(title)
    ax.grid(alpha=0.7)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=len(CLASS_ORDER), bbox_to_anchor=(0.5, -0.08))
fig.suptitle("Mean Positive Revenue vs. ADI")
plt.show()


# %% [markdown]
# # Section: Scale-variance metrics

# %% [markdown]
# ## Subsec: Extend summary_df with scale-variance metrics

# %%
# std on all y (full history)
scale_stats = df.groupby("unique_id", as_index=False).agg(
    std_y_full=("y", "std"),
)

# IQR: Q75 - Q25
q75 = df.groupby("unique_id")["y"].quantile(0.75)
q25 = df.groupby("unique_id")["y"].quantile(0.25)
iqr_df = (q75 - q25).rename("iqr_y_full").reset_index()

# MAD: median(abs(y - median(y))) per series
_df_mad = df.assign(_y_median=df.groupby("unique_id")["y"].transform("median"))
_df_mad = _df_mad.assign(_abs_dev=(_df_mad["y"] - _df_mad["_y_median"]).abs())
mad_stats = _df_mad.groupby("unique_id", as_index=False).agg(
    mad_y_full=("_abs_dev", "median"),
)

# Median of positive-only y (full history)
median_pos_stats = (
    df[df["y"] > 0]
    .groupby("unique_id", as_index=False)
    .agg(median_pos_full=("y", "median"))
)

# Merge into summary_df
summary_df = (
    summary_df
    .merge(scale_stats[["unique_id", "std_y_full"]], on="unique_id", how="left")
    .merge(iqr_df[["unique_id", "iqr_y_full"]], on="unique_id", how="left")
    .merge(mad_stats[["unique_id", "mad_y_full"]], on="unique_id", how="left")
    .merge(median_pos_stats[["unique_id", "median_pos_full"]], on="unique_id", how="left")
)

# median_pos_full is NaN (not 0) for series with no positive weeks — division propagates NaN naturally
summary_df["cv_std_full"] = summary_df["std_y_full"] / summary_df["mean_pos_short"]
summary_df["cv_iqr_full"] = summary_df["iqr_y_full"] / summary_df["median_pos_full"]


# %%
summary_df[["unique_id", "std_y_full", "iqr_y_full", "mad_y_full", "median_pos_full", "cv_std_full", "cv_iqr_full"]].describe()

# %%
summary_df[summary_df["cv_std_full"].isna()][["unique_id", "n_positive_weeks_long", "intermittency_class_long"]]


# %% [markdown]
# ## Subsec: Mean Positive Revenue vs. Absolute Variability

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

subplot_configs_abs_var = [
    (axes[0], "iqr_y_full", "IQR of weekly revenue"),
    (axes[1], "std_y_full", "Std of weekly revenue"),
]

for ax, y_col, title in subplot_configs_abs_var:
    for cls in CLASS_ORDER:
        sub = summary_df[summary_df["intermittency_class_full"] == cls].dropna(subset=["mean_pos_short", y_col])
        ax.scatter(
            sub["mean_pos_short"], sub[y_col],
            color=CLASS_COLOR_MAP[cls],
            marker=CLASS_MARKER_MAP_MATPLOTLIB[cls],
            label=cls,
            s=30, alpha=0.8, edgecolors="none",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean positive weekly revenue (trailing 52w, log scale)")
    ax.set_ylabel("Absolute variability (log scale)")
    ax.set_title(title)
    ax.grid(alpha=0.7)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=len(CLASS_ORDER), bbox_to_anchor=(0.5, -0.08))
fig.suptitle("Mean Positive Revenue vs. Absolute Variability")
plt.show()


# %%
fig_abs_variability = make_subplots(
    rows=1, cols=2,
    subplot_titles=["IQR of weekly revenue", "Std of weekly revenue"],
)

subplot_configs_abs_var = [
    (1, "iqr_y_full"),
    (2, "std_y_full"),
]

for col, y_col in subplot_configs_abs_var:
    for cls in CLASS_ORDER:
        sub = summary_df[summary_df["intermittency_class_full"] == cls].dropna(subset=["mean_pos_short", y_col])
        hover_data = sub[["unique_id", "mean_pos_short", y_col, "intermittency_class_full"]].values
        fig_abs_variability.add_trace(
            go.Scatter(
                x=sub["mean_pos_short"],
                y=sub[y_col],
                mode="markers",
                name=cls,
                legendgroup=cls,
                marker=dict(color=CLASS_COLOR_MAP[cls], symbol=CLASS_MARKER_MAP_PLOTLY[cls], size=6),
                customdata=hover_data,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Mean pos revenue (52w): %{customdata[1]:,.0f}<br>"
                    "Absolute variability: %{customdata[2]:,.0f}<br>"
                    "Class: %{customdata[3]}<extra></extra>"
                ),
                showlegend=(col == 1),
            ),
            row=1, col=col,
        )

fig_abs_variability.update_xaxes(type="log", title_text="Mean positive weekly revenue (trailing 52w, log scale)")
fig_abs_variability.update_yaxes(type="log", title_text="Absolute variability (log scale)")
fig_abs_variability.update_layout(title="Mean Positive Revenue vs. Absolute Variability", height=500)
fig_abs_variability.show()


# %% [markdown]
# ## Subsec: Mean Positive Revenue vs. Relative Variability

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

cv_std_p99 = summary_df["cv_std_full"].quantile(0.99)
cv_iqr_p99 = summary_df["cv_iqr_full"].quantile(0.99)

subplot_configs_rel_var = [
    (axes[0], "cv_std_full", "CV (std / mean positive, trailing 52w)",          cv_std_p99),
    (axes[1], "cv_iqr_full", "Robust CV (IQR / median positive, full history)", cv_iqr_p99),
]

for ax, y_col, title, p99 in subplot_configs_rel_var:
    for cls in CLASS_ORDER:
        sub = summary_df[summary_df["intermittency_class_full"] == cls].dropna(subset=["mean_pos_short", y_col])
        ax.scatter(
            sub["mean_pos_short"], sub[y_col],
            color=CLASS_COLOR_MAP[cls],
            marker=CLASS_MARKER_MAP_MATPLOTLIB[cls],
            label=cls,
            s=30, alpha=0.8, edgecolors="none",
        )
    ax.set_xscale("log")
    ax.set_ylim(0, p99)
    ax.set_xlabel("Mean positive weekly revenue (trailing 52w, log scale)")
    ax.set_ylabel("Relative variability")
    ax.set_title(title)
    ax.grid(alpha=0.7)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=len(CLASS_ORDER), bbox_to_anchor=(0.5, -0.08))
fig.suptitle("Mean Positive Revenue vs. Relative Variability")
plt.show()


# %%
fig_rel_variability = make_subplots(
    rows=1, cols=2,
    subplot_titles=[
        "CV (std / mean positive, trailing 52w)",
        "Robust CV (IQR / median positive, full history)",
    ],
)

subplot_configs_rel_var = [
    (1, "cv_std_full"),
    (2, "cv_iqr_full"),
]

for col, y_col in subplot_configs_rel_var:
    for cls in CLASS_ORDER:
        sub = summary_df[summary_df["intermittency_class_full"] == cls].dropna(subset=["mean_pos_short", y_col])
        hover_data = sub[["unique_id", "mean_pos_short", y_col, "intermittency_class_full"]].values
        fig_rel_variability.add_trace(
            go.Scatter(
                x=sub["mean_pos_short"],
                y=sub[y_col],
                mode="markers",
                name=cls,
                legendgroup=cls,
                marker=dict(color=CLASS_COLOR_MAP[cls], symbol=CLASS_MARKER_MAP_PLOTLY[cls], size=6),
                customdata=hover_data,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Mean pos revenue (52w): %{customdata[1]:,.0f}<br>"
                    "Relative variability: %{customdata[2]:.3f}<br>"
                    "Class: %{customdata[3]}<extra></extra>"
                ),
                showlegend=(col == 1),
            ),
            row=1, col=col,
        )

cv_std_p99 = summary_df["cv_std_full"].quantile(0.99)
cv_iqr_p99 = summary_df["cv_iqr_full"].quantile(0.99)

fig_rel_variability.update_xaxes(type="log", title_text="Mean positive weekly revenue (trailing 52w, log scale)")
fig_rel_variability.update_yaxes(title_text="Relative variability")
fig_rel_variability.update_layout(
    title="Mean Positive Revenue vs. Relative Variability",
    height=500,
    yaxis=dict(range=[0, cv_std_p99]),
    yaxis2=dict(range=[0, cv_iqr_p99]),
)
fig_rel_variability.show()


# %% [markdown]
# # Section: Remaining month versus dispersion 

# %%
mtd_df["y_remaining"] = mtd_df["y_final_month"] - mtd_df["y_mtd"]
mtd_df_core = mtd_df[mtd_df["y_final_month"] >= mtd_df["core_threshold"]].copy()

# %%
_grp = ["bucket", "weeks_in_month", "origin_week"]


def compute_dispersion_ratios(data, grp_cols):
    rem_q75 = data.groupby(grp_cols, observed=True)["y_remaining"].quantile(0.75)
    rem_q25 = data.groupby(grp_cols, observed=True)["y_remaining"].quantile(0.25)
    rem_q90 = data.groupby(grp_cols, observed=True)["y_remaining"].quantile(0.90)
    rem_q10 = data.groupby(grp_cols, observed=True)["y_remaining"].quantile(0.10)
    fin_q75 = data.groupby(grp_cols, observed=True)["y_final_month"].quantile(0.75)
    fin_q25 = data.groupby(grp_cols, observed=True)["y_final_month"].quantile(0.25)
    fin_q90 = data.groupby(grp_cols, observed=True)["y_final_month"].quantile(0.90)
    fin_q10 = data.groupby(grp_cols, observed=True)["y_final_month"].quantile(0.10)

    quantile_df = pd.concat(
        [
            (rem_q75 - rem_q25).rename("iqr_rem"),
            (rem_q90 - rem_q10).rename("tail_rem"),
            (fin_q75 - fin_q25).rename("iqr_fin"),
            (fin_q90 - fin_q10).rename("tail_fin"),
        ],
        axis=1,
    ).reset_index()

    _data = data.assign(
        _rem_median=data.groupby(grp_cols, observed=True)["y_remaining"].transform("median"),
        _fin_median=data.groupby(grp_cols, observed=True)["y_final_month"].transform("median"),
    )
    _data = _data.assign(
        _rem_abs_dev=(_data["y_remaining"] - _data["_rem_median"]).abs(),
        _fin_abs_dev=(_data["y_final_month"] - _data["_fin_median"]).abs(),
    )
    mad_df = _data.groupby(grp_cols, as_index=False, observed=True).agg(
        mad_rem=("_rem_abs_dev", "median"),
        mad_fin=("_fin_abs_dev", "median"),
    )

    out = quantile_df.merge(mad_df, on=grp_cols)
    out["iqr_ratio"] = out["iqr_rem"] / out["iqr_fin"]
    out["mad_ratio"] = out["mad_rem"] / out["mad_fin"]
    out["tail_spread_ratio"] = out["tail_rem"] / out["tail_fin"]

    return out[grp_cols + ["iqr_ratio", "mad_ratio", "tail_spread_ratio"]]


ratio_df = compute_dispersion_ratios(mtd_df_core, _grp)

mtd_df_long_core = mtd_df[
    mtd_df["fiscal_year_month"].isin(trailing_104_months)
    & (mtd_df["y_final_month"] >= mtd_df["core_threshold"])
].copy()

ratio_df_long = compute_dispersion_ratios(mtd_df_long_core, _grp)


# %%
ratio_df.shape

# %%
ratio_df_long.shape

# %%
ratio_df["weeks_in_month"].value_counts()

# %%
WEEK_LENGTHS = [4, 5]
ROW_LABELS = {4: "4-week months", 5: "5-week months"}
RATIO_LINE_SPECS = [
    ("iqr_ratio", "IQR ratio", "tab:blue"),
    ("mad_ratio", "MAD ratio", "tab:orange"),
    ("tail_spread_ratio", "Tail-spread ratio (P90–P10)", "tab:green"),
]


# %%
def plot_dispersion_ratios(ratio_data, title):
    fig, axes = plt.subplots(len(WEEK_LENGTHS), 5, figsize=(18, 7), constrained_layout=True)

    for row_idx, n_weeks in enumerate(WEEK_LENGTHS):
        for col_idx, bucket in enumerate(BUCKETS):
            ax = axes[row_idx, col_idx]
            sub = ratio_data[
                (ratio_data["weeks_in_month"] == n_weeks)
                & (ratio_data["bucket"] == bucket)
            ].sort_values("origin_week")

            for col_name, label, color in RATIO_LINE_SPECS:
                ax.plot(sub["origin_week"], sub[col_name], marker="o", color=color, label=label)

            ax.axhline(y=1.0, linestyle="--", color="black", linewidth=0.8, alpha=0.6)
            ax.set_xticks(range(1, n_weeks + 1))

            if row_idx == 0:
                ax.set_title(bucket)
            if col_idx == 0:
                ax.set_ylabel(ROW_LABELS[n_weeks])

    fig.supxlabel("Forecast origin (weeks of actuals)")
    fig.supylabel("Remaining / Total dispersion ratio")
    fig.suptitle(title)

    legend_handles = [
        plt.Line2D([0], [0], color=color, marker="o", label=label)
        for col_name, label, color in RATIO_LINE_SPECS
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncols=3, bbox_to_anchor=(0.5, -0.05))

    plt.show()


plot_dispersion_ratios(
    ratio_df,
    "Target Dispersion Ratio by Formulation and Forecast Origin — Full History",
)
plot_dispersion_ratios(
    ratio_df_long,
    "Target Dispersion Ratio by Formulation and Forecast Origin — Trailing 104 Weeks",
)


# %% [markdown]
# # Section: Time Series Length Histogram

# %%
hist_df = summary_df.dropna(subset=["time_series_length_weeks"])

fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

ax.hist(
    hist_df["time_series_length_weeks"],
    bins=range(
        int(hist_df["time_series_length_weeks"].min()),
        int(hist_df["time_series_length_weeks"].max()) + 2,
    ),
    color="silver",
    edgecolor="silver",
)
ax.set_xlabel("Time series length (weeks)")
ax.set_ylabel("Number of series")
ax.set_title("Distribution of Time Series Lengths")

plt.show()
