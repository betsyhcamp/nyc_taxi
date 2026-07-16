# %%
import sys
import pandas as pd
import numpy as np
import yaml
import fsspec

from fcstnyctaxi.lib.utils import get_project_root_dir
from fcstnyctaxi.lib.fold_metrics import (
    compute_wrmae_pooled,
    compute_wrmae_per_series,
    compute_wape,
    compute_weighted_signed_bias,
    compute_signed_bias_pooled,
    compute_signed_bias_per_series,
)

pd.options.display.max_columns = 40
pd.options.display.max_rows = 100

# %%
project_root = get_project_root_dir()
sys.path.insert(0, str(project_root))

leaderboard_runs_path = (
    project_root / "notebooks" / "backtest_configs" / "leaderboard_runs.yaml"
)
with open(leaderboard_runs_path) as f:
    leaderboard_runs = yaml.safe_load(f)

print(leaderboard_runs)


# %%
def resolve_sidecar_uri(entry, default_base_uri) -> str:
    if "sidecar_uri" in entry:
        return entry["sidecar_uri"]

    return f"{default_base_uri}{entry['sidecar_id']}/"


# %%
runs = []
for entry in leaderboard_runs["runs"]:
    uri = resolve_sidecar_uri(entry, leaderboard_runs["default_base_uri"])

    try:
        with fsspec.open(f"{uri}composed_config.yaml", "r") as f:
            composed_cfg = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Sidecar not found for model '{entry['model']}': {uri}"
        )

    runs.append({
        "model": entry["model"],
        "framing": entry["framing"],
        "is_benchmark": entry.get("benchmark", False),
        "model_section": composed_cfg["model"],
        "metrics_df": pd.read_parquet(f"{uri}metrics.parquet"),
        "monthly_series_df": pd.read_parquet(f"{uri}monthly_series.parquet"),
        "calendar_df": pd.read_parquet(f"{uri}fiscal_calendar.parquet")[["ds", "fiscal_year_month"]].drop_duplicates(),
    })

benchmark_run = next(r for r in runs if r["is_benchmark"])
challenger_runs = [r for r in runs if not r["is_benchmark"]]
print(f"Benchmark: {benchmark_run['model']}")
print(f"Challengers: {[r['model'] for r in challenger_runs]}")

# %%
benchmark_monthly_series = benchmark_run["monthly_series_df"]

all_monthly_series = pd.concat(
    [run["monthly_series_df"].assign(model=run["model"]) for run in challenger_runs],
    ignore_index=True
)

# %%
all_monthly_series.head()

# %%
calendar_df = benchmark_run["calendar_df"] # benchmark, challenger all have the same cal
calendar_df = calendar_df.rename(
    columns={
        "ds": "forecast_origin_date", "fiscal_year_month": "origin_fiscal_year_month"
        }
)

# %%
calendar_df["forecast_origin_date"] = pd.to_datetime(
    calendar_df["forecast_origin_date"]
)
all_monthly_series["forecast_origin_date"] = pd.to_datetime(
    all_monthly_series["forecast_origin_date"]
)

benchmark_monthly_series["forecast_origin_date"] = pd.to_datetime(
    benchmark_monthly_series["forecast_origin_date"]
)

# %%
all_monthly_series =all_monthly_series.merge(
    calendar_df,
    on="forecast_origin_date",
    how="left"
)

benchmark_monthly_series =benchmark_monthly_series.merge(
    calendar_df,
    on="forecast_origin_date",
    how="left"
)


# %%
def _derive_horizon_label(df):
    origin = df["origin_fiscal_year_month"]
    frac = df["origin_month_fraction_elapsed"]
    target = df["predicted_fiscal_year_month"]

    year = origin // 100
    month = origin % 100
    prev_fiscal_year_month = pd.Series(
        np.where(month == 1, (year - 1) * 100 + 12, origin - 1),
        index=df.index,
    )
    last_completed = pd.Series(
        np.where(frac == 1.0, origin, prev_fiscal_year_month),
        index=df.index,
    )
    month_difference = (
        (target // 100 - last_completed // 100) * 12 
        + (target % 100 - last_completed % 100)
    )
    return "horizon_" + month_difference.astype(str)

all_monthly_series["horizon"] = _derive_horizon_label(all_monthly_series)
benchmark_monthly_series["horizon"] = _derive_horizon_label(benchmark_monthly_series)


# %%
all_monthly_series.head()

# %%
all_monthly_series[["forecast_origin_date","origin_fiscal_year_month",	"predicted_fiscal_year_month", "horizon"]].drop_duplicates().sort_values(by=["forecast_origin_date","origin_fiscal_year_month",	"predicted_fiscal_year_month", "horizon"])

# %%
tiers = all_monthly_series["tier"].cat.categories

# %%
summary_keys = list(
    all_monthly_series[["model", "horizon"]]
    .drop_duplicates()
    .sort_values(["model", "horizon"])
    .itertuples(index=False)
)


rows = []
for keys in summary_keys:
    ch = all_monthly_series[
        (all_monthly_series["model"] == keys.model) &
        (all_monthly_series["horizon"] == keys.horizon)
    ]
    bm = benchmark_monthly_series[benchmark_monthly_series["horizon"] == keys.horizon]

    row = {"model": keys.model, "horizon": keys.horizon}

    for tier_label, tier_val in [("global", None)] + [(t, t) for t in tiers]:
        row[f"wrmae_pooled_{tier_label}"] = compute_wrmae_pooled(ch, bm, tier=tier_val)
        row[f"wrmae_per_series_{tier_label}"] = compute_wrmae_per_series(
            ch, bm, tier=tier_val
        )
        row[f"wape_{tier_label}"] = compute_wape(ch, tier=tier_val)
        row[f"wsb_{tier_label}"] = compute_weighted_signed_bias(ch, tier=tier_val)

    rows.append(row)

summary_df = pd.DataFrame(rows)


# %%
display(summary_df)

# %%
period_keys = list(
    all_monthly_series[["model", "horizon", "predicted_fiscal_year_month"]]
    .drop_duplicates()
    .sort_values(["model", "predicted_fiscal_year_month", "horizon"])
    .itertuples(index=False)
)

rows = []
for keys in period_keys:
    ch = all_monthly_series[
        (all_monthly_series["model"] == keys.model)
        & (all_monthly_series["horizon"] == keys.horizon)
        & (
            all_monthly_series["predicted_fiscal_year_month"] 
            == keys.predicted_fiscal_year_month
        )
    ]
    bm = benchmark_monthly_series[
        (benchmark_monthly_series["horizon"] == keys.horizon)
        & (
            benchmark_monthly_series["predicted_fiscal_year_month"] 
             == keys.predicted_fiscal_year_month
        )
    ]

    row = {
        "model": keys.model, 
        "horizon": keys.horizon, 
        "predicted_fiscal_year_month": keys.predicted_fiscal_year_month
    }

    for tier_label, tier_val in [("global", None)] + [(t, t) for t in tiers]:
        row[f"wrmae_pooled_{tier_label}"] = compute_wrmae_pooled(ch, bm, tier=tier_val)
        row[f"wrmae_per_series_{tier_label}"] = compute_wrmae_per_series(
            ch, bm, tier=tier_val
        )
        row[f"wape_{tier_label}"] = compute_wape(ch, tier=tier_val)
        row[f"wsb_{tier_label}"] = compute_weighted_signed_bias(ch, tier=tier_val)
        row[f"signed_bias_pooled_{tier_label}"] = compute_signed_bias_pooled(
            ch, tier=tier_val
        )
        row[f"signed_bias_per_series_{tier_label}"] = compute_signed_bias_per_series(
            ch, tier=tier_val
        )
    rows.append(row)

period_breakdown_df = pd.DataFrame(rows)

# %%
display(period_breakdown_df)

# %%
fold_keys = list(
    all_monthly_series[
        ["model", "horizon", "forecast_origin_date", "predicted_fiscal_year_month"]
        ]
    .drop_duplicates()
    .sort_values(
        ["model", "forecast_origin_date", "predicted_fiscal_year_month", "horizon"]
        )
    .itertuples(index=False)
)

rows = []
for keys in fold_keys:
    ch = all_monthly_series[
        (all_monthly_series["model"] == keys.model)
        & (all_monthly_series["horizon"] == keys.horizon)
        & (
            all_monthly_series["predicted_fiscal_year_month"] 
            == keys.predicted_fiscal_year_month
            )
        & (all_monthly_series["forecast_origin_date"] == keys.forecast_origin_date)
    ]
    bm = benchmark_monthly_series[
        (benchmark_monthly_series["horizon"] == keys.horizon)
        & (
            benchmark_monthly_series["predicted_fiscal_year_month"] 
            == keys.predicted_fiscal_year_month
            )
        & (
            benchmark_monthly_series["forecast_origin_date"] 
            == keys.forecast_origin_date
            )
    ]

    row = {
        "model": keys.model, 
        "horizon": keys.horizon, 
        "predicted_fiscal_year_month": keys.predicted_fiscal_year_month,
        "forecast_origin_date": keys.forecast_origin_date
    }

    for tier_label, tier_val in [("global", None)] + [(t, t) for t in tiers]:
        row[f"wrmae_pooled_{tier_label}"] = compute_wrmae_pooled(ch, bm, tier=tier_val)
        row[f"wrmae_per_series_{tier_label}"] = compute_wrmae_per_series(
            ch, bm, tier=tier_val
        )
        row[f"wape_{tier_label}"] = compute_wape(ch, tier=tier_val)
        row[f"wsb_{tier_label}"] = compute_weighted_signed_bias(ch, tier=tier_val)
        row[f"signed_bias_pooled_{tier_label}"] = compute_signed_bias_pooled(
            ch, tier=tier_val
        )
        row[f"signed_bias_per_series_{tier_label}"] = compute_signed_bias_per_series(
            ch, tier=tier_val
        )
    rows.append(row)

fold_breakdown_df = pd.DataFrame(rows)

# %%
display(fold_breakdown_df)

# %%
print(all_monthly_series.shape)
print(benchmark_monthly_series.shape)

# %%
ch_cols = [
    "forecast_origin_date", "predicted_fiscal_year_month", "unique_id",
    "model", "horizon", "tier", "series_weight",
    "monthly_forecast", "actual_monthly_total",
]
bm_cols = [
    "forecast_origin_date", "predicted_fiscal_year_month", "unique_id",
    "monthly_forecast",
]

per_series_df = (
    all_monthly_series[ch_cols]
    .merge(
        benchmark_monthly_series[bm_cols],
        on = ["forecast_origin_date", "predicted_fiscal_year_month", "unique_id"],
        suffixes=("_ch", "_bm"),
        how="inner",
        )
)
per_series_df["relative_mae"] = (
    (per_series_df["monthly_forecast_ch"]-per_series_df["actual_monthly_total"]).abs()
    / (per_series_df["monthly_forecast_bm"]-per_series_df["actual_monthly_total"]).abs()
)

per_series_df["signed_relative_bias"] = (
    (per_series_df["monthly_forecast_ch"] - per_series_df["actual_monthly_total"])
    / per_series_df["actual_monthly_total"].abs()
)


# %%
per_series_df.shape

# %%
(per_series_df
 .query("model == 'autoets' and horizon == 'horizon_2' "
        "and predicted_fiscal_year_month == 202506 "
        "and forecast_origin_date == '2025-04-27' "
        "and tier == 'high'")
 .sort_values("relative_mae", ascending=False)
 [["unique_id", "series_weight", "relative_mae", "monthly_forecast_ch", "monthly_forecast_bm", "actual_monthly_total"]]
)

# %%
(71514.121582-68641)/(68660.0 - 68641)

# %%
