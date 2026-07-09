# %%
import sys
import pandas as pd
import yaml
import fsspec

from fcstnyctaxi.lib.utils import get_project_root_dir

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
        "model_section": composed_cfg["model"],
        "metrics_df": pd.read_parquet(f"{uri}metrics.parquet"),
        "monthly_series_df": pd.read_parquet(f"{uri}monthly_series.parquet"),
        "calendar_df": pd.read_parquet(f"{uri}fiscal_calendar.parquet")[["ds", "fiscal_year_month"]].drop_duplicates(),
    })

print(f"Loaded {len(runs)} runs: {[r['model'] for r in runs]}")

# %%
all_monthly_series = pd.concat(
    [run["monthly_series_df"].assign(model=run["model"]) for run in runs],
    ignore_index=True
)

all_monthly_series["abs_error"] = (
    all_monthly_series["monthly_forecast"]
    - all_monthly_series["actual_monthly_total"]
).abs()

# %%
all_monthly_series.head()

# %%
calendar_df = runs[0]["calendar_df"]
calendar_df = calendar_df.rename(
    columns={"ds": "forecast_origin_date", "fiscal_year_month": "origin_fiscal_month"}
)

# %%
calendar_df["forecast_origin_date"] = pd.to_datetime(
    calendar_df["forecast_origin_date"]
)
all_monthly_series["forecast_origin_date"] = pd.to_datetime(
    all_monthly_series["forecast_origin_date"]
)

# %%
all_monthly_series =all_monthly_series.merge(
    calendar_df,
    on="forecast_origin_date",
    how="left"
)

# %%
end = all_monthly_series["predicted_fiscal_year_month"]
start = all_monthly_series["origin_fiscal_month"]
all_monthly_series["target"] = (
    (end // 100 - start // 100) * 12 + (end % 100 - start % 100)
)


# %%
all_monthly_series.head()

# %%
period_breakdown = (
    all_monthly_series
    .groupby(["model", "predicted_fiscal_year_month", "target", "tier"], observed=True)["abs_error"]
    .mean()
    .reset_index()
    .rename(columns={"abs_error": "mae"})
    .sort_values(["model", "predicted_fiscal_year_month", "target", "tier"])
)
period_breakdown

# %%
fold_breakdown = (
    all_monthly_series
    .groupby(["model", "forecast_origin_date", "predicted_fiscal_year_month", "target", "tier"], observed=True)["abs_error"]
    .mean()
    .reset_index()
    .rename(columns={"abs_error": "mae"})
    .sort_values(["model", "forecast_origin_date", "predicted_fiscal_year_month", "target", "tier"]))
fold_breakdown

# %%
global_monthly_mae = (
    all_monthly_series
    .groupby("model")["abs_error"]
    .mean()
    .rename("global_monthly_mae")
)

# %%

tier_monthly_mae = (
    all_monthly_series.groupby(["model", "tier"], observed=True)["abs_error"]
    .mean()
    .unstack("tier")
    .add_prefix("tier_monthly_mae_")
)

# %%
global_avg_weekly_mae = pd.Series(
    {run["model"]: run["metrics_df"]["value"].mean() for run in runs},
    name="global_avg_weekly_mae",
)
global_avg_weekly_mae.index.name = "model"

# %%
summary_df = pd.concat(
    [global_monthly_mae, tier_monthly_mae, global_avg_weekly_mae],
    axis=1).reset_index()
display(summary_df)
