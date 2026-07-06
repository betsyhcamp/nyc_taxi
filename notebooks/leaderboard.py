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
        "model_section": composed_cfg["model"],
        "metrics_df": pd.read_parquet(f"{uri}metrics.parquet"),
        "agg_metrics_df": pd.read_parquet(f"{uri}aggregated_metrics.parquet"),
        "monthly_metrics_df": pd.read_parquet(f"{uri}monthly_metrics.parquet"),
    })

print(f"Loaded {len(runs)} runs: {[r['model'] for r in runs]}")

# %%
# monthly_metrics_df uses "metric_value"/"fold_id" — normalize when metric PR updates backtest_weekly

summary_rows = []
for run in runs:
    row = {}
    row["model"] = run["model"]
    row["callable"] = run["model_section"]["callable"]
    row["hyperparameters"] = run["model_section"].get("hyperparameters", {})
    row["weekly_error"] = run["metrics_df"]["value"].mean()
    row["pred_remaining_error"]=run["agg_metrics_df"]["value"].mean()
    row["monthly_error"] = run["monthly_metrics_df"]["metric_value"].mean()
    row["metric_name"] =run["metrics_df"]['metric_name'].unique()[0]
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
display(summary_df)

# %%
fold_rows = []
for run in runs:
    weekly_per_fold = run["metrics_df"].groupby("fold")["value"].mean()
    agg_per_fold = run["agg_metrics_df"].groupby("fold")["value"].mean()
    monthly_per_fold = run["monthly_metrics_df"].set_index("fold_id")["metric_value"]

    for fold in weekly_per_fold.index:
        fold_rows.append({
            "model": run["model"],
            "fold": fold,
            "weekly_error": weekly_per_fold[fold],
            "pred_remaining_error": agg_per_fold[fold],
            "monthly_error": monthly_per_fold[fold],
        })

fold_df = pd.DataFrame(fold_rows).sort_values(["model", "fold"]).reset_index(drop=True)
display(fold_df)
