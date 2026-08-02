# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: nyc_taxi (3.12.9.final.0)
#     language: python
#     name: python3
# ---

# %%
import sys
import json
import yaml
import fsspec
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from tsbricks.backtesting.schema import ModelConfig
from tsbricks.runner import dynamic_import
from tsbricks.blocks.metadata import get_git_hash, get_uv_lock_info

from fcstnyctaxi.lib.calibration import (
    calibrate_n_estimators,
    most_parsimonious_n_estimators
)
from fcstnyctaxi.lib.monthly_aggregation import compute_actual_monthly_totals
from fcstnyctaxi.lib.period_utils import generate_origins_for_periods
from fcstnyctaxi.lib.io import write_text_to_gcs
from fcstnyctaxi.lib.utils import get_project_root_dir, generate_run_id
from fcstnyctaxi.lib.config_utils import save_config

# %%
project_root = get_project_root_dir()
sys.path.insert(0, str(project_root))

run_config_path = project_root / "notebooks" / "backtest_configs" / "run_config.yaml"
run_cfg = yaml.safe_load(run_config_path.read_text())
calibration_config_path = project_root / "notebooks" / "backtest_configs" / "calibration_config.yaml"
cal_cfg = yaml.safe_load(calibration_config_path.read_text())

model_cfg_path = project_root / cal_cfg["model"]
model_cfg_dict = yaml.safe_load(model_cfg_path.read_text())["model"]
model_cfg = ModelConfig(**model_cfg_dict)

sidecar_dir = generate_run_id()
sidecar_uri = (
    f"{run_cfg['project']['gcs_bucket']}/dev/backtests/calibrate_n_estimators/{sidecar_dir}/"
)

# %%
timeseries_uri = (
    f"{run_cfg['project']['gcs_bucket']}/{run_cfg['project']['time_series_uri']}"
)
ts_df = pd.read_parquet(timeseries_uri)

calendar_uri = (
    f"{run_cfg['project']['gcs_bucket']}/{run_cfg['project']['fiscal_calendar_uri']}"
)
calendar_df = pd.read_parquet(calendar_uri)

# %%
# Expand monthly calibration_periods.start_months into the exact weekly
# {origin, horizon} list
calibration_periods = cal_cfg["calibration_periods"]
calibration_origin_pairs = generate_origins_for_periods(
    start_months=calibration_periods["start_months"],
    forecast_horizon_months=calibration_periods["forecast_horizon_months"],
    calendar_df=calendar_df,
    calendar_time_col="ds",
)
print(f"Generated {len(calibration_origin_pairs)} calibration origins")
calibration_origin_pairs

# %%
# Leakage gate 
backtest_cfg_path = project_root / run_cfg["configs"]["backtest_config"]
backtest_cfg = yaml.safe_load(backtest_cfg_path.read_text())
# Get forecast horizon fiscal months of calibration origins. Require no overlap
# between horizons in calibration w/ what is used in read backtest cross-evaluation
period_col = backtest_cfg["aggregation"]["period_col"]
month_by_ds = calendar_df.set_index("ds")[period_col]
calibration_target_months = {
    month_by_ds[week]
    for pair in calibration_origin_pairs
    for week in calendar_df.loc[
        calendar_df["ds"] > pd.Timestamp(pair["origin"]), "ds"
    ].head(pair["horizon"])
}

# Derive calibration boundary by reading the backtest's own start_months, so
# leakage gate can't become stale when the backtest window is modified 
backtest_start_months = backtest_cfg["evaluation_periods"]["start_months"]

# Raise exception if leakage gate doesn't pass
earliest_backtest_target_month = min(backtest_start_months)
latest_calibration_target_month = max(calibration_target_months)
if latest_calibration_target_month >= earliest_backtest_target_month:
    raise ValueError(
        "Leakage gate failed: latest calibration target month "
        f"{latest_calibration_target_month} is not strictly before the earliest " 
        f"backtest target month {earliest_backtest_target_month}"
)

# %%
grid_cfg = cal_cfg["n_estimators_grid"]
n_estimators_grid = list(
    range(
        grid_cfg["step"], # grid start
        grid_cfg["ceiling"]+1, # grid end inclusive of ceiling
        grid_cfg["step"] # grid step size
    )
)

set_truncation_iteration = dynamic_import(cal_cfg["truncation_adapter"])

# %%
data = backtest_cfg["data"]
actual_monthly_df = compute_actual_monthly_totals(
    ts_df = ts_df, 
    calendar_df = calendar_df, 
    period_col = period_col, 
    time_col = data["date_col"], 
    id_col = data["id_col"], 
    target_col = data["target_col"]
)

# %%
calibration_origins = [
    (pair["origin"], pair["horizon"]) for pair in calibration_origin_pairs
]

# %%
scores_df = calibrate_n_estimators(
    ts_df=ts_df,
    calendar_df=calendar_df,
    actual_monthly_df=actual_monthly_df,
    model_config=model_cfg,
    n_estimators_grid=n_estimators_grid,
    calibration_origins=calibration_origins,
    set_truncation_iteration=set_truncation_iteration,
)

# %%
sel_cfg = cal_cfg["selection"]
rows = []
for (origin, horizon), grp in scores_df.groupby(["origin", "horizon"]):
    curve = grp.set_index("n_estimators")["score"].sort_index() # index = n_estimators
    recommended_n_estimators = most_parsimonious_n_estimators(
        curve, smoothing_window=sel_cfg["smoothing_window"], epsilon=sel_cfg["epsilon"]
    )
    rows.append(
        {
            "origin": origin,
            "horizon": horizon,
            "recommended_n_estimators": recommended_n_estimators
        }
    )
recommended_n_estimators_df = pd.DataFrame(rows)


# %%
inconclusive_tail_fraction = sel_cfg["inconclusive_tail_fraction"]
grid = sorted(n_estimators_grid)
tail_threshold = grid[-1] - inconclusive_tail_fraction * (grid[-1] - grid[0])
recommended_n_estimators_df["inconclusive"] = (
    recommended_n_estimators_df["recommended_n_estimators"] >= tail_threshold
)

# %%
report = (
    recommended_n_estimators_df
    .groupby("horizon")["inconclusive"]
    .agg(n_flagged="sum", n_origins="count")
)

recommended_n_estimators_df.groupby("horizon")["recommended_n_estimators"].describe()
# tight cluster -> trustworthy; wide -> origin-sensitive

# %%
# print example: "horizon_2: 5/9 inconclusive: widen grid and re-run" only where n_flagged > 0
for horizon, row in report.iterrows():
    if row["n_flagged"] > 0:
        print(
            f"{horizon}: {row['n_flagged']}/{row['n_origins']} inconclusive; "
            "widen grid and re-run."
        )


# %%
def gray(lightness: float) -> str:
    """lightness in [0,1]: 0=black, 1=white. Returns a plotly rgb() gray."""
    v = round(max(0.0, min(1.0, lightness)) * 255)   # clamp so a stray 1.2 can't error
    return f"rgb({v},{v},{v})"

bg_lightness = 0.96

horizons = sorted(scores_df["horizon"].unique())

# deterministic color per origin: same origin -> same color in every subplot 
origins = sorted(scores_df["origin"].unique()) # sorted() on timestamps/ISO dates = chronological
colors = px.colors.sample_colorscale(
    "Viridis", len(origins), low=0.10, high=0.98
)
color_by_origin = dict(zip(origins, colors))

fig = make_subplots(
    rows=len(horizons), cols=1, shared_xaxes=True,
    subplot_titles=horizons,
    vertical_spacing=0.05,
)
fig.update_layout(height=900, width=900, plot_bgcolor=gray(bg_lightness))

for i, h in enumerate(horizons, start=1):
    show_in_legend = (i == 1)                       # only subplot 1 populates the single legend
    sub = scores_df[scores_df["horizon"] == h]

    # (a) one faint line per origin — the overlaid curves
    for origin, g in sub.groupby("origin"):
        g = g.sort_values("n_estimators")
        fig.add_trace(
            go.Scatter(x=g["n_estimators"], y=g["score"],
                       name=str(origin), opacity=0.8,
                       showlegend=show_in_legend,
                       line=dict(color=color_by_origin[origin])),
            row=i, col=1,
        )

    # (b) average-across-origins reference line (thick/dashed, fixed distinct color)
    avg = sub.groupby("n_estimators")["score"].mean()
    fig.add_trace(
        go.Scatter(x=avg.index, y=avg.values, name="avg",
                   showlegend=show_in_legend,
                   line=dict(width=3, dash="dash", color="black")),
        row=i, col=1,
    )

    # (c) empirical-min marker per origin at its recommended n_estimators
    rec_h = recommended_n_estimators_df[recommended_n_estimators_df["horizon"] == h]
    for _, r in rec_h.iterrows():
        origin = r["origin"]
        k = r["recommended_n_estimators"]
        symbol = "star" if r["inconclusive"] else "circle"      # flagged -> star, else circle
        # raw score at that n_estimators for this origin/horizon (k is a grid value, so it exists)
        score_at_k = sub.loc[
            (sub["origin"] == origin) & (sub["n_estimators"] == k), "score"
        ].iloc[0]
        fig.add_trace(
            go.Scatter(
                x=[k], y=[score_at_k], mode="markers",
                showlegend=False,                      # markers stay out of the legend
                opacity=0.6,
                marker=dict(color=color_by_origin[origin], size=12, symbol=symbol,
                            line=dict(color="black", width=1)),
                hovertext=f"{origin}: rec={k}", hoverinfo="text",
            ),
            row=i, col=1,
        )

# axis titles are figure-level — set once, after the loop
fig.update_yaxes(title_text="score")
fig.update_xaxes(title_text="n_estimators", row=len(horizons), col=1)

fig.show()


# %%
# select n_estimators and run failure gate if too many inconclusive cases
percentile = sel_cfg["percentile"]
max_inconclusive_fraction = sel_cfg["max_inconclusive_fraction"]

# pool across BOTH origin and horizon
n_pairs = len(recommended_n_estimators_df)
n_inconclusive = int(recommended_n_estimators_df["inconclusive"].sum())
n_resolved = n_pairs - n_inconclusive
resolved = recommended_n_estimators_df.loc[~recommended_n_estimators_df["inconclusive"]]

inconclusive_fraction = n_inconclusive / n_pairs if n_pairs else 1.0
gate_passed = (not resolved.empty) and (inconclusive_fraction <= max_inconclusive_fraction)

if gate_passed:
    selected_n_estimators = int(
        np.percentile(resolved["recommended_n_estimators"], percentile, method="higher")
    )
else:
    selected_n_estimators = None  # recorded refusal, not a computed-over-nothing number


# %%
# pooled-overall reference points recorded in the decision artifact as crosscheck
pooled_avg_by_n_estimators = scores_df.groupby("n_estimators")["score"].mean()
best_score = float(pooled_avg_by_n_estimators.min())
best_n_estimators = int(pooled_avg_by_n_estimators.idxmin())
if selected_n_estimators is not None:
    selected_score = float(pooled_avg_by_n_estimators[selected_n_estimators])
    selected_pct_above_best = round(100 * (selected_score / best_score - 1), 3)
else:
    selected_score = selected_pct_above_best = None

# %%
# build up selection decisions metadata for run artifacts
selected = {
    "selected_n_estimators": selected_n_estimators,
    "method": f"pooled_p{percentile}",
    "percentile": percentile,
    "pooled_across": ["origin", "horizon"],
    "n_pairs": int(n_pairs),
    "n_resolved": int(n_resolved),
    "n_inconclusive": n_inconclusive,
    "smoothing_window": sel_cfg["smoothing_window"],
    "epsilon": sel_cfg["epsilon"],
    "inconclusive_fraction": round(inconclusive_fraction, 4),
    "inconclusive_tail_fraction": inconclusive_tail_fraction,
    "max_inconclusive_fraction": max_inconclusive_fraction,
    "inconclusive_gate_passed": bool(gate_passed),
    "pooled_overall": {
      "best_score": round(best_score, 2),
      "best_n_estimators": best_n_estimators,
      "selected_score": round(selected_score, 2) if selected_score is not None else None,
      "selected_pct_above_best": selected_pct_above_best,
    },
}

# %%
selected

# %%
# -- create sidecar contents for this calibration run --------------------

# composed config: calibration knobs + the resolved weekly origins (sanitized to plain types)
resolved_origins = [
    {"origin": str(pd.Timestamp(p["origin"]).date()), "horizon": int(p["horizon"])}
    for p in calibration_origin_pairs
]
composed = {**cal_cfg, "resolved_calibration_origins": resolved_origins}
save_config(composed, f"{sidecar_uri}composed_config.yaml")

# run metadata: the lineage triple (git / uv.lock / input URIs)
run_metadata = {
    "git_hash": get_git_hash(),
    "uv_lock_info": get_uv_lock_info(),
    "ts_data_uri": timeseries_uri,
    "calendar_uri": calendar_uri,
}
write_text_to_gcs(json.dumps(run_metadata, indent=2), f"{sidecar_uri}run_metadata.json")

# input snapshots for self-contained reproduction
calendar_df.to_parquet(f"{sidecar_uri}fiscal_calendar.parquet", index=False)
ts_df.to_parquet(f"{sidecar_uri}time_series_snapshot.parquet", index=False)

# result tables
scores_df.to_parquet(f"{sidecar_uri}calibration_scores.parquet", index=False)
recommended_n_estimators_df.to_parquet(
    f"{sidecar_uri}recommended_n_estimators_summary.parquet", index=False
)

# the decision record (single source of truth for the chosen value)
write_text_to_gcs(json.dumps(selected, indent=2), f"{sidecar_uri}selected_n_estimators.json")

# the plot: interactive HTML (self-contained) + static PNG (kaleido)
write_text_to_gcs(fig.to_html(include_plotlyjs=True), f"{sidecar_uri}calibration_plot.html")
with fsspec.open(f"{sidecar_uri}calibration_plot.png", "wb") as f:
    f.write(fig.to_image(format="png"))

print(f"Sidecar written: {sidecar_uri}")

# %%
if selected_n_estimators is None:
    raise RuntimeError(
        f"Calibration inconclusive: gate failed "
        f"({n_inconclusive}/{n_pairs} inconclusive > {max_inconclusive_fraction}). "
        f"No n_estimators selected. Evidence written to sidecar: {sidecar_uri}"
    )
