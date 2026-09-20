# %%
# The evaluate migration gate, ported from notebooks/leaderboard.py. Throwaway:
# delete once the pipeline is complete.
#
# Only these loading cells may differ from leaderboard.py. This file proves
# evaluate_impl equals the copy below; `diff` against leaderboard.py proves the
# copy equals the notebook, so a hunk further down is a hole in the evidence.
#
# One check cannot be a cell, because it changes a library this file imports.
# Re-run the whole file with the row-exclusion fix in fold_metrics.py reverted,
#
#     git show d7d3269^:src/fcstnyctaxi/lib/fold_metrics.py \
#         > src/fcstnyctaxi/lib/fold_metrics.py
#
# and confirm every number is unchanged. A moved number is a finding rather than
# a failure: it would mean these sidecars carry a fold with no weight or a nan
# row, which the last two cells report on directly.
import shutil
import tempfile
from pathlib import Path

import fsspec
import pandas as pd
import yaml

from fcstnyctaxi.core.train.evaluate_impl import evaluate_impl
from fcstnyctaxi.lib.fold_metrics import (
    compute_wrmae_pooled,
    compute_wrmae_per_series,
    compute_wape,
    compute_weighted_signed_bias,
    compute_signed_bias_pooled,
    compute_signed_bias_per_series,
)
from fcstnyctaxi.lib.period_utils import derive_horizon_label

pd.options.display.max_columns = 40
pd.options.display.max_rows = 100

try:  # `display` is IPython's, and the cells below are copied rather than edited.
    display
except NameError:
    display = print

# %%
# Pinned, immutable, and already proven equivalent to this notebook by
# backtest's own gate, so trusting them needs no new setup.
PINNED_RUN_URI = "gs://nyc-taxi-ehc--modeling/dev/train/20260918T030436650350Z/"
CHALLENGER_MODEL = "lightgbm"
BENCHMARK_MODEL = "naive"

# The local runner's own mirror, so a re-run restages nothing. One recursive
# get: the run root holds compose_configs/, run_identity.json and both sidecars,
# which is exactly evaluate_impl's four arguments.
run_root = (
    Path(tempfile.gettempdir())
    / "fcstnyctaxi"
    / PINNED_RUN_URI.removeprefix("gs://").rstrip("/")
)
if not run_root.is_dir():
    fsspec.filesystem("gs").get(PINNED_RUN_URI, f"{run_root}/", recursive=True)

compose_configs_dir = run_root / "compose_configs"
challenger_dir = run_root / "backtest" / CHALLENGER_MODEL
benchmark_dir = run_root / "backtest" / BENCHMARK_MODEL
print(f"staged at {run_root}")

# %%
# From the staged copies, not leaderboard_runs.yaml: both sides then read one
# set of bytes, and no tracked config is touched.
runs = []
for model, sidecar_dir in (
    (BENCHMARK_MODEL, benchmark_dir),
    (CHALLENGER_MODEL, challenger_dir),
):
    composed_cfg = yaml.safe_load((sidecar_dir / "composed_config.yaml").read_text())

    runs.append({
        "model": model,
        "framing": "monthly_rollup",
        "is_benchmark": model == BENCHMARK_MODEL,
        "model_section": composed_cfg["model"],
        "metrics_df": pd.read_parquet(sidecar_dir / "metrics.parquet"),
        "monthly_series_df": pd.read_parquet(sidecar_dir / "monthly_series.parquet"),
        "calendar_df": pd.read_parquet(sidecar_dir / "fiscal_calendar.parquet")[["ds", "fiscal_year_month"]].drop_duplicates(),
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
all_monthly_series["horizon"] = derive_horizon_label(
    predicted_fiscal_year_month = all_monthly_series["predicted_fiscal_year_month"],
    origin_fiscal_year_month = all_monthly_series["origin_fiscal_year_month"],
    origin_month_fraction_elapsed = all_monthly_series["origin_month_fraction_elapsed"]
)

benchmark_monthly_series["horizon"] = derive_horizon_label(
    predicted_fiscal_year_month = benchmark_monthly_series["predicted_fiscal_year_month"],
    origin_fiscal_year_month = benchmark_monthly_series["origin_fiscal_year_month"],
    origin_month_fraction_elapsed = benchmark_monthly_series["origin_month_fraction_elapsed"]
)


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
# ================================================
# The gate proper. Everything above this line is leaderboard.py.
# Every check below prints PASS; anything else is a failure.
# ================================================
shutil.rmtree(run_root / "evaluate", ignore_errors=True)
impl_summary_result = evaluate_impl(
    challenger_dir=challenger_dir,
    benchmark_dir=benchmark_dir,
    compose_configs_dir=compose_configs_dir,
    out_dir=run_root / "evaluate",
)
# Read off disk rather than kept in memory, so the write path is in scope.
impl_per_series = pd.read_parquet(run_root / "evaluate" / "per_series_comparison.parquet")
impl_fold = pd.read_parquet(run_root / "evaluate" / "fold_metrics.parquet")
impl_period = pd.read_parquet(run_root / "evaluate" / "period_metrics.parquet")
impl_summary = pd.read_parquet(run_root / "evaluate" / "summary_metrics.parquet")

print(impl_summary_result)
for name, frame in (
    ("per_series_comparison", impl_per_series),
    ("fold_metrics", impl_fold),
    ("period_metrics", impl_period),
    ("summary_metrics", impl_summary),
):
    print(f"{name:24s} {frame.shape}")

# %%
# Written here and never imported: reusing the impl's melt would let a melt bug
# cancel itself. The one name the notebook spells differently is renamed here.
NOTEBOOK_METRIC_NAMES = {"weighted_signed_bias": "wsb"}
RTOL = 1e-9


def to_notebook_wide(long_df, index_keys, model):
    """One model's long score table, in the notebook's wide column shape."""
    wide = (
        long_df[long_df["model"] == model]
        .pivot(index=index_keys, columns=["metric", "tier"], values="value")
    )
    wide.columns = [
        f"{NOTEBOOK_METRIC_NAMES.get(metric, metric)}_{tier}"
        for metric, tier in wide.columns
    ]
    return wide.reset_index()


def compare(name, notebook_df, impl_df, keys):
    """The notebook's own columns, canonically sorted, within rtol."""
    left = notebook_df.sort_values(keys).reset_index(drop=True)
    # Taking the notebook's columns carves out what the impl adds, no list to keep.
    right = impl_df[list(notebook_df.columns)].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        left, right, rtol=RTOL, check_dtype=False, check_categorical=False
    )
    print(f"PASS {name:22s} {left.shape[0]:5d} rows x {left.shape[1]:3d} cols identical")


# %%
# The challenger only: the benchmark rows have no counterpart in the notebook,
# and scoring itself at 1.0 is what checks them instead.
compare(
    "summary_metrics",
    summary_df,
    to_notebook_wide(impl_summary, ["model", "horizon"], CHALLENGER_MODEL),
    ["model", "horizon"],
)
compare(
    "period_metrics",
    period_breakdown_df,
    to_notebook_wide(
        impl_period,
        ["model", "horizon", "predicted_fiscal_year_month"],
        CHALLENGER_MODEL,
    ),
    ["model", "horizon", "predicted_fiscal_year_month"],
)
compare(
    "fold_metrics",
    fold_breakdown_df,
    to_notebook_wide(
        impl_fold,
        ["model", "horizon", "predicted_fiscal_year_month", "forecast_origin_date"],
        CHALLENGER_MODEL,
    ),
    ["model", "horizon", "predicted_fiscal_year_month", "forecast_origin_date"],
)

# %%
# The base frame, on the column intersection and matched rows only: each side
# carries columns the other lacks, and intersecting handles both directions.
per_series_keys = [
    "forecast_origin_date",
    "predicted_fiscal_year_month",
    "unique_id",
]
shared = [c for c in per_series_df.columns if c in impl_per_series.columns]
print(f"shared columns: {shared}")
print(f"impl only: {sorted(set(impl_per_series.columns) - set(per_series_df.columns))}")
print(f"notebook only: {sorted(set(per_series_df.columns) - set(impl_per_series.columns))}")
compare(
    "per_series_comparison",
    per_series_df[shared],
    impl_per_series[impl_per_series["_merge"] == "both"],
    per_series_keys,
)

# %%
# Scored against itself the pooled ratio is bit-exact; the per-series one
# renormalizes weights and only reaches 1.0 to tolerance, so never test equality.
benchmark_rows = impl_summary[impl_summary["model"] == BENCHMARK_MODEL]
for metric, tolerance in (("wrmae_pooled", 0.0), ("wrmae_per_series", 1e-9)):
    values = benchmark_rows.loc[benchmark_rows["metric"] == metric, "value"].dropna()
    worst = (values - 1.0).abs().max()
    assert worst <= tolerance, f"{metric} is not the reference line"
    print(f"PASS {metric:22s} {len(values):3d} cells at 1.0, max |v - 1| = {worst:.3e}")

# %%
# The derivation identity on real data, one cell three ways: the notebook's call
# over the whole slice, a nanmean of the impl's fold values taken here, and the
# impl's summary. The middle one is the claim that summary is derivable at all.
horizon = "horizon_1"
notebook_direct = compute_wrmae_pooled(
    all_monthly_series[all_monthly_series["horizon"] == horizon],
    benchmark_monthly_series[benchmark_monthly_series["horizon"] == horizon],
    tier=None,
)
cell = (impl_fold["model"] == CHALLENGER_MODEL) & (impl_fold["horizon"] == horizon)
cell &= (impl_fold["tier"] == "global") & (impl_fold["metric"] == "wrmae_pooled")
by_hand = impl_fold.loc[cell, "value"].mean()  # nanmean: pandas skips nan
impl_cell = impl_summary[
    (impl_summary["model"] == CHALLENGER_MODEL)
    & (impl_summary["horizon"] == horizon)
    & (impl_summary["tier"] == "global")
    & (impl_summary["metric"] == "wrmae_pooled")
]["value"].item()

print(f"notebook, whole slice : {notebook_direct!r}")
print(f"nanmean of impl folds : {by_hand!r}  over {int(cell.sum())} folds")
print(f"impl summary cell     : {impl_cell!r}")
assert abs(by_hand - impl_cell) <= RTOL * abs(impl_cell)
assert abs(notebook_direct - impl_cell) <= RTOL * abs(impl_cell)
print("PASS all three paths agree on one summary cell")

# %%
# What makes the reverted-library re-run at the top readable: unchanged numbers
# mean either that the row-exclusion fix is a no-op here or that its path was
# never reached. A zero-weight row adds 0 to both sides of the ratio, so
# excluding it changes nothing; a slice carrying no weight at all normalizes 0/0
# and used to sum to 0.0, the best attainable value. These counts say which.
fold_keys = ["forecast_origin_date", "predicted_fiscal_year_month"]
zero_weight = impl_per_series[impl_per_series["series_weight"] <= 0]
print(
    f"zero-weight rows: {len(zero_weight)} across "
    f"{zero_weight['unique_id'].nunique()} series and "
    f"{len(zero_weight[fold_keys].drop_duplicates())} folds"
)
for tier in [None, *impl_per_series["tier"].cat.categories]:
    sliced = (
        impl_per_series if tier is None else impl_per_series[impl_per_series["tier"] == tier]
    )
    totals = sliced.groupby(fold_keys, observed=True)["series_weight"].sum()
    print(f"  tier={str(tier):10s} slices with no weight at all: {int((totals <= 0).sum())} of {len(totals)}")
nulls = impl_per_series[["monthly_forecast_ch", "monthly_forecast_bm"]].isna().any(axis=1)
print(f"rows with a null forecast on either side: {int(nulls.sum())}")

# %%
# Every fold belongs to exactly one horizon, so fold rows over summary rows is
# folds per horizon.
folds_per_horizon = len(impl_fold) / len(impl_summary)
assert folds_per_horizon == impl_summary_result.n_folds_total / 2
print(f"PASS fold rows / summary rows = {folds_per_horizon:g} = folds per horizon")

# %%
# Plausibility eyeball: the whole summary table, both models, wide.
display(
    impl_summary.pivot(
        index=["model", "horizon"], columns=["metric", "tier"], values="value"
    )
)

# %%
# The comparison must be able to fail, or every PASS above proves nothing. Feed
# it the benchmark's scores where the challenger's belong, relabeled first:
# otherwise it fails on the model name and never reaches a number.
wrong_scores = to_notebook_wide(impl_summary, ["model", "horizon"], BENCHMARK_MODEL)
wrong_scores["model"] = CHALLENGER_MODEL
try:
    compare("summary_metrics", summary_df, wrong_scores, ["model", "horizon"])
except AssertionError as exc:
    print("PASS the comparison rejected the benchmark's scores, as it must:")
    print("\n".join(str(exc).splitlines()[:4]))
else:
    raise AssertionError("the comparison accepted the benchmark's scores")

# %%
# The swap this step exists to catch, checked on the guard rather than on the
# comparison. The roles are read off each sidecar's own manifest, so it is
# refused before any table is written and the good output above survives.
try:
    evaluate_impl(
        challenger_dir=benchmark_dir,
        benchmark_dir=challenger_dir,
        compose_configs_dir=compose_configs_dir,
        out_dir=run_root / "evaluate",
    )
except ValueError as exc:
    print(f"PASS the swapped pair was refused:\n{exc}")
else:
    raise AssertionError("the swapped pair was accepted")

assert (run_root / "evaluate" / "evaluate_manifest.json").is_file()
print("PASS the completion marker survived the refused swap")
