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
import json

import fsspec
import pandas as pd
import yaml

# %%
# The two sidecars part1 is the EARLIER half, part2 the LATER half (ordering enforced).
# Fill with the real GCS URIs, trailing slash included.
SIDECAR_URI_1 = "gs://nyc-taxi-ehc--modeling/dev/backtests/backtest_weekly/20260802T015412644312Z/"  # earlier half
SIDECAR_URI_2 = "gs://nyc-taxi-ehc--modeling/dev/backtests/backtest_weekly/20260803T014716680214Z/"  # later half


# %%
def read_sidecar_part(uri: str) -> dict:
    """Read needed artifacts from a single sidecar.

    Notebook-local read helper; reused throughout notebook. Text via fsspec.open,
    parquet via pd.read_parquet.

    Args:
        uri: Sidecar prefix with trailing slash, e.g.
            "gs://.../backtest_weekly/<run_id>/".

    Returns:
        dict with keys "monthly_series", "metrics", "fiscal_calendar"
        (DataFrames) and "composed_config", "run_metadata" (parsed dicts).
    """
    with fsspec.open(f"{uri}composed_config.yaml", "r") as f:
        composed_config = yaml.safe_load(f)
    with fsspec.open(f"{uri}run_metadata.json", "r") as f:
        run_metadata = json.load(f)
    return {
        "monthly_series": pd.read_parquet(f"{uri}monthly_series.parquet"),
        "metrics": pd.read_parquet(f"{uri}metrics.parquet"),
        "fiscal_calendar": pd.read_parquet(f"{uri}fiscal_calendar.parquet"),
        "composed_config": composed_config,
        "run_metadata": run_metadata,
    }


# %%
def compose_hyperparameters(hp1: dict, hp2: dict) -> dict:
    """Generic, model-agnostic diff of two parts' model.hyperparameters.

    Never hardcodes any model's hyperparameter vocabulary (e.g. n_estimators). It 
    composes a generic diff so if the same hyperparam has a different value per half, 
    that parameter lands in `divergent`, and a model without that key simply never 
    contributes it. 

    Args:
        hp1: part1's model.hyperparameters dict.
        hp2: part2's model.hyperparameters dict.

    Returns:
        {
          "shared":     {k: v},                          # in both, equal
          "divergent":  {k: {"part1": v1, "part2": v2}}, # in both, differ
          "part1_only": {k: v1},                         # only in part1
          "part2_only": {k: v2},                         # only in part2
        }
    """
    keys1, keys2 = set(hp1), set(hp2)
    both = keys1.intersection(keys2)
    return {
        "shared": {k: hp1[k] for k in both if hp1[k] == hp2[k]},
        "divergent": {
            k:{"part1": hp1[k], "part2": hp2[k]} for k in both if hp1[k]!=hp2[k]
        },
        "part1_only":{k: hp1[k] for k in keys1.difference(keys2)},
        "part2_only":{k: hp2[k] for k in keys2.difference(keys1)}
        }


# %%
def concatenate_monthly_series(ms1: pd.DataFrame, ms2: pd.DataFrame) -> pd.DataFrame:
    """Concatenate two monthly_series frames, preserving the `tier` categorical.

    pd.concat of two Categorical columns whose category sets/orders differ
    silently downgrades the result to object dtype leading to downstream failures.
    Re-apply the shared ordered CategoricalDtype after the concat so merged `tier` 
    is Categorical exactly as a single-run monthly_series.parquet would be.

    Args:
        ms1: part1's monthly_series.parquet frame.
        ms2: part2's monthly_series.parquet frame.

    Returns:
        Row concatenation of ms1 and ms2 with `tier` restored to the shared
        ordered Categorical dtype.
    """
    merged = pd.concat([ms1, ms2], ignore_index=True)

    assert isinstance(ms1["tier"].dtype, pd.CategoricalDtype), \
        "part 1 tier column not categorical datatype"

    assert isinstance(ms2["tier"].dtype, pd.CategoricalDtype), \
        "part 2 tier column not categorical datatype"

    # checks same categories + order 
    assert ms1["tier"].dtype == ms2["tier"].dtype, \
        "part 1 vs 2 differ in categories, order, or order flag (bool flag)"

    # define custom type from input dataframe and apply to output dataframe tier column
    tier_dtype = pd.CategoricalDtype(
        categories=ms1["tier"].cat.categories,
        ordered=ms1["tier"].cat.ordered,
    )
    merged["tier"] = merged["tier"].astype(tier_dtype)
    
    return merged



# %%
part1 = read_sidecar_part(SIDECAR_URI_1)
part2 = read_sidecar_part(SIDECAR_URI_2)


# %%
hp_composed = compose_hyperparameters(
    part1["composed_config"]["model"].get("hyperparameters", {}),
    part2["composed_config"]["model"].get("hyperparameters", {}),
)
hp_composed

# %%
concatenated_monthly_series = concatenate_monthly_series(part1["monthly_series"], part2["monthly_series"])
assert len(concatenated_monthly_series) == len(part1["monthly_series"]) + len(part2["monthly_series"])
print("rows:", len(concatenated_monthly_series), "| tier dtype:", concatenated_monthly_series["tier"].dtype)
concatenated_monthly_series.head()

