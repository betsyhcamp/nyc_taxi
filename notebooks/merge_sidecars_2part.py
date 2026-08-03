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

from fcstnyctaxi.lib.io import write_text_to_gcs
from fcstnyctaxi.lib.utils import generate_run_id
from tsbricks.blocks.metadata import get_git_hash, get_uv_lock_info

# %%
# The two sidecars part1 is the EARLIER half, part2 the LATER half (ordering enforced).
# Fill with the real GCS URIs, trailing slash included.
SIDECAR_URI_1 = "gs://nyc-taxi-ehc--modeling/dev/backtests/backtest_weekly/20260802T015412644312Z/"  # earlier half
SIDECAR_URI_2 = "gs://nyc-taxi-ehc--modeling/dev/backtests/backtest_weekly/20260803T014716680214Z/"  # later half

merged_output_uri = (
    "gs://nyc-taxi-ehc--modeling/dev/backtests/backtest_weekly_merged/"
    f"{generate_run_id()}/" # fresh timestamp at runtime; trailing slash matters
)


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
    Reapply shared ordered CategoricalDtype after the concat so merged `tier` 
    is Categorical exactly as a original monthly_series.parquet would be.

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
def copy_text_artifact(src_uri: str, dst_uri: str) -> None:
    """Copy a text artifact (yaml/json) byte-verbatim from src_uri to dst_uri.

    re-reads the raw source text rather than re-serializing a parsed dict, so each 
    parent's exact bytes are preserved. Text via fsspec.open / write_text_to_gcs.
    """
    with fsspec.open(src_uri, "r") as f:
        text = f.read()
    write_text_to_gcs(text, dst_uri)


# %%
def concatenate_sidecars(
    sidecar_uri_1: str,   # the EARLIER half (part1)
    sidecar_uri_2: str,   # the LATER half   (part2)
    output_uri: str,      # gs://.../backtest_weekly_merged/{run_id}/
) -> None:
    """Concatenate two non-overlapping backtest sidecars into one merged, sidecar
    artifact at output_uri, consumable by downstream leaderboard unchanged.
    Model agnostic.
    """
    # read both parts
    part1 = read_sidecar_part(sidecar_uri_1)
    part2 = read_sidecar_part(sidecar_uri_2)

    ms1, ms2 = part1["monthly_series"], part2["monthly_series"]
    metrics1, metrics2 = part1["metrics"], part2["metrics"]
    model1 = part1["composed_config"]["model"]
    model2 = part2["composed_config"]["model"]
    cal1 = (
        part1["fiscal_calendar"][["ds", "fiscal_year_month"]]
        .drop_duplicates()
        .sort_values(by=["ds", "fiscal_year_month"])
        .reset_index(drop=True)
    )
    cal2 = (
        part2["fiscal_calendar"][["ds", "fiscal_year_month"]]
        .drop_duplicates()
        .sort_values(by=["ds", "fiscal_year_month"])
        .reset_index(drop=True)
    )

    # check every loaded parquet is non-empty (fail fast)
    for label, part, uri in [("part1", part1, sidecar_uri_1),
                             ("part2", part2, sidecar_uri_2)]:
        for artifact in ("monthly_series", "metrics", "fiscal_calendar"):
            assert not part[artifact].empty, f"{label} {artifact}.parquet is empty: {uri}"

    # validation battery

    # check: forecast origins disjoint + ordered (enforces non-overlap AND that
    #  part1 is the earlier half).
    assert ms1["forecast_origin_date"].max() < ms2["forecast_origin_date"].min(), \
        "Error: monthly series forecast origin date ranges of part1 & part2 overlap"
    
    # check: column name set equality
    assert set(ms1.columns) == set(ms2.columns), \
        "Error: part 1 monthly series column set not same as part 2 column set"
    
    # check: same fit/predict and predict model callable; a differing callable means two
    # DIFFERENT models; refuse the merge
    for model_type in ("fit_predict_callable", "predict_callable"):
        assert model1.get(model_type)== model2.get(model_type), \
            f"{model_type} differs between parts; refusing merge of different models"
    
    # check: calendars match: both halves ran against the same fiscal calendar.
    assert cal1.equals(cal2), "fiscal calendars not the same between part1 vs part 2"

    # concatenate the core data
    merged_ms = concatenate_monthly_series(ms1, ms2)
    assert len(merged_ms) == len(ms1) + len(ms2), \
        "row count not conserved by concat"
    
    merged_metrics = pd.concat([metrics1, metrics2], ignore_index=True)

    # compose honest metadata for lineage
    hp = compose_hyperparameters(
        model1.get("hyperparameters", {}), model2.get("hyperparameters", {})
    )

    def _origin_summary(ms: pd.DataFrame) -> dict:
        origins = pd.to_datetime(ms["forecast_origin_date"])
        return {
            "n_origins": int(origins.nunique()),
            "origin_min": str(origins.min().date()),
            "origin_max": str(origins.max().date()),
        }

    run_metadata = {
        "merge_type": "two_part_window_concatenation",
        "source_sidecar_uris": [sidecar_uri_1, sidecar_uri_2],
        "parts": {"part1": _origin_summary(ms1), "part2": _origin_summary(ms2)},
        "model": {
            "fit_predict_callable": model1["fit_predict_callable"],
            "predict_callable": model1.get("predict_callable"),
            "hyperparameters_shared": hp["shared"],
            "hyperparameters_divergent": hp["divergent"],
            "hyperparameters_part1_only": hp["part1_only"],
            "hyperparameters_part2_only": hp["part2_only"],
        },
        "merge_git_hash": get_git_hash(),
        "merge_uv_lock_info": get_uv_lock_info(),
        "merge_run_id": output_uri.rstrip("/").split("/")[-1],
    }

    # write merged sidecar to output_uri; all files required by downstream leaderboard
    merged_ms.to_parquet(f"{output_uri}monthly_series.parquet", index=False)
    merged_metrics.to_parquet(f"{output_uri}metrics.parquet", index=False)
    part1["fiscal_calendar"].to_parquet(
        f"{output_uri}fiscal_calendar.parquet", index=False
    )
    copy_text_artifact(
        f"{sidecar_uri_1}composed_config.yaml", f"{output_uri}composed_config.yaml"
    )  # verbatim copy of part1;satisfies leaderboard ["model"] subscript)
    
    write_text_to_gcs(
        json.dumps(run_metadata, indent=2), f"{output_uri}run_metadata.json"
    )

    # verbatim parent copies
    copy_text_artifact(
        f"{sidecar_uri_1}composed_config.yaml", f"{output_uri}composed_config_part1.yaml"
    )
    copy_text_artifact(
        f"{sidecar_uri_2}composed_config.yaml", f"{output_uri}composed_config_part2.yaml"
    )
    copy_text_artifact(
        f"{sidecar_uri_1}run_metadata.json", f"{output_uri}run_metadata_part1.json"
    )
    copy_text_artifact(
        f"{sidecar_uri_2}run_metadata.json", f"{output_uri}run_metadata_part2.json"
    )

    print(f"Merged sidecar written: {output_uri}")


# %%
concatenate_sidecars(
    sidecar_uri_1 = SIDECAR_URI_1,
    sidecar_uri_2 = SIDECAR_URI_2,
    output_uri = merged_output_uri
)
