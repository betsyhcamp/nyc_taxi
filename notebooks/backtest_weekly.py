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
import pandas as pd
import json
import yaml
import fsspec  # reads composed_config.yaml from GCS in compare_sidecars

from fcstnyctaxi.lib.backtest_results import build_backtest_results, build_cv_results
from fcstnyctaxi.lib.config_utils import merge_configs, save_config
from fcstnyctaxi.lib.cross_validation_utils import sorted_origin_horizon_pairs
from fcstnyctaxi.lib.io import write_text_to_gcs
from fcstnyctaxi.lib.monthly_aggregation import (
    build_monthly_forecast_vs_actual,
    attach_tier_and_weight,
    compute_actual_monthly_totals,
)
from fcstnyctaxi.lib.period_utils import (
    assign_tiers,
    compute_series_weights,
    generate_origins_for_periods
)
from fcstnyctaxi.lib.utils import get_project_root_dir, generate_run_id

from tsbricks.backtesting import (
    evaluate_metrics,
    generate_folds
)

from tsbricks.runner import (
    apply_transforms,
    fit_transforms,
    inverse_transforms,
    invoke_model,
)

# may want to remove for prod code since imports are unused directly
import fs  # noqa: F401 - surfaces pkg_resources deprecation warning here, not mid-loop
import tqdm.auto  # noqa: F401 - surfaces TqdmWarning here, not mid-loop

# %%
project_root = get_project_root_dir()
sys.path.insert(0, str(project_root))

run_config_path = project_root / "notebooks" / "backtest_configs" / "run_config.yaml"
run_cfg = yaml.safe_load(run_config_path.read_text())


backtest_cfg_path = project_root / run_cfg["configs"]["backtest_config"]
model_cfg_path = project_root / run_cfg["configs"]["model"]

sidecar_dir = generate_run_id()
sidecar_uri = (
    f"{run_cfg['project']['gcs_bucket']}/dev/backtests/backtest_weekly/{sidecar_dir}/"
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
raw_backtest_cfg = yaml.safe_load(backtest_cfg_path.read_text())
explicit_origins = (
    raw_backtest_cfg.get("cross_validation") or {}
    ).get("forecast_origins")
if explicit_origins is not None:
    origin_pairs = explicit_origins
else:
    eval_periods = raw_backtest_cfg["evaluation_periods"]

    origin_pairs = generate_origins_for_periods(
        start_months=eval_periods["start_months"],
        forecast_horizon_months=eval_periods["forecast_horizon_months"],
        calendar_df=calendar_df,
        calendar_time_col="ds"
    )

print(f"Forecast origins to use:\n {len(origin_pairs)}")

# %%
global_horizon = (raw_backtest_cfg.get("cross_validation") or {}).get("horizon")
if global_horizon:
    forecast_origins = [p["origin"] for p in origin_pairs]
else:
    forecast_origins = origin_pairs

# %%
runtime_overrides = {
    "aggregation": {"calendar_source": calendar_uri},
    "cross_validation": {"forecast_origins": forecast_origins},
}
cfg = merge_configs(backtest_cfg_path, model_cfg_path, runtime_overrides)

# %%
actual_monthly_df = compute_actual_monthly_totals(
    ts_df,
    calendar_df,
    period_col=cfg.aggregation.period_col
)

# %%
print(yaml.dump(cfg.model_dump(by_alias=True, exclude_none=True), default_flow_style=False))


# %%
def evaluate_model(
    cfg,                           # BacktestConfig — varies per Optuna trial
    ts_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    actual_monthly_df: pd.DataFrame,
) -> dict:
    """
    Runs the composable fold loop for one model config.
    Optuna-compatible: data loaded once outside; cfg varies per trial.

    Returns dict with keys:
        backtest_results,
        monthly_series_df  # columns: "forecast_origin_date",
                           # "predicted_fiscal_year_month","unique_id", "tier",
                           # "monthly_forecast",
                           # "actual_monthly_total", "series_weight"
                           # "origin_month_fraction_elapsed"
        monthly_forecast_components_df
                           # columns: "forecast_origin_date",
                           # "predicted_fiscal_year_month", "unique_id",
                           # "mtd_revenue", "predicted_remaining"
    """

    # -- 1. generate folds --------------------
    cv_folds, _ = generate_folds(
        ts_df, 
        cfg.cross_validation,
        cfg.data
    )
    # -- 2. weekly fold loop: fit -> invoke -> inverse -> evaluate ------
    per_fold_metrics = []
    per_fold_forecasts: dict[str, pd.DataFrame] = {}

    origin_horizon_pairs = sorted_origin_horizon_pairs(
        cfg.cross_validation.origin_horizon_pairs(),
        cfg.data.freq
        )
    monthly_series_rows = []
    components_rows = []
    seen_origins = set() # dedup: forecast_origin_date

    fraction_by_origin = calendar_df.set_index("ds")["origin_month_fraction_elapsed"]

    for fold_idx, (fold_id, splits) in enumerate(cv_folds.items()):
        fold_origin, fold_horizon = origin_horizon_pairs[fold_idx]
        print(f"fold origin: {fold_origin}, fold horizon: {fold_horizon}")

        if fold_origin in seen_origins: # deduplicate fcsts w/ same forecast_origin_date
            continue
        seen_origins.add(fold_origin)

        train, val = splits["train"], splits["val"]

        # tier and weight at this fold's origin
        tier_df = assign_tiers(train, fold_origin, calendar_df)
        weight_df = compute_series_weights(train, fold_origin, calendar_df)
        
        fitted_transforms, train_t = fit_transforms(train, cfg.transforms or [])
        
        _ = apply_transforms(val, fitted_transforms) # here for consistency
        
        forecast_df, _fitted, _model_obj = invoke_model(
            train_t, cfg.model, fold_horizon, future_x_df=calendar_df
        )

        forecast_original_scale = inverse_transforms(forecast_df, fitted_transforms)
        per_fold_forecasts[fold_id] = forecast_original_scale

        monthly_rows_df = build_monthly_forecast_vs_actual(
            forecast_df=forecast_original_scale,
            train_df=train,
            calendar_df=calendar_df,
            actual_monthly_df=actual_monthly_df,
            period_col=cfg.aggregation.period_col,
            time_col= "ds",
            id_col= "unique_id",
            forecast_col= "ypred",
            target_col= "y",
        )
        fold_rows = attach_tier_and_weight(
            monthly_rows_df=monthly_rows_df,
            tier_df=tier_df,
            weight_df=weight_df,
            fold_origin=fold_origin,
            origin_month_fraction_elapsed=fraction_by_origin[fold_origin],
            period_col=cfg.aggregation.period_col,
            id_col = "unique_id",
        )

        monthly_series_rows.append(fold_rows)
        components_rows.append(
            monthly_rows_df
            .assign(forecast_origin_date = fold_origin)[
                ["forecast_origin_date", cfg.aggregation.period_col, "unique_id",
                 "mtd_revenue", "predicted_remaining"]
            ]
        )

        fold_metrics = evaluate_metrics(
            y_true=val,
            y_pred=forecast_original_scale,
            y_train=train,
            metrics_config=cfg.evaluation.native.metrics,
            fold_id=fold_id,
        )
        fold_metrics["fold_origin"] = fold_origin
        fold_metrics["fold_horizon"] = fold_horizon

        per_fold_metrics.append(fold_metrics)

    metrics = pd.concat(per_fold_metrics, ignore_index=True)
    monthly_series_df = pd.concat(monthly_series_rows, ignore_index=True)
    monthly_forecast_components_df = (
        pd.concat(components_rows, ignore_index=True)
        .rename(columns={cfg.aggregation.period_col:"predicted_fiscal_year_month"})
    )
    
    # -- 3. build weekly results -----------------------------
    cv_results = build_cv_results(
        forecasts_per_fold=per_fold_forecasts,
        train_val_splits_per_fold=cv_folds,
        metrics = metrics,
        origin_horizon_pairs=origin_horizon_pairs
    )
    backtest_results = build_backtest_results(
        cv = cv_results,
        config = cfg.model_dump(by_alias=True, exclude_none=True),
        origin_horizon_pairs=origin_horizon_pairs,
        capture_lineage=True
    )


    return {
        "backtest_results": backtest_results,
        "monthly_series_df": monthly_series_df,
        "monthly_forecast_components_df": monthly_forecast_components_df,
    }

# %%
result = evaluate_model(cfg, ts_df, calendar_df, actual_monthly_df)
backtest_results = result["backtest_results"]
monthly_series_df = result["monthly_series_df"]
monthly_forecast_components_df = result["monthly_forecast_components_df"]

# %%
# extract the raw forecasts generated during cross-validation
cv_forecasts_df = pd.concat(
    [df.assign(fold_id=fold_id)
     for fold_id, df in backtest_results.cv.forecasts_per_fold.items()],
    ignore_index=True
)

# %%
cv_forecasts_df["forecast_origin_date"] = (
    cv_forecasts_df["fold_id"]
    .map(backtest_results.cv.fold_id_to_origin)
    .astype("datetime64[s]")  # timestamp unit to match the other output files
)

cv_forecasts_df = cv_forecasts_df[
    ["forecast_origin_date", "unique_id", "ds", "ypred", "fold_id"]
]

# %%
# -- create sidecar contents for this run -------------------

# write composed config
save_config(cfg, f"{sidecar_uri}composed_config.yaml")

# create and write run_metadata.json contents
run_metadata = {
    "git_hash": backtest_results.git_hash,
    "uv_lock_info": backtest_results.uv_lock_info,
    "ts_data_uri": timeseries_uri,
}
write_text_to_gcs(json.dumps(run_metadata, indent=2), f"{sidecar_uri}run_metadata.json")

# input data snapshots for self-contained debugging 
calendar_df.to_parquet(f"{sidecar_uri}fiscal_calendar.parquet", index=False)
ts_df.to_parquet(f"{sidecar_uri}time_series_snapshot.parquet", index=False)

# write output results data to sidecar in GCS
cv_forecasts_df.to_parquet(f"{sidecar_uri}raw_cv_forecasts.parquet", index=False)
backtest_results.cv.metrics.to_parquet(f"{sidecar_uri}metrics.parquet", index=False)
monthly_series_df.to_parquet(f"{sidecar_uri}monthly_series.parquet", index=False)
monthly_forecast_components_df.to_parquet(
    f"{sidecar_uri}monthly_forecast_components.parquet", index=False
)

print(f"Sidecar written: {sidecar_uri}")

# %%
# -- verification helpers ----------------------------------------------------

COMPONENT_KEYS = [
    "forecast_origin_date",
    "predicted_fiscal_year_month",
    "unique_id",
]


def _read(uri: str, name: str) -> pd.DataFrame | None:
    """Read a sidecar parquet, or None if absent/unreadable.

    Returns None rather than raising so one missing artifact fails its own
    checks with a named reason instead of aborting the whole report.
    """
    try:
        return pd.read_parquet(f"{uri}{name}")
    except Exception:
        return None


def _decategorize(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten category columns to object before sorting and comparing values.

    A category sorts by its category order, so two frames holding identical
    values compare unequal when their categories were built in a different
    order. Dtypes are compared separately in _compare_frames, so this cannot
    hide a real dtype difference. monthly_series.parquet's "tier" is the case
    in point.
    """
    cat_cols = [c for c in df.columns if isinstance(df[c].dtype, pd.CategoricalDtype)]
    return df.astype({c: "object" for c in cat_cols}) if cat_cols else df


def _compare_frames(new_df: pd.DataFrame, ref_df: pd.DataFrame) -> tuple[bool, str]:
    """Compare two frames: sort by every column in canonical name order,
    reset the index, then compare column set, dtypes, and values.

    Sorting by every column is deterministic without naming key columns, which
    matters for tsbricks-owned schemas and survives an upstream schema change.

    Returns:
        (passed, detail) where detail names what differed — differing columns,
        dtype mismatches, or per-column max absolute delta.
    """
    new_cols, ref_cols = set(new_df.columns), set(ref_df.columns)
    if new_cols != ref_cols:
        return False, (
            f"column sets differ — only_new={sorted(new_cols - ref_cols)}, "
            f"only_ref={sorted(ref_cols - new_cols)}"
        )

    cols = sorted(new_cols)
    problems = []

    dtype_diffs = {
        c: f"{new_df[c].dtype} vs {ref_df[c].dtype}"
        for c in cols
        if new_df[c].dtype != ref_df[c].dtype
    }
    if dtype_diffs:
        problems.append(f"dtype mismatches: {dtype_diffs}")

    if len(new_df) != len(ref_df):
        problems.append(f"row counts differ: new={len(new_df)} ref={len(ref_df)}")
        return False, "; ".join(problems)

    a = _decategorize(new_df[cols]).sort_values(cols).reset_index(drop=True)
    b = _decategorize(ref_df[cols]).sort_values(cols).reset_index(drop=True)

    value_diffs = []
    for c in cols:
        if a[c].equals(b[c]):
            continue
        if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
            value_diffs.append(f"{c} (max abs delta {(a[c] - b[c]).abs().max():.6g})")
        else:
            value_diffs.append(f"{c} ({int((a[c] != b[c]).sum())} rows differ)")
    if value_diffs:
        problems.append("values differ in " + ", ".join(value_diffs))

    return (not problems), ("identical" if not problems else "; ".join(problems))


def _dict_diff(new_obj, ref_obj, path: str = "") -> list[str]:
    """Leaf-level diff of two parsed-YAML structures.

    Compared as parsed objects rather than as text: YAML formatting is not a
    config change, and a text diff would report a failure the report cannot
    explain.
    """
    if isinstance(new_obj, dict) and isinstance(ref_obj, dict):
        out: list[str] = []
        for key in sorted(set(new_obj) | set(ref_obj), key=str):
            sub = f"{path}.{key}" if path else str(key)
            if key not in new_obj:
                out.append(f"{sub}: missing in new")
            elif key not in ref_obj:
                out.append(f"{sub}: missing in reference")
            else:
                out += _dict_diff(new_obj[key], ref_obj[key], sub)
        return out
    return [] if new_obj == ref_obj else [f"{path}: new={new_obj!r} ref={ref_obj!r}"]


# %%
def compare_sidecars(new_uri: str, reference_uri: str) -> pd.DataFrame:
    """Check that a re-run reproduced a prior registration.

    Compares the run's inputs (composed config, time-series snapshot, fiscal
    calendar), monthly_series.parquet, and raw_cv_forecasts.parquet against a
    reference sidecar. See notes/spec__weekly_sidecar_forecast_components.md
    for the full check list and the rationale behind each.

    Requires a prior registered run of the SAME model. There is nothing to
    compare against otherwise, which is why this is a separate function rather
    than an optional mode of validate_sidecar(): callers running a new model
    skip it entirely instead of reading rows that report "not applicable".

    Every check runs and reports; nothing raises on failure, so one call
    surfaces all problems rather than only the first.

    Args:
        new_uri: Sidecar URI just written by this notebook; trailing slash.
        reference_uri: Registered sidecar URI for the same model.

    Returns:
        DataFrame with columns (check, status, detail), one row per check.
    """
    results: list[dict] = []

    def record(check: str, passed: bool, detail: str) -> None:
        results.append(
            {"check": check, "status": "PASS" if passed else "FAIL", "detail": detail}
        )

    # -- inputs and config identical --------------------------------------
    # Checked first because it is diagnostically prior: config drift and
    # regenerated input data are the likeliest causes of the monthly_series
    # comparison below failing, and the cheapest to test. A failure here
    # explains that one instead of competing with it.
    try:
        with fsspec.open(f"{new_uri}composed_config.yaml", "r") as f:
            new_cfg_yaml = yaml.safe_load(f)
        with fsspec.open(f"{reference_uri}composed_config.yaml", "r") as f:
            ref_cfg_yaml = yaml.safe_load(f)
        diffs = _dict_diff(new_cfg_yaml, ref_cfg_yaml)
        record(
            "composed_config unchanged",
            not diffs,
            "identical"
            if not diffs
            else f"{len(diffs)} diffs: " + "; ".join(diffs[:5]),
        )
    except Exception as exc:
        record("composed_config unchanged", False, f"{type(exc).__name__}: {exc}")

    for label, filename in [
        ("time_series_snapshot unchanged", "time_series_snapshot.parquet"),
        ("fiscal_calendar unchanged", "fiscal_calendar.parquet"),
    ]:
        new_df, ref_df = _read(new_uri, filename), _read(reference_uri, filename)
        if new_df is None or ref_df is None:
            absent = "new" if new_df is None else "reference"
            record(label, False, f"{filename} unreadable in {absent} sidecar")
        else:
            record(label, *_compare_frames(new_df, ref_df))

    # -- monthly_series.parquet identical ---------------------------------
    # The load-bearing check: for the benchmark model this file is the
    # denominator of every skill score in the project.
    ms_new = _read(new_uri, "monthly_series.parquet")
    ms_ref = _read(reference_uri, "monthly_series.parquet")
    if ms_new is None or ms_ref is None:
        absent = "new" if ms_new is None else "reference"
        record("monthly_series unchanged", False, f"unreadable in {absent} sidecar")
    else:
        record("monthly_series unchanged", *_compare_frames(ms_new, ms_ref))

    # -- raw_cv_forecasts identical on its pre-existing columns --------------
    # Independent of the monthly_series comparison, not a restatement of it:
    # two different weekly
    # forecast vectors can sum to identical monthly totals, so monthly_series
    # equivalence does not by itself prove the weekly forecasts are unchanged.
    # This is the check that establishes "the same forecasts, row for row".
    raw_new = _read(new_uri, "raw_cv_forecasts.parquet")
    raw_ref = _read(reference_uri, "raw_cv_forecasts.parquet")
    if raw_new is None or raw_ref is None:
        absent = "new" if raw_new is None else "reference"
        record("raw_cv_forecasts unchanged", False, f"unreadable in {absent} sidecar")
    else:
        added = set(raw_new.columns) - set(raw_ref.columns)
        removed = set(raw_ref.columns) - set(raw_new.columns)
        if added != {"forecast_origin_date"} or removed:
            record(
                "raw_cv_forecasts unchanged",
                False,
                f"forecast_origin_date should be the only addition — "
                f"added={sorted(added)}, removed={sorted(removed)}",
            )
        else:
            # Compare on the reference's columns, dropping this PR's addition.
            passed, detail = _compare_frames(raw_new[list(raw_ref.columns)], raw_ref)
            record("raw_cv_forecasts unchanged", passed, detail)

    return pd.DataFrame(results, columns=["check", "status", "detail"])


# %%
def validate_sidecar(uri: str) -> pd.DataFrame:
    """Check that one sidecar is internally well-formed.

    Checks that the components file reconstructs monthly_series row for row,
    that forecast_origin_date carries the same dtype in every file, and that
    the raw forecasts are keyed by origin. See
    notes/spec__weekly_sidecar_forecast_components.md for the rationale.

    Needs no reference, so it applies to every run — including the first run of
    a brand-new model, which is when a malformed sidecar is most likely, since
    new code wrote it.

    Every check runs and reports; nothing raises on failure.

    Args:
        uri: Sidecar URI to validate; trailing slash.

    Returns:
        DataFrame with columns (check, status, detail), one row per check.
    """
    results: list[dict] = []

    def record(check: str, passed: bool, detail: str) -> None:
        results.append(
            {"check": check, "status": "PASS" if passed else "FAIL", "detail": detail}
        )

    ms_df = _read(uri, "monthly_series.parquet")
    comp_df = _read(uri, "monthly_forecast_components.parquet")
    raw_df = _read(uri, "raw_cv_forecasts.parquet")

    # -- components key-set equality and reconstruction -------------------
    # Outer merge (not inner) with indicator: an inner merge would verify
    # reconstruction on the intersection alone, so a components file missing
    # one key while carrying a spurious one would pass.
    if ms_df is None or comp_df is None:
        record(
            "components key-set + reconstruction",
            False,
            "monthly_series.parquet or monthly_forecast_components.parquet "
            "unreadable",
        )
    else:
        try:
            merged = ms_df.merge(
                comp_df,
                on=COMPONENT_KEYS,
                how="outer",
                indicator=True,
                validate="one_to_one",
            )
            keys_ok = bool((merged["_merge"] == "both").all())
            residual = (
                merged["mtd_revenue"]
                + merged["predicted_remaining"]
                - merged["monthly_forecast"]
            )
            # Exact, not approximate: monthly_forecast IS this sum, computed in
            # float64 and round-tripped losslessly through parquet. A tolerance
            # here would only hide a dropped or swapped addend.
            sum_ok = bool((residual == 0).all())

            if keys_ok and sum_ok:
                detail = f"{len(merged)} rows matched; reconstruction exact"
            elif not keys_ok:
                counts = merged["_merge"].value_counts().to_dict()
                sample = (
                    merged.loc[merged["_merge"] != "both", COMPONENT_KEYS]
                    .head(3)
                    .to_dict("records")
                )
                detail = f"key sets differ — {counts}; first unmatched: {sample}"
            else:
                detail = (
                    f"reconstruction failed on {int((residual != 0).sum())} rows; "
                    f"max |residual| {residual.abs().max():.6g}"
                )
            record("components key-set + reconstruction", keys_ok and sum_ok, detail)
        except Exception as exc:
            record(
                "components key-set + reconstruction",
                False,
                f"{type(exc).__name__}: {exc}",
            )

    # -- forecast_origin_date dtype agrees across files -------------------
    # Asserted as equality among the three, with no expected literal: a
    # hardcoded dtype would go red on a pandas/pyarrow upgrade that broke
    # nothing. Expect datetime64[ms] today; the spec's future-work section
    # carries the root fix (normalizing the origin unit upstream).
    frames = {
        "raw_cv_forecasts": raw_df,
        "monthly_series": ms_df,
        "monthly_forecast_components": comp_df,
    }
    absent_frames = [name for name, df in frames.items() if df is None]
    lacking = [
        name
        for name, df in frames.items()
        if df is not None and "forecast_origin_date" not in df.columns
    ]
    if absent_frames:
        record(
            "forecast_origin_date dtype", False, f"unreadable: {absent_frames}"
        )
    elif lacking:
        record("forecast_origin_date dtype", False, f"column absent in: {lacking}")
    else:
        dtypes = {n: str(df["forecast_origin_date"].dtype) for n, df in frames.items()}
        agreed = len(set(dtypes.values())) == 1
        record(
            "forecast_origin_date dtype",
            agreed,
            f"all {next(iter(dtypes.values()))}"
            if agreed
            else "; ".join(f"{n}={d}" for n, d in dtypes.items()),
        )

    # -- raw forecasts are keyed by origin --------------------------------
    # .map() returns NaT for an unmapped key rather than raising, so the null
    # count is the guard against a silently incomplete fold_id_to_origin.
    if raw_df is None:
        record(
            "raw_cv_forecasts keying", False, "raw_cv_forecasts.parquet unreadable"
        )
    elif "forecast_origin_date" not in raw_df.columns:
        record("raw_cv_forecasts keying", False, "forecast_origin_date absent")
    else:
        null_count = int(raw_df["forecast_origin_date"].isna().sum())
        first_col = list(raw_df.columns)[0]
        record(
            "raw_cv_forecasts keying",
            null_count == 0 and first_col == "forecast_origin_date",
            f"{null_count} null origins; first column is {first_col!r}",
        )

    return pd.DataFrame(results, columns=["check", "status", "detail"])


# %%
# validate_sidecar() runs on every run. compare_sidecars() runs only when this
# model has a prior registration to reproduce — verification.reference_model is
# null for a brand-new model, and the equivalence half is then not invoked at
# all rather than reported as failures.
#
# The reference is resolved from leaderboard_runs.yaml rather than hardcoded, so
# re-registering in a later commit cannot strand a stale URI in this notebook.
reference_model = run_cfg["verification"]["reference_model"]

if reference_model is None:
    reference_sidecar_uri = None
    print("verification.reference_model is null — validation only, no comparison.")
else:
    leaderboard_runs = yaml.safe_load(
        (
            project_root / "notebooks" / "backtest_configs" / "leaderboard_runs.yaml"
        ).read_text()
    )
    entry = next(
        (r for r in leaderboard_runs["runs"] if r["model"] == reference_model), None
    )
    if entry is None:
        raise ValueError(f"{reference_model!r} not in leaderboard_runs.yaml")
    reference_sidecar_uri = (
        entry["sidecar_uri"]
        if "sidecar_uri" in entry
        else f"{leaderboard_runs['default_base_uri']}{entry['sidecar_id']}/"
    )
    print(f"Comparing against {reference_model}: {reference_sidecar_uri}")

# Comparison first: the input-equivalence rows are diagnostically prior, so
# they should be read before anything else in the report.
reports = []
if reference_sidecar_uri is not None:
    reports.append(compare_sidecars(sidecar_uri, reference_sidecar_uri))
reports.append(validate_sidecar(sidecar_uri))
report = pd.concat(reports, ignore_index=True)

print(report.to_string(index=False))
print(f"\nALL PASS: {(report['status'] == 'PASS').all()}")

# %%
#train_slice = ts_df[ts_df["ds"] <= "2025-04-27"]
#forecast_df, fitted_df, mlfcst = invoke_model(
#    train_slice, cfg.model, horizon=8, future_x_df=calendar_df
#)
#
#booster = mlfcst.models_["LGBMRegressor"]
#
#importance_df = pd.DataFrame({
#    "feature": booster.feature_name_,
#    "importance": booster.feature_importances_,
#}).sort_values("importance", ascending=False)


# %%
#importance_df
