from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd

from fcstnyctaxi.lib.column_checks import require_columns
from fcstnyctaxi.lib.metrics import (
    _SMALL_NUM_BOUND,
)

_log = logging.getLogger(__name__)

_FOLD_KEYS = ["forecast_origin_date", "predicted_fiscal_year_month"]
_JOIN_KEYS = _FOLD_KEYS + ["unique_id"]
_PROGRESS_COLS = ["weeks_in_month", "weeks_actualized"]
_PROGRESS_SKILL_COLS = _PROGRESS_COLS + ["n_events", "wrmae_vs_benchmark"]


def compute_wrmae_pooled(
    challenger_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    tier: str | None = None,
) -> float:
    """Fold-averaged pooled WRMAE. tier=None -> all series."""
    if tier is not None:
        challenger_df = challenger_df[challenger_df["tier"] == tier]
        benchmark_df = benchmark_df[benchmark_df["tier"] == tier]

    ch_cols = _JOIN_KEYS + ["monthly_forecast", "actual_monthly_total", "series_weight"]
    bm_cols = _JOIN_KEYS + ["monthly_forecast"]

    merged = challenger_df[ch_cols].merge(
        benchmark_df[bm_cols],
        on=_JOIN_KEYS,
        suffixes=("_ch", "_bm"),
        how="inner",
    )
    if merged.empty:
        return np.nan

    merged["challenger_errors"] = (
        merged["monthly_forecast_ch"] - merged["actual_monthly_total"]
    ).abs()
    merged["weighted_errors_ch"] = merged["series_weight"] * merged["challenger_errors"]

    merged["benchmark_errors"] = (
        merged["monthly_forecast_bm"] - merged["actual_monthly_total"]
    ).abs()
    merged["weighted_errors_bm"] = merged["series_weight"] * merged["benchmark_errors"]
    fold_totals = merged.groupby(_FOLD_KEYS)[
        ["weighted_errors_ch", "weighted_errors_bm"]
    ].sum()

    # compute the exclusion mask once, reuse for log and guard
    excluded_mask = ~(fold_totals["weighted_errors_bm"] > _SMALL_NUM_BOUND)
    if excluded_mask.any():
        _log.warning(
            "compute_wrmae_pooled: %d fold(s) excluded near-zero benchmark sum "
            "(tier=%r):\n%s",
            int(excluded_mask.sum()),
            tier,
            fold_totals[excluded_mask].to_string(),
        )
    # replace near-zero fold denominators before dividing to avoid inf propagation
    safe_denom = fold_totals["weighted_errors_bm"].where(~excluded_mask)

    per_fold_ratio = fold_totals["weighted_errors_ch"] / safe_denom

    # catch-all: nan in ratio that isn't explained by the benchmark guard
    unexpected_nan = per_fold_ratio.isna() & ~excluded_mask
    if unexpected_nan.any():
        _log.warning(
            "compute_wrmae_pooled: %d fold(s) have unexpected nan in per fold ratio "
            "(tier=%r): %s",
            int(unexpected_nan.sum()),
            tier,
            per_fold_ratio[unexpected_nan].index.tolist(),
        )

    return np.nanmean(per_fold_ratio.to_numpy(dtype=np.float64)).item()


def compute_wrmae_per_series(
    challenger_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    tier: str | None = None,
) -> float:
    """Fold-averaged per series WRMAE. tier=None -> all series."""
    if tier is not None:
        challenger_df = challenger_df[challenger_df["tier"] == tier]
        benchmark_df = benchmark_df[benchmark_df["tier"] == tier]

    ch_cols = _JOIN_KEYS + ["monthly_forecast", "actual_monthly_total", "series_weight"]
    bm_cols = _JOIN_KEYS + ["monthly_forecast"]

    merged = challenger_df[ch_cols].merge(
        benchmark_df[bm_cols],
        on=_JOIN_KEYS,
        suffixes=("_ch", "_bm"),
        how="inner",
    )
    if merged.empty:
        return np.nan

    merged["benchmark_errors"] = (
        merged["monthly_forecast_bm"] - merged["actual_monthly_total"]
    ).abs()

    mask = merged["benchmark_errors"] > _SMALL_NUM_BOUND
    n_excluded = int((~mask).sum())
    if n_excluded > 0:
        _log.warning(
            "compute_wrmae_per_series: %d rows excluded near zero benchmark error "
            "(tier=%r)",
            n_excluded,
            tier,
        )

    merged = merged[mask].copy()

    if merged.empty:
        return np.nan

    merged["challenger_errors"] = (
        merged["monthly_forecast_ch"] - merged["actual_monthly_total"]
    ).abs()

    fold_weight_sum = merged.groupby(_FOLD_KEYS)["series_weight"].transform("sum")
    merged["normalized_series_weights"] = merged["series_weight"] / fold_weight_sum
    merged["normalized_weighted_ratio"] = merged["normalized_series_weights"] * (
        merged["challenger_errors"] / merged["benchmark_errors"]
    )

    fold_totals = merged.groupby(_FOLD_KEYS)["normalized_weighted_ratio"].sum()

    return np.nanmean(fold_totals.to_numpy(dtype=np.float64)).item()


def compute_signed_bias_pooled(
    challenger_df: pd.DataFrame,
    tier: str | None = None,
) -> float:
    """Fold-averaged pooled signed relative bias. tier=None -> all series.

    Positive values indicate systematic over-forecasting.
    """
    if tier is not None:
        challenger_df = challenger_df[challenger_df["tier"] == tier]

    cols = _JOIN_KEYS + ["monthly_forecast", "actual_monthly_total", "series_weight"]
    df = challenger_df[cols].copy()

    if df.empty:
        return np.nan

    df["numerator"] = df["series_weight"] * (
        df["monthly_forecast"] - df["actual_monthly_total"]
    )
    df["denominator"] = df["series_weight"] * df["actual_monthly_total"].abs()

    fold_totals = df.groupby(_FOLD_KEYS)[["numerator", "denominator"]].sum()

    # compute the exclusion mask once, reuse for log and guard
    excluded_mask = ~(fold_totals["denominator"] > _SMALL_NUM_BOUND)
    if excluded_mask.any():
        _log.warning(
            "compute_signed_bias_pooled: %d folds excluded near zero weighted actuals "
            "(tier=%r):\n%s",
            int(excluded_mask.sum()),
            tier,
            fold_totals[excluded_mask].to_string(),
        )
    # replace near-zero fold denominators before dividing to avoid inf propagation
    safe_denom = fold_totals["denominator"].where(~excluded_mask)

    per_fold_bias = fold_totals["numerator"] / safe_denom

    # catch-all: nan in ratio that isn't explained by the denominator or exclusion guard
    unexpected_nan = per_fold_bias.isna() & ~excluded_mask
    if unexpected_nan.any():
        _log.warning(
            "compute_signed_bias_pooled: %d folds have unexpected nan in per fold bias "
            "(tier=%r): %s",
            int(unexpected_nan.sum()),
            tier,
            per_fold_bias[unexpected_nan].index.tolist(),
        )

    return np.nanmean(per_fold_bias.to_numpy(dtype=np.float64)).item()


def compute_signed_bias_per_series(
    challenger_df: pd.DataFrame,
    tier: str | None = None,
) -> float:
    """Fold-averaged per-series signed relative bias. tier=None -> all series.

    Positive values indicate systematic over-forecasting.
    """
    if tier is not None:
        challenger_df = challenger_df[challenger_df["tier"] == tier]

    cols = _JOIN_KEYS + ["monthly_forecast", "actual_monthly_total", "series_weight"]
    df = challenger_df[cols].copy()

    if df.empty:
        return np.nan

    df["abs_actual"] = df["actual_monthly_total"].abs()
    mask = df["abs_actual"] > _SMALL_NUM_BOUND

    n_excluded = int((~mask).sum())
    if n_excluded > 0:
        _log.warning(
            "compute_signed_bias_per_series: %d rows excluded near zero actuals "
            "(tier=%r)",
            n_excluded,
            tier,
        )

    df = df[mask].copy()
    if df.empty:
        return np.nan

    fold_weight_total = df.groupby(_FOLD_KEYS)["series_weight"].transform("sum")

    df["normalized_weight"] = df["series_weight"] / fold_weight_total

    df["per_series_bias"] = (df["monthly_forecast"] - df["actual_monthly_total"]) / df[
        "abs_actual"
    ]
    df["weighted_bias"] = df["normalized_weight"] * df["per_series_bias"]

    fold_totals = df.groupby(_FOLD_KEYS)["weighted_bias"].sum()

    return np.nanmean(fold_totals.to_numpy(dtype=np.float64)).item()


def compute_wape(
    challenger_df: pd.DataFrame,
    tier: str | None = None,
) -> float:
    """Fold-averaged WAPE. tier=None -> all series."""
    if tier is not None:
        challenger_df = challenger_df[challenger_df["tier"] == tier]

    cols = _JOIN_KEYS + ["monthly_forecast", "actual_monthly_total"]
    df = challenger_df[cols].copy()

    if df.empty:
        return np.nan

    df["abs_error"] = (df["monthly_forecast"] - df["actual_monthly_total"]).abs()
    df["abs_actual"] = df["actual_monthly_total"].abs()
    fold_totals = df.groupby(_FOLD_KEYS)[["abs_error", "abs_actual"]].sum()

    # compute the exclusion mask once, reuse for log and guard
    excluded_mask = ~(fold_totals["abs_actual"] > _SMALL_NUM_BOUND)
    if excluded_mask.any():
        _log.warning(
            "compute_wape: %d folds excluded near zero actuals (tier=%r):\n%s",
            int(excluded_mask.sum()),
            tier,
            fold_totals[excluded_mask].to_string(),
        )

    # replace near-zero fold denominators before dividing to avoid inf propagation
    safe_denom = fold_totals["abs_actual"].where(~excluded_mask)

    per_fold_wape = fold_totals["abs_error"] / safe_denom

    # catch-all: nan in ratio that isn't explained by the denominator or exclusion guard
    unexpected_nan = per_fold_wape.isna() & ~excluded_mask
    if unexpected_nan.any():
        _log.warning(
            "compute_wape: %d folds have unexpected nan in per fold wape (tier=%r): %s",
            int(unexpected_nan.sum()),
            tier,
            per_fold_wape[unexpected_nan].index.tolist(),
        )

    return np.nanmean(per_fold_wape.to_numpy(dtype=np.float64)).item()


def compute_weighted_signed_bias(
    challenger_df: pd.DataFrame,
    tier: str | None = None,
) -> float:
    """Fold-averaged WAPE-style signed bias. tier=None -> all series.

    Positive values indicate systematic over-forecasting.
    |Weighted Signed Bias| <= WAPE always holds.
    """
    if tier is not None:
        challenger_df = challenger_df[challenger_df["tier"] == tier]

    cols = _JOIN_KEYS + ["monthly_forecast", "actual_monthly_total"]
    df = challenger_df[cols].copy()

    if df.empty:
        return np.nan

    df["signed_error"] = df["monthly_forecast"] - df["actual_monthly_total"]
    df["abs_actual"] = df["actual_monthly_total"].abs()
    fold_totals = df.groupby(_FOLD_KEYS)[["signed_error", "abs_actual"]].sum()

    # compute the exclusion mask once, reuse for log and guard
    excluded_mask = ~(fold_totals["abs_actual"] > _SMALL_NUM_BOUND)
    if excluded_mask.any():
        _log.warning(
            "compute_weighted_signed_bias: %d folds excluded near zero actuals "
            "(tier=%r):\n%s",
            int(excluded_mask.sum()),
            tier,
            fold_totals[excluded_mask].to_string(),
        )

    # replace near-zero fold denominators before dividing to avoid inf propagation
    safe_denom = fold_totals["abs_actual"].where(~excluded_mask)

    per_fold_wsb = fold_totals["signed_error"] / safe_denom

    # catch-all: nan in ratio that isn't explained by the denominator or exclusion guard
    unexpected_nan = per_fold_wsb.isna() & ~excluded_mask
    if unexpected_nan.any():
        _log.warning(
            "compute_weighted_signed_bias: %d folds have unexpected nan in per fold "
            "weighted signed bias (tier=%r): %s",
            int(unexpected_nan.sum()),
            tier,
            per_fold_wsb[unexpected_nan].index.tolist(),
        )

    return np.nanmean(per_fold_wsb.to_numpy(dtype=np.float64)).item()


def _attach_progress(df: pd.DataFrame, origin_spine: pd.DataFrame) -> pd.DataFrame:
    """Join the spine's progress columns onto a fold frame, losing no rows.

    Args:
        df: Challenger or benchmark frame at series x fold grain, already
            filtered to a single horizon.
        origin_spine: enumerate_origins output with one row per origin, carrying
            target_month and the progress columns.

    Returns:
        df with _PROGRESS_COLS added; same row count.

    Raises:
        ValueError: If df already carries the progress columns, leaving the row count
            intact and the frame wrong. Or if any df row has no origin_spine entry,
            which is what an unfiltered multi-horizon frame produces.
        pd.errors.MergeError: If origin_spine is not unique on the fold keys.
            The public caller checks uniqueness first, so this is a backstop
            with a less specific message.
    """
    spine = origin_spine[["forecast_origin_date", "target_month"] + _PROGRESS_COLS]
    spine = spine.rename(columns={"target_month": "predicted_fiscal_year_month"})

    col_collisions = [c for c in df.columns if c in _PROGRESS_COLS]
    if col_collisions:
        raise ValueError(
            f"Input df has columns: {col_collisions}; these columns cause "
            "name collision during join within _attach_progress()"
        )

    lost = df.loc[
        ~df.set_index(_FOLD_KEYS).index.isin(spine.set_index(_FOLD_KEYS).index),
        _FOLD_KEYS,
    ]
    if not lost.empty:
        raise ValueError(
            f"{len(lost)} of {len(df)} rows have no origin_spine entry:\n"
            f"{lost.drop_duplicates().head(5).to_string(index=False)}"
        )

    out = df.merge(spine, on=_FOLD_KEYS, how="inner", validate="many_to_one")

    return out


def compute_wrmae_by_progress(
    challenger_df: pd.DataFrame, benchmark_df: pd.DataFrame, origin_spine: pd.DataFrame
) -> pd.DataFrame:
    """Fold-averaged pooled WRMAE per origin-progress cohort.

    The progress axis comes from origin_spine, not from either frame's own
    columns: the benchmark carries no progress columns, so a challenger-sourced
    axis cannot segment both sides of a skill ratio.

    Args:
        challenger_df: Series x fold frame, the columns compute_wrmae_pooled
            needs. Must be filtered to a single horizon first since horizon_2 rows
            predict M+1 while their origin's spine entry says M, so they match
            no spine key and trip the row-loss check in _attach_progress.
        benchmark_df: Same grain and horizon as challenger_df.
        origin_spine: enumerate_origins output, unique on
            (forecast_origin_date, target_month) and carrying the progress
            columns.

    Returns:
        One row per (weeks_in_month, weeks_actualized) cohort present in
        challenger_df, with n_events and wrmae_vs_benchmark, sorted by cohort.
        Empty with the declared columns when challenger_df is empty rather than
        raising.

    Raises:
        ValueError: If a frame lacks a required column, or origin_spine is not
            unique on its keys.
    """

    spine_keys = ["forecast_origin_date", "target_month"]
    require_columns(challenger_df, _FOLD_KEYS, "challenger_df")
    require_columns(benchmark_df, _FOLD_KEYS, "benchmark_df")
    require_columns(origin_spine, spine_keys + _PROGRESS_COLS, "origin_spine")

    dupes = origin_spine.loc[
        origin_spine.duplicated(spine_keys, keep=False), spine_keys
    ]
    if not dupes.empty:
        raise ValueError(
            f"origin_spine must be unique on {spine_keys}; "
            f"{len(dupes)} rows share {len(dupes.drop_duplicates())} key(s):\n"
            f"{dupes.drop_duplicates().head(5).to_string(index=False)}"
        )

    ch = _attach_progress(challenger_df, origin_spine)
    bm = _attach_progress(benchmark_df, origin_spine)
    rows = []
    for _, wim, wa in ch[_PROGRESS_COLS].drop_duplicates().itertuples():
        ch_cohort = ch[(ch["weeks_in_month"] == wim) & (ch["weeks_actualized"] == wa)]
        bm_cohort = bm[(bm["weeks_in_month"] == wim) & (bm["weeks_actualized"] == wa)]
        rows.append(
            {
                "weeks_in_month": wim,
                "weeks_actualized": wa,
                "n_events": len(ch_cohort),
                "wrmae_vs_benchmark": compute_wrmae_pooled(ch_cohort, bm_cohort),
            }
        )

    return (
        pd.DataFrame(rows, columns=_PROGRESS_SKILL_COLS)
        .sort_values(_PROGRESS_COLS)
        .reset_index(drop=True)
    )


def _wape_adapter(
    challenger_df: pd.DataFrame,
    benchmark_df: pd.DataFrame | None,
    tier: str | None,
) -> float:
    """
    Adapter needed since WAPE doesn't use a benchmark_df but needs a signature
    consistent with WRMAE functions
    """
    return compute_wape(challenger_df, tier)


_RELATIVE_METRICS: frozenset[str] = frozenset({"wrmae_pooled", "wrmae_per_series"})

HERO_METRICS: dict[str, Callable[..., float]] = {
    "wrmae_pooled": compute_wrmae_pooled,
    "wrmae_per_series": compute_wrmae_per_series,
    "wape": _wape_adapter,
}


def compute_hero_metric(
    challenger_df: pd.DataFrame,
    benchmark_df: pd.DataFrame | None,
    metric_name: str,
    target: str,
) -> float:
    """Filters to rows where horizon == target, then dispatches to
    HERO_METRICS[metric_name].

    target: 'horizon_1' | 'horizon_2' | ...
    benchmark_df may be None when metric_name == 'wape'.
    Raises ValueError for unknown metric_name or relative metric with benchmark_df=None.
    """

    if metric_name not in HERO_METRICS:
        raise ValueError(
            f"Called metric {metric_name} not in allowed hero metrics "
            f"list. Allowed hero metrics are: {list(HERO_METRICS)}"
        )

    if metric_name in _RELATIVE_METRICS and benchmark_df is None:
        raise ValueError(
            f"Metric {metric_name} is listed as a relative metric "
            f"but benchmark_df not provided. Relative metrics must have a "
            f"benchmark_df provided."
        )

    challenger_df = challenger_df[challenger_df["horizon"] == target]
    if benchmark_df is not None:
        benchmark_df = benchmark_df[benchmark_df["horizon"] == target]

    return HERO_METRICS[metric_name](challenger_df, benchmark_df, None)
