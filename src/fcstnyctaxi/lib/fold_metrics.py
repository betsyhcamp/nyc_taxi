from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from fcstnyctaxi.lib.metrics import (
    _SMALL_NUM_BOUND,
)

_log = logging.getLogger(__name__)

_FOLD_KEYS = ["forecast_origin_date", "predicted_fiscal_year_month"]
_JOIN_KEYS = _FOLD_KEYS + ["unique_id"]


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
