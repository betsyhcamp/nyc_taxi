from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from tsbricks.blocks.metrics import wape, weighted_signed_bias

from fcstnyctaxi.lib.metrics import (
    _SMALL_NUM_BOUND,
    signed_bias_per_series,
    signed_bias_pooled,
    wrmae_per_series,
    wrmae_pooled,
)

_log = logging.getLogger(__name__)

_FOLD_KEYS = ["forecast_origin_date", "predicted_fiscal_year_month"]
_JOIN_KEYS = _FOLD_KEYS + ["unique_id"]


def compute_wrmae_pooled(
    challenger_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    tier: str | None = None,
) -> float:
    """Fold-averaged pooled WRMAE (V1). tier=None → all series."""
    if tier is not None:
        challenger_df = challenger_df[challenger_df["tier"] == tier]
        benchmark_df = benchmark_df[benchmark_df["tier"] == tier]
    merged = challenger_df.merge(
        benchmark_df,
        on=_JOIN_KEYS,
        suffixes=("_ch", "_bm"),
        how="inner",
    )
    if merged.empty:
        return np.nan
    # TODO: vectorized fold-level aggregation
    # Column names after join:
    #   monthly_forecast_ch, actual_monthly_total_ch, series_weight_ch
    #   monthly_forecast_bm, actual_monthly_total_bm
    #
    # Steps:
    # 1. Compute absolute errors for challenger and benchmark.
    # 2. Pre-compute series_weight_ch * error columns for both sides.
    # 3. merged.groupby(_FOLD_KEYS)[weighted cols].sum() → fold-level totals.
    # 4. Per-fold ratio = weighted_ch_sum / weighted_bm_sum.
    # 5. Return float(np.nanmean(per_fold_ratio.values))
