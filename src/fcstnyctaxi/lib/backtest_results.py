"""Assemble tsbricks CVResults/BacktestResults from a per-fold loop.

Mirrors the intended interface for tsbricks.backtesting.results.build_cv_results/
build_backtest_results; lives here until the API is validated through use in
this project
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from tsbricks.backtesting import BacktestResults, CVResults
from tsbricks.blocks.metadata import get_git_hash, get_uv_lock_info


def build_cv_results(
    forecasts_per_fold: dict[str, pd.DataFrame],
    train_val_splits_per_fold: dict[str, dict[str, pd.DataFrame]],
    metrics: pd.DataFrame,
    origin_horizon_pairs: list[tuple],
    **optional_fields: Any,
) -> CVResults:
    """Assemble CVResults from a per-fold loop's outputs.

    Args:
        forecasts_per_fold: Fold ID to forecast DataFrame, original scale.
        train_val_splits_per_fold: Fold ID to {"train": df, "val": df},
            e.g. the dict returned by tsbricks.backtesting.generate_folds.
        metrics: Long-format per-fold metrics DataFrame (evaluate_metrics output,
            concatenated across folds).
        origin_horizon_pairs: (origin, horizon) tuples in fold order, e.g.
            cfg.cross_validation.origin_horizon_pairs().
        **optional_fields: Forwarded to CVResults (fitted_values,
            fitted_values_model_scale, transform_params, fitted_models, ...).

    Returns:
        A CVResults ready to pass into build_backtest_results.

    Raises:
        ValueError: If the fold keyed inputs and origin_horizon_pairs disagree on fold
            count, per the validation strategy below.
        ValueError: If forecasts_per_fold and train_val_splits_per_fold have
            the same fold count but different fold_id keys.
    """
    # Cannot catch a same length reorder of origin_horizon_pairs relative to
    # forecasts_per_fold's key order; limitation of origin_horizon_pairs being list.
    # So, code at call site should use sorted_origin_horizon_pairs() for this reason
    if len(origin_horizon_pairs) != len(forecasts_per_fold) or len(
        origin_horizon_pairs
    ) != len(train_val_splits_per_fold):
        raise ValueError(
            "lengths of origin horizon pairs, forecast folds, or train validation "
            "splits are mismatched."
        )

    if forecasts_per_fold.keys() != train_val_splits_per_fold.keys():
        raise ValueError(
            "Identities of fold ids are not the same between forecast results and "
            "train/val splits."
        )

    fold_ids = list(forecasts_per_fold.keys())
    fold_origins = [origin for origin, _ in origin_horizon_pairs]

    return CVResults(
        forecasts_per_fold=forecasts_per_fold,
        metrics=metrics,
        fold_origins=fold_origins,
        fold_id_to_origin=dict(zip(fold_ids, fold_origins, strict=True)),
        train_val_splits_per_fold=train_val_splits_per_fold,
        **optional_fields,
    )


def build_backtest_results(
    cv: CVResults,
    config: dict,
    origin_horizon_pairs: list[tuple],
    capture_lineage: bool = True,
    **optional_fields: Any,
) -> BacktestResults:
    """Wrap CVResults into a BacktestResults, optionally capturing git/uv lineage.

    Args:
        cv: Output of build_cv_results.
        config: Raw config dict, e.g.
            cfg.model_dump(by_alias=True, exclude_none=True).
        origin_horizon_pairs: Same (origin, horizon) tuples passed to
            build_cv_results. NOTE: this is deliberately not derived from
            cv.fold_id_to_origin since that dict maps fold_id -> origin, with no
            horizon, so it cannot reconstruct (origin, horizon) pairs.
        capture_lineage: If True, populate git_hash/uv_lock_info via the same
            tsbricks.blocks.metadata helpers run_backtest() uses internally.
        **optional_fields: Forwarded to BacktestResults (test, aggregated, extra).

    Returns:
        A BacktestResults ready to pass into tsbricks.backtesting.aggregate_backtest.

    Raises:
        ValueError: If provided CVResults and origin_horizon_pairs disagree
            on fold origins.
    """

    is_same_origins = [o for o, _ in origin_horizon_pairs] == cv.fold_origins
    if not is_same_origins:
        raise ValueError(
            "fold origins in crossval results not same as origin_horizon_pairs arg"
        )

    return BacktestResults(
        cv=cv,
        horizon=list(origin_horizon_pairs),
        config=config,
        git_hash=get_git_hash() if capture_lineage else None,
        uv_lock_info=get_uv_lock_info() if capture_lineage else None,
        **optional_fields,
    )
