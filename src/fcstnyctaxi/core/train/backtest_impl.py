"""The weekly fold loop, ported from `backtest_weekly.py`'s `evaluate_model`.

Near-verbatim: the port's only proof is a manual diff against the notebook.
"""

import logging
from collections import Counter
from dataclasses import dataclass
from typing import cast

import pandas as pd
from tsbricks.backtesting import evaluate_metrics, generate_folds
from tsbricks.backtesting.schema import AggregationConfig, BacktestConfig
from tsbricks.runner import (
    apply_transforms,
    fit_transforms,
    inverse_transforms,
    invoke_model,
)

from fcstnyctaxi.lib.backtest_results import build_backtest_results, build_cv_results
from fcstnyctaxi.lib.monthly_aggregation import (
    attach_tier_and_weight,
    build_monthly_forecast_vs_actual,
    compute_actual_monthly_totals,
)
from fcstnyctaxi.lib.period_utils import (
    DAMPENING_FNS,
    ORIGIN_TIME_UNIT,
    assign_tiers,
    compute_series_weights,
    normalized_origin_horizon_pairs,
)
from fcstnyctaxi.schemas.config.train import TrainModelingConfig

_log = logging.getLogger(__name__)

_MONTHLY_SERIES_KEYS = [
    "forecast_origin_date",
    "predicted_fiscal_year_month",
    "unique_id",
]
_RAW_CV_FORECAST_COLUMNS = [
    "forecast_origin_date",
    "unique_id",
    "ds",
    "ypred",
    "fold_id",
]


@dataclass(frozen=True)
class BacktestOutputs:
    """The four frames the sidecar persists; the field names are the filenames."""

    monthly_series: pd.DataFrame
    monthly_forecast_components: pd.DataFrame
    metrics: pd.DataFrame
    raw_cv_forecasts: pd.DataFrame


def compute_backtest_outputs(
    *,
    cfg: BacktestConfig,
    modeling: TrainModelingConfig,
    ts_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
) -> BacktestOutputs:
    """Run the composable fold loop for one model config.

    Two configs: `BacktestConfig` is tsbricks', tiering and weighting this project's.
    `actual_monthly_df` is derived here, so actuals and folds cannot differ in panel.

    Args:
        cfg: Composed tsbricks config; its forecast_origins drive the loop.
        modeling: Only `tiering` and `weighting` are read.
        ts_df: The trimmed weekly panel.
        calendar_df: The trimmed fiscal calendar, also the future exogenous table.

    Raises:
        ValueError: If origins repeat, if the fold count disagrees with the
            origin/horizon pairs, if a raw forecast row ends with a null origin, or
            if `monthly_series` keys are not unique.

    Returns:
        BacktestOutputs: The four frames, one per sidecar file.
    """
    # Optional in the tsbricks schema, always declared by the composed config.
    period_col = cast(AggregationConfig, cfg.aggregation).period_col

    actual_monthly_df = compute_actual_monthly_totals(
        ts_df, calendar_df, period_col=period_col
    )

    # -- 1. generate folds --------------------
    cv_folds, _ = generate_folds(ts_df, cfg.cross_validation, cfg.data)

    # -- 2. weekly fold loop: fit -> invoke -> inverse -> evaluate ------
    per_fold_metrics = []
    per_fold_forecasts: dict[str, pd.DataFrame] = {}

    origin_horizon_pairs = normalized_origin_horizon_pairs(
        cfg.cross_validation.origin_horizon_pairs(), cfg.data.freq
    )
    monthly_series_rows = []
    components_rows = []

    # Replaces the notebook's `seen_origins` skip; outputs are keyed by fold.
    repeated = sorted(
        origin
        for origin, count in Counter(o for o, _ in origin_horizon_pairs).items()
        if count > 1
    )
    if repeated:
        raise ValueError(
            f"forecast origins repeat {repeated}, so two folds would carry one "
            "forecast_origin_date and every output file would key those rows twice."
        )
    if len(cv_folds) != len(origin_horizon_pairs):
        raise ValueError(
            f"fold count {len(cv_folds)} against {len(origin_horizon_pairs)} "
            "origin/horizon pairs, which the loop indexes by fold position: more "
            "folds raises IndexError mid-loop, fewer runs the whole backtest "
            "against a prefix."
        )

    fraction_by_origin = calendar_df.set_index("ds")["origin_month_fraction_elapsed"]

    for fold_idx, (fold_id, splits) in enumerate(cv_folds.items()):
        fold_origin, fold_horizon = origin_horizon_pairs[fold_idx]
        _log.info("fold origin: %s, fold horizon: %s", fold_origin, fold_horizon)

        train, val = splits["train"], splits["val"]

        # tier and weight at this fold's origin
        tier_df = assign_tiers(
            train,
            fold_origin,
            calendar_df,
            trailing_weeks=modeling.tiering.trailing_weeks,
            tier_labels=tuple(modeling.tiering.tier_labels),
        )
        weight_df = compute_series_weights(
            train,
            fold_origin,
            calendar_df,
            trailing_weeks=modeling.weighting.trailing_weeks,
            dampening_fn=DAMPENING_FNS[modeling.weighting.dampening],
        )

        fitted_transforms, train_t = fit_transforms(train, cfg.transforms or [])

        # Ported untouched; the notebook does not say whether it mutates. Leave it
        # until a committed regression golden can protect the answer.
        _ = apply_transforms(val, fitted_transforms)  # here for consistency

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
            period_col=period_col,
            time_col="ds",
            id_col="unique_id",
            forecast_col="ypred",
            target_col="y",
        )
        fold_rows = attach_tier_and_weight(
            monthly_rows_df=monthly_rows_df,
            tier_df=tier_df,
            weight_df=weight_df,
            fold_origin=fold_origin,
            origin_month_fraction_elapsed=fraction_by_origin[fold_origin],
            period_col=period_col,
            id_col="unique_id",
        )

        monthly_series_rows.append(fold_rows)
        components_rows.append(
            monthly_rows_df.assign(forecast_origin_date=fold_origin)[
                [
                    "forecast_origin_date",
                    period_col,
                    "unique_id",
                    "mtd_revenue",
                    "predicted_remaining",
                ]
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

    duplicated = monthly_series_df.duplicated(_MONTHLY_SERIES_KEYS)
    if duplicated.any():
        sample = monthly_series_df.loc[duplicated, _MONTHLY_SERIES_KEYS].head(3)
        raise ValueError(
            f"monthly_series has {int(duplicated.sum())} duplicate key row(s), which "
            "inflate every weighted sum in every score; first few: "
            f"{sample.to_dict('records')}"
        )

    monthly_forecast_components_df = pd.concat(
        components_rows, ignore_index=True
    ).rename(columns={period_col: "predicted_fiscal_year_month"})

    # -- 3. build weekly results -----------------------------
    cv_results = build_cv_results(
        forecasts_per_fold=per_fold_forecasts,
        train_val_splits_per_fold=cv_folds,
        metrics=metrics,
        origin_horizon_pairs=origin_horizon_pairs,
    )
    # capture_lineage=False: both fields fed only run_metadata.json, which this
    # sidecar drops, and both return None in the Train container. Do not restore.
    backtest_results = build_backtest_results(
        cv=cv_results,
        config=cfg.model_dump(by_alias=True, exclude_none=True),
        origin_horizon_pairs=origin_horizon_pairs,
        capture_lineage=False,
    )

    # -- 4. raw forecasts at native weekly grain --------------
    cv_forecasts_df = pd.concat(
        [
            df.assign(fold_id=fold_id)
            for fold_id, df in backtest_results.cv.forecasts_per_fold.items()
        ],
        ignore_index=True,
    )
    cv_forecasts_df["forecast_origin_date"] = (
        cv_forecasts_df["fold_id"]
        .map(backtest_results.cv.fold_id_to_origin)
        .astype(f"datetime64[{ORIGIN_TIME_UNIT}]")  # one unit across every output file
    )

    unmapped = int(cv_forecasts_df["forecast_origin_date"].isna().sum())
    if unmapped:
        raise ValueError(
            f"{unmapped} raw forecast row(s) carry a null forecast_origin_date: "
            "fold_id_to_origin does not cover every fold in the forecasts."
        )

    # Explicit reindex: validate_sidecar asserts the first column is the origin.
    cv_forecasts_df = cv_forecasts_df[_RAW_CV_FORECAST_COLUMNS]

    return BacktestOutputs(
        monthly_series=monthly_series_df,
        monthly_forecast_components=monthly_forecast_components_df,
        metrics=backtest_results.cv.metrics,
        raw_cv_forecasts=cv_forecasts_df,
    )
