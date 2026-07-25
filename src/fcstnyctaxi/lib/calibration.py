from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd
from tsbricks.backtesting import generate_folds
from tsbricks.backtesting.schema import CrossValidationConfig, DataConfig
from tsbricks.runner import invoke_model

from fcstnyctaxi.lib.cross_validation_utils import sorted_origin_horizon_pairs
from fcstnyctaxi.lib.metrics import weighted_mae
from fcstnyctaxi.lib.monthly_aggregation import build_monthly_forecast_vs_actual
from fcstnyctaxi.lib.period_utils import compute_series_weights, derive_horizon_label


def calibrate_n_estimators(
    ts_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    actual_monthly_df: pd.DataFrame,
    model_config: Any,
    n_estimators_grid: Sequence[int],
    calibration_origins: Sequence[
        tuple[Any, int]
    ],  # (origin, horizon) pairs, disjoint from real backtest origins
    set_truncation_iteration: Callable[[Any, int], None],
    predict_fn: Callable[[Any, int, pd.DataFrame | None], pd.DataFrame],
    scoring_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float] = weighted_mae,
) -> pd.DataFrame:
    """Calibrate a GBDT model's round count against held-out historical data.

    For each calibration origin, fits once via invoke_model() at max(n_estimators_grid),
    then sweeps the grid by truncating predictions via set_truncation_iteration and
    repredicting via predict_fn, rather than retraining. Scores each candidate at the
    monthly aggregated, horizon split level using scoring_fn.

    Both set_truncation_iteration and predict_fn are injected, model-specific
    adapters keeping this function agnostic across model families.

    Args:
        ts_df: Full historical panel, used to generate calibration folds.
        calendar_df: Fiscal calendar; also passed through untouched as the
            future exogenous table for both the ceiling fit and predict_fn.
        actual_monthly_df: Realized monthly totals; precomputed once outside
            this function, parallel to evaluate_model()'s own parameter
            see fcstnyctaxi.lib.monthly_aggregation.compute_actual_monthly_totals.
        set_truncation_iteration: Mutates a fitted model_obj in place to
            truncate it to k boosting rounds for the next predict.
        predict_fn: Predicts from an already-fitted model_obj without
            refitting the model specific "predict-only" adapter,
            e.g. models.lightgbm_weekly.lightgbm_weekly_predict.

    Returns:
        DataFrame with columns: origin, n_estimators, horizon, score.
        So scoring is per-origin, per-candidate; not pooled.
    """
    calibration_cv_config = CrossValidationConfig(
        mode="explicit",
        forecast_origins=[{"origin": o, "horizon": h} for o, h in calibration_origins],
    )

    data_config = DataConfig(freq=model_config.hyperparameters["freq"])

    cv_folds, _ = generate_folds(ts_df, calibration_cv_config, data_config)

    origin_horizon_pairs = sorted_origin_horizon_pairs(
        calibration_cv_config.origin_horizon_pairs(), data_config.freq
    )

    fraction_by_origin = calendar_df.set_index("ds")["origin_month_fraction_elapsed"]
    fiscal_month_by_origin = calendar_df.set_index("ds")["fiscal_year_month"]
    ceiling_hyperparameters = {
        **model_config.hyperparameters,
        "n_estimators": max(n_estimators_grid),
    }
    ceiling_config = model_config.model_copy(
        update={"hyperparameters": ceiling_hyperparameters}
    )
    scored_results: list[dict] = []
    for fold_idx, (_, splits) in enumerate(cv_folds.items()):
        origin, horizon = origin_horizon_pairs[fold_idx]
        train = splits["train"]

        # one fit, at the ceiling (unchanged production path)
        _, _, mlfcst = invoke_model(
            train, ceiling_config, horizon, future_x_df=calendar_df
        )

        # once per origin; doesn't vary with k
        weight_df = compute_series_weights(
            train, cast(pd.Timestamp, origin), calendar_df
        )

        for k in n_estimators_grid:
            set_truncation_iteration(mlfcst, k)
            # predict_fn owns feature building internally so current script doesn't
            truncated_estimators_fcst = predict_fn(mlfcst, horizon, calendar_df)

            monthly_rows = build_monthly_forecast_vs_actual(
                forecast_df=truncated_estimators_fcst,
                train_df=train,
                calendar_df=calendar_df,
                actual_monthly_df=actual_monthly_df,
            )

            monthly_rows = monthly_rows.assign(
                horizon_label=derive_horizon_label(
                    monthly_rows["fiscal_year_month"],
                    fiscal_month_by_origin[origin],
                    fraction_by_origin[origin],
                )
            ).merge(weight_df, on="unique_id", how="left")

            for horizon_label, group in monthly_rows.groupby("horizon_label"):
                score = scoring_fn(
                    group["actual_monthly_total"].to_numpy(),
                    group["monthly_forecast"].to_numpy(),
                    group["series_weight"].to_numpy(),
                )
                scored_results.append(
                    {
                        "origin": origin,
                        "n_estimators": k,
                        "horizon": horizon_label,
                        "score": score,
                    }
                )
    return pd.DataFrame(scored_results)
