from __future__ import annotations

import warnings

import numpy as np

# Small number bound to use in checking stability of numerical results
_SMALL_NUM_BOUND = max(np.finfo(float).tiny, 1e-12)


def _check_1d_same_shape(*arrays: np.ndarray) -> None:
    if any(a.ndim != 1 for a in arrays):
        raise ValueError("All arrays must be 1-D.")
    shapes = {a.shape for a in arrays}
    if len(shapes) > 1:
        raise ValueError(f"Array shapes must match; got {shapes}.")


def _scale_is_invalid(scale: float) -> bool:
    return not np.isfinite(scale) or (scale <= _SMALL_NUM_BOUND)


def _sanitize_value(x: float) -> float:
    """Helper function to act as a final guard to avoid returning +/-inf"""
    if np.isfinite(x):
        return float(x)
    warnings.warn(
        "_sanitize_value received non-finite input. Check upstream guards.",
        RuntimeWarning,
        stacklevel=1,
    )
    return np.nan


def weighted_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Cube-root-weighted mean absolute error (MAE) for a single scoring window.

    Computes sum(w * |y_pred - y_true|) / sum(w). Scoring-only — never
    touches the LightGBM training loss (still plain unweighted L1).

    Args:
        y_true: Actual values, shape (n,).
        y_pred: Predicted values, shape (n,).
        weights: Non-negative series weights, shape (n,).

    Returns:
        Weighted mean absolute error, or nan on empty input, non-finite
        values, or near-zero total weight.
    """
    # shape guard: error if arrays mismatched
    _check_1d_same_shape(y_true, y_pred, weights)

    # empty guard for data edge case
    if y_true.size == 0:
        return np.nan

    # non-finite guard
    input_arrays = (y_true, y_pred, weights)
    if not all(np.all(np.isfinite(arr)) for arr in input_arrays):
        return np.nan

    total_weight = np.sum(weights)
    if _scale_is_invalid(total_weight):
        return np.nan

    weighted_abs_error = np.sum(weights * np.abs(y_pred - y_true))
    return _sanitize_value(weighted_abs_error / total_weight)


def wrmae_pooled(
    challenger_errors: np.ndarray,
    benchmark_errors: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Pooled Weighted Relative MAE for a single fold.

    Computes sum(w * |e_challenger|) / sum(w * |e_benchmark|). Applies abs() internally,
    so signed residuals and pre-computed absolute errors are both valid inputs.

    Args:
        challenger_errors: Forecast errors for the challenger, shape (n,).
        benchmark_errors: Forecast errors for the benchmark, shape (n,).
        weights: Non-negative series weights, shape (n,).

    Returns:
        Ratio of weighted challenger error to weighted benchmark error,
        or nan on empty input, non-finite values, or near-zero denominator.
    """
    # belt-and-suspenders: in case errors are signed, make into absolute value
    challenger_errors = np.abs(challenger_errors)
    benchmark_errors = np.abs(benchmark_errors)

    # shape guard: error if arrays mismatched
    _check_1d_same_shape(challenger_errors, benchmark_errors, weights)

    # empty guard for data edge case
    if challenger_errors.size == 0:
        return np.nan

    # non-finite guard
    input_arrays = (challenger_errors, benchmark_errors, weights)
    if not all(np.all(np.isfinite(arr)) for arr in input_arrays):
        return np.nan

    numerator = np.sum(weights * challenger_errors)
    denominator = np.sum(weights * benchmark_errors)

    # small denominator guard: return nan if zero or near-zero
    if _scale_is_invalid(denominator):
        return np.nan

    # return sanitized ratio
    return _sanitize_value(numerator / denominator)


def wrmae_per_series(
    challenger_errors: np.ndarray,
    benchmark_errors: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Per-series Weighted Relative MAE for a single fold.

    Computes sum(w_norm_i * |e_c_i| / |e_b_i|), where w_norm renormalizes over
    surviving series. Series with near-zero benchmark error are excluded
    before computing per-series ratios. Applies abs() internally,
    so signed residuals and pre-computed absolute errors are both valid inputs.

    Args:
        challenger_errors: Forecast errors for the challenger, shape (n,).
        benchmark_errors: Forecast errors for the benchmark, shape (n,).
            Near-zero entries are excluded from the computation.
        weights: Non-negative series weights, shape (n,).

    Returns:
        Weighted mean of per-series error ratios, or nan on empty input,
        non-finite values, or when all benchmark errors are near-zero.
    """
    # belt-and-suspenders: in case errors are signed, make into absolute value
    challenger_errors = np.abs(challenger_errors)
    benchmark_errors = np.abs(benchmark_errors)

    # shape guard: error if arrays mismatched
    _check_1d_same_shape(challenger_errors, benchmark_errors, weights)

    # empty guard for data edge case
    if challenger_errors.size == 0:
        return np.nan

    # non-finite guard
    input_arrays = (challenger_errors, benchmark_errors, weights)
    if not all(np.all(np.isfinite(arr)) for arr in input_arrays):
        return np.nan

    # Exclude series where benchmark is zero or near-zero
    mask = benchmark_errors > _SMALL_NUM_BOUND

    if not np.any(mask):
        return np.nan
    # apply mask to all three arrays
    challenger_errors = challenger_errors[mask]
    benchmark_errors = benchmark_errors[mask]
    weights = weights[mask]

    # normalize weights over remaining series
    total_weight = np.sum(weights)

    # total_weight near zero means all remaining series had zero weight
    if _scale_is_invalid(total_weight):
        return np.nan

    weights_normalized = weights / total_weight

    # per series error ratios
    ratios = challenger_errors / benchmark_errors

    # Guard against non-finite error ratios
    if not np.all(np.isfinite(ratios)):
        return np.nan

    # weighted mean of ratios
    return _sanitize_value(np.sum(weights_normalized * ratios))


def signed_bias_pooled(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Pooled weighted signed relative bias for a single fold.

    Computes sum(w * (y_pred - y_true)) / sum(w * |y_true|). Positive
    values indicate systematic over-forecasting; negative values indicate
    systematic under-forecasting.

    Args:
        y_true: Actual values, shape (n,).
        y_pred: Predicted values, shape (n,).
        weights: Non-negative series weights, shape (n,).

    Returns:
        Signed bias as a fraction of total weighted actual magnitude,
        or nan on empty input, non-finite values, or near-zero denominator.
    """
    # shape guard
    _check_1d_same_shape(y_true, y_pred, weights)

    # empty guard
    if y_true.size == 0:
        return np.nan

    # non-finite guard
    input_arrays = (y_true, y_pred, weights)
    if not all(np.all(np.isfinite(arr)) for arr in input_arrays):
        return np.nan

    # compute numerator and denominator
    numerator = np.sum(weights * (y_pred - y_true))
    denominator = np.sum(weights * np.abs(y_true))

    # denominator guard
    if _scale_is_invalid(denominator):
        return np.nan

    # return sanitized ratio
    return _sanitize_value(numerator / denominator)


def signed_bias_per_series(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Per-series weighted signed relative bias for a single fold.

    Computes sum(w_norm_i * (y_pred_i - y_true_i) / |y_true_i|), where
    w_norm renormalizes over remaining series. Series with near-zero
    |y_true| are excluded before computing per-series bias.

    Args:
        y_true: Actual values, shape (n,). Near-zero entries are excluded.
        y_pred: Predicted values, shape (n,).
        weights: Non-negative series weights, shape (n,).

    Returns:
        Weighted mean of per-series signed relative bias. Positive values
        indicate systematic over-forecasting. Returns nan on empty input,
        non-finite values, or when all |y_true| values are near-zero.
    """
    # shape guard
    _check_1d_same_shape(y_true, y_pred, weights)

    if y_true.size == 0:
        return np.nan

    # non-finite guard
    input_arrays = (y_true, y_pred, weights)
    if not all(np.all(np.isfinite(arr)) for arr in input_arrays):
        return np.nan

    # Exclude series where |y_true| is zero or near-zero
    mask = np.abs(y_true) > _SMALL_NUM_BOUND
    if not np.any(mask):
        return np.nan

    y_true = y_true[mask]
    y_pred = y_pred[mask]
    weights = weights[mask]

    # Normalize weights over remaining series
    total_weight = np.sum(weights)

    if _scale_is_invalid(total_weight):
        return np.nan

    weights_normalized = weights / total_weight

    # Per-series signed relative bias
    per_series_bias = (y_pred - y_true) / np.abs(y_true)

    # Guard: if any per-series value is non-finite
    if not np.all(np.isfinite(per_series_bias)):
        return np.nan

    # Weighted mean; result can be negative
    return _sanitize_value(np.sum(weights_normalized * per_series_bias))
