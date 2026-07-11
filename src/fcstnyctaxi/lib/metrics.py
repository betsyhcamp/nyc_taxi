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


def _bad_numerator_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> bool:
    """Helper function that checks shape of input arrays for forecast error
    for numerator of metrics. Also checks if contents of input arrays are finite.
    """
    if y_true.ndim != 1 or y_pred.ndim != 1 or (y_true.shape != y_pred.shape):
        raise ValueError("y_true and y_pred must be 1D arrays of same shape.")

    is_finite = np.all(np.isfinite(y_true)) and np.all(np.isfinite(y_pred))
    return not is_finite or y_true.size == 0


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


def wrmae_pooled(
    challenger_errors: np.ndarray,
    benchmark_errors: np.ndarray,
    weights: np.ndarray,
) -> float:
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
    # IF _scale_is_invalid(total_weight): return nan
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
