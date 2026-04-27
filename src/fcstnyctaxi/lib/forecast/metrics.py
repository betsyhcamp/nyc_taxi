from __future__ import annotations

from typing import Iterable, Literal, Tuple

import numpy as np

# Small number bound to use in checking stability of numerical results
_SMALL_NUM_BOUND = max(np.finfo(float).tiny, 1e-12)


def _bad_numerator_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> bool:
    """Helper function that checks shape of input arrays for forecast error
    for numerator of metrics. Also checks if contents of input arrays are finite.
    """
    if y_true.ndim != 1 or y_pred.ndim != 1 or (y_true.shape != y_pred.shape):
        raise ValueError("y_true and y_pred must be 1D arrays of same shape.")

    is_finite = np.all(np.isfinite(y_true)) and np.all(np.isfinite(y_pred))
    return not is_finite or y_true.size == 0


def _bad_denominator_inputs(y_train: np.ndarray, m: int) -> bool:
    """Helper function that checks shape of input arrays for forecast error
    for denominator of metrics. Checks if difference, m, is >=1.
    Also checks if contents of input arrays are finite.
    """
    if m < 1:
        raise ValueError("m must be >= 1.")
    if y_train.ndim != 1:
        raise ValueError("y_train must be 1D array.")
    is_finite = np.all(np.isfinite(y_train))
    return y_train.size <= m or not is_finite


def _scale_is_invalid(scale: float) -> bool:
    return not np.isfinite(scale) or (scale <= _SMALL_NUM_BOUND)


def _sanitize_value(x: float) -> float:
    """Helper function to act as a final guard to avoid returning +/-inf"""
    return float(x) if np.isfinite(x) else float("nan")


def difference_scale(
    y_train: Iterable[float], m: int = 1, scale_stat: Literal["rms", "meanabs"] = "rms"
) -> Tuple[float, bool]:
    """Return the Root Mean Square Differences or Mean Absolute Differences as
       a scale where the difference is between subsequent points m indicies apart

    Args:
        y_train (Iterable[float]): The 1D iterable of data to use to contruct
            the scale.
        m (int): The number of indices between datapoints that will be
            subtracted. Defaults to 1.
        scale_stat (Literal["rms", "meanabs"], optional): Desired scale
            statistic. Either "rms" or "meanabs". Defaults to "rms".

    Raises:
        ValueError: Raises ValueError if m<1, y_train is not a 1D array,
            or scale stat is not in {'rms', 'meanabs'}.

    Returns:
        Tuple[float, bool]: value of scale as float, bad scale (True/False)
    """

    y_train = np.asarray(y_train, dtype=float)

    if _bad_denominator_inputs(y_train, m):
        return (np.nan, True)

    # m-lag diffs
    train_diffs = y_train[m:] - y_train[:-m]

    # Scale statistic
    if scale_stat == "rms":
        scale_val = float(np.sqrt(np.mean(train_diffs**2)))
    elif scale_stat == "meanabs":
        scale_val = float(np.mean(np.abs(train_diffs)))
    else:
        raise ValueError("scale_stat must be 'rms' or 'meanabs'.")

    if _scale_is_invalid(scale_val):
        return (np.nan, True)

    return (scale_val, False)


def rmsse(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    *,
    m: int = 1,
    y_train: Iterable[float] | None = None,
    fallback_scale: float | None = None,
    return_components: bool = False,
) -> Tuple[float, bool] | Tuple[float, float, float, bool]:
    """
    Return the Root Mean Squared Scaled Error (RMSSE) for a forecast, using an
    m-lag difference-based scale computed from the in-sample training series.
    Can use provided scale from `fallback_scale`.

    Args:
        y_true (Iterable[float]): Actual values for the evaluation window,
            aligned to `y_pred`.
        y_pred (Iterable[float]): Forecast values aligned to `y_true`.
        m (int): The number of indices between datapoints used to construct
            the denominator scale (m-lag differences). Defaults to 1.
        y_train (Iterable[float] | None): The 1D in-sample series used to construct
            scaling term via m-lag differences. None if scale provided in
            `fallback_scale`.
        fallback_scale (float | None): Scale to use if y_train not provided.
        return_components (bool): If True, also return the numerator MSE and
            the denominator scale, along with instability flag. Defaults to False.

    Raises:
        ValueError: Either `y_train` or `fallback_scale` must be provided. Both
            cannot be None. If both provided, `y_train` takes precedence.

    Returns:
        Tuple[float, bool]: If `return_components` is False, returns
            (rmsse_value, unstable_scale) where:
              - rmsse_value (float): The RMSSE, or NaN if inputs invalid
                or scale unstable.
              - unstable_scale (bool): True if denominator scale zero,
                non-finite, or unusable; False otherwise.

        Tuple[float, float, float, bool]: If `return_components` is True, returns
            (rmsse_value, mse, scale, unstable_scale) where:
              - rmsse_value (float): The RMSSE, or NaN on invalid/unstable inputs.
              - mse (float): Mean squared error of `y_true - y_pred`, or NaN if
                numerator invalid.
              - scale (float): The m-lag difference scale computed from `y_train`,
                or NaN if unstable.
              - unstable_scale (bool): True if the scale is zero/non-finite or
                otherwise unusable; False otherwise.
    """
    if y_train is None and fallback_scale is None:
        raise ValueError(
            "Invalid arguments: either provide `y_train` to compute the scale "
            "or set `fallback_scale` (float). Both cannot be None."
        )
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float) if y_train is not None else None

    # Fail fast on non-finite or empty numerator data
    if _bad_numerator_inputs(y_true, y_pred):
        return (np.nan, np.nan, np.nan, False) if return_components else (np.nan, False)

    error = y_true - y_pred
    mse = float(np.mean(error**2))
    if not np.isfinite(mse):
        # overflow/underflow protection on numerator
        return (np.nan, np.nan, np.nan, False) if return_components else (np.nan, False)

    # ensure base data to create denominator is finite
    if y_train is not None:
        scale, is_bad_scale = difference_scale(y_train, m)
    else:
        if fallback_scale is None:
            raise ValueError(
                "Invalid arguments: either provide `y_train` or set `fallback_scale`."
            )
        scale = float(fallback_scale)
        is_bad_scale = _scale_is_invalid(scale)

    if is_bad_scale:
        return (np.nan, mse, np.nan, True) if return_components else (np.nan, True)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        value = _sanitize_value(np.sqrt(mse) / scale)

    return (value, mse, scale, False) if return_components else (value, False)


def difference_scaled_bias(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    *,
    m: int = 1,
    y_train: Iterable[float] | None = None,
    fallback_scale: float | None = None,
    scale_stat: Literal["rms", "meanabs"] = "rms",
    return_components: bool = False,
) -> Tuple[float, bool] | Tuple[float, float, float, bool]:
    """Return the difference-scaled bias (DSB): the mean signed error scaled by an
    m-lag difference-based statistic (RMS or mean-absolute) computed from `y_train`.
    Can use provided scale from `fallback_scale`.

    Args:
        y_true (Iterable[float]): Actual values for the evaluation window,
            aligned to `y_pred`.
        y_pred (Iterable[float]): Forecast values aligned to `y_true`.
        m (int): Number of indices between datapoints used to construct
            denominator scale (m-lag differences). Defaults to 1.
        y_train (Iterable[float]|None): The 1D in-sample series used to construct
            scaling term via m-lag differences. Defaults to None.
        fallback_scale (float | None): Scale to use if y_train not provided.
        scale_stat (Literal["rms", "meanabs"], optional): Statistic used to turn
            m-lag differences into scale. Either "rms" (root-mean-square) or
            "meanabs" (mean absolute). Defaults to "rms".
        return_components (bool): If True, also return the numerator mean error
            and the denominator scale, along with the instability flag.
            Defaults to False.

    Raises:
        ValueError: Either `y_train` or `fallback_scale` must be provided. Both
            cannot be None. If both provided, `y_train` takes precedence.

    Returns:
        Tuple[float, bool]: If `return_components` is False, returns
            (dsb_value, unstable_scale) where:
              - dsb_value (float): The difference-scaled bias, or NaN if inputs
                are invalid or the scale is unstable/degenerate.
              - unstable_scale (bool): True if the denominator scale was zero,
                non-finite, or otherwise unusable; False otherwise.

    Tuple[float, float, float, bool]: If `return_components` is True, returns
        (dsb_value, mean_error, scale, unstable_scale) where:
          - dsb_value (float): The difference-scaled bias, or NaN on
            invalid/unstable inputs.
          - mean_error (float): Mean signed error of `y_true - y_pred`, or NaN
            if the numerator is invalid.
          - scale (float): The m-lag difference scale from `y_train` using
            `scale_stat`, or NaN if unstable.
          - unstable_scale (bool): True if the scale is zero/non-finite or
            otherwise unusable; False otherwise.
    """
    if y_train is None and fallback_scale is None:
        raise ValueError(
            "Invalid arguments: either provide `y_train` to compute the scale "
            "or set `fallback_scale` (float). Both cannot be None."
        )
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float) if y_train is not None else None

    # Fail fast on non-finite or empty numerator data
    if _bad_numerator_inputs(y_true, y_pred):
        return (np.nan, np.nan, np.nan, False) if return_components else (np.nan, False)

    mean_error = float(np.mean(y_pred - y_true))
    if not np.isfinite(mean_error):
        return (np.nan, np.nan, np.nan, False) if return_components else (np.nan, False)

    # ensure base data to create denominator is finite
    if y_train is not None:
        scale_val, is_bad_scale = difference_scale(y_train, m, scale_stat)
    else:
        if fallback_scale is None:
            raise ValueError(
                "Invalid arguments: either provide `y_train` or set `fallback_scale`."
            )
        scale_val = float(fallback_scale)
        is_bad_scale = _scale_is_invalid(scale_val)

    if is_bad_scale:
        return (
            (np.nan, mean_error, np.nan, True) if return_components else (np.nan, True)
        )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        value = _sanitize_value(mean_error / scale_val)

    return (
        (value, mean_error, scale_val, False) if return_components else (value, False)
    )
