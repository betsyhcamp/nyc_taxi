from __future__ import annotations

import numpy as np
import pytest

from fcstnyctaxi.lib.metrics import (
    _SMALL_NUM_BOUND,
    _check_1d_same_shape,
    _sanitize_value,
    signed_bias_per_series,
    signed_bias_pooled,
    wrmae_per_series,
    wrmae_pooled,
)

# ================================================
# Fixtures
# ================================================


@pytest.fixture
def arr():
    """Convert list to 1-D float64 array."""
    return lambda vals: np.array(vals, dtype=float)


@pytest.fixture
def eps_tiny():
    """Value just below _SMALL_NUM_BOUND — exercises near-zero denominator guard."""
    return _SMALL_NUM_BOUND * 0.5


# ================================================
# _sanitize_value
# ================================================


def test_sanitize_value_positive_finite_passthrough():
    """Returns float unchanged for positive finite input — no warning."""
    np.testing.assert_allclose(_sanitize_value(3.14), 3.14, rtol=1e-7)


def test_sanitize_value_negative_finite_passthrough():
    """Negative finite passthrough; signed bias can legitimately return negative."""
    np.testing.assert_allclose(_sanitize_value(-3.14), -3.14, rtol=1e-7)


def test_sanitize_value_zero_passthrough():
    """Returns 0.0 unchanged — perfect forecast scenario."""
    np.testing.assert_allclose(_sanitize_value(0.0), 0.0, rtol=1e-7, atol=1e-10)


def test_sanitize_value_pos_inf_warns_and_returns_nan():
    """Positive infinity triggers RuntimeWarning and returns nan."""
    with pytest.warns(RuntimeWarning, match="upstream guards"):
        result = _sanitize_value(float("inf"))
    assert np.isnan(result)


def test_sanitize_value_neg_inf_warns_and_returns_nan():
    """Negative infinity triggers RuntimeWarning and returns nan."""
    with pytest.warns(RuntimeWarning, match="upstream guards"):
        result = _sanitize_value(float("-inf"))
    assert np.isnan(result)


def test_sanitize_value_nan_input_warns_and_returns_nan():
    """nan is non-finite — triggers RuntimeWarning and returns nan."""
    with pytest.warns(RuntimeWarning, match="upstream guards"):
        result = _sanitize_value(float("nan"))
    assert np.isnan(result)


# ================================================
# _check_1d_same_shape
# ================================================


def test_check_1d_same_shape_valid_two_arrays(arr):
    """Two 1-D arrays of equal length -> no exception raised."""
    _check_1d_same_shape(arr([1.0, 2.0]), arr([3.0, 4.0]))


def test_check_1d_same_shape_valid_three_arrays(arr):
    """Three 1-D arrays of equal length -> no exception raised."""
    _check_1d_same_shape(arr([1.0, 2.0]), arr([3.0, 4.0]), arr([5.0, 6.0]))


def test_check_1d_same_shape_2d_input_raises():
    """2-D array input raises ValueError."""
    with pytest.raises(ValueError, match="1-D"):
        _check_1d_same_shape(np.array([[1.0, 2.0]]), np.array([[3.0, 4.0]]))


def test_check_1d_same_shape_mismatched_lengths_raises(arr):
    """Arrays of different lengths raise ValueError."""
    with pytest.raises(ValueError, match="shapes must match"):
        _check_1d_same_shape(arr([1.0, 2.0]), arr([1.0, 2.0, 3.0]))


# ================================================
# wrmae_pooled
# ================================================


def test_wrmae_pooled_known_value_equal_weights(arr):
    """Pooled WRMAE with equal weights and 2:1 error ratio -> 0.5."""
    # c=[1,2,3], b=[2,4,6], w=[1,1,1] -> num=6, den=12 -> 0.5
    result = wrmae_pooled(
        arr([1.0, 2.0, 3.0]), arr([2.0, 4.0, 6.0]), arr([1.0, 1.0, 1.0])
    )
    np.testing.assert_allclose(result, 0.5, rtol=1e-7)


def test_wrmae_pooled_known_value_unequal_weights(arr):
    """Unequal weights shift the pooled ratio away from a per-series average."""
    # c=[1,3], b=[2,2], w=[3,1] -> num=3*1+1*3=6, den=3*2+1*2=8 -> 0.75
    result = wrmae_pooled(arr([1.0, 3.0]), arr([2.0, 2.0]), arr([3.0, 1.0]))
    np.testing.assert_allclose(result, 0.75, rtol=1e-7)


def test_wrmae_pooled_challenger_worse_than_benchmark(arr):
    """Result > 1 when challenger errors exceed benchmark errors."""
    # c=[2,4], b=[1,2], w=[1,1] -> num=6, den=3 -> 2.0
    result = wrmae_pooled(arr([2.0, 4.0]), arr([1.0, 2.0]), arr([1.0, 1.0]))
    np.testing.assert_allclose(result, 2.0, rtol=1e-7)


def test_wrmae_pooled_perfect_challenger_returns_zero(arr):
    """Challenger with all-zero errors -> numerator is 0 -> result is 0.0."""
    result = wrmae_pooled(arr([0.0, 0.0]), arr([2.0, 4.0]), arr([1.0, 1.0]))
    np.testing.assert_allclose(result, 0.0, rtol=1e-7, atol=1e-10)


def test_wrmae_pooled_empty_returns_nan(arr):
    """Empty arrays -> nan."""
    assert np.isnan(wrmae_pooled(arr([]), arr([]), arr([])))


def test_wrmae_pooled_nonfinite_challenger_returns_nan(arr):
    """Non-finite value in challenger_errors -> nan."""
    result = wrmae_pooled(arr([np.inf, 1.0]), arr([1.0, 1.0]), arr([1.0, 1.0]))
    assert np.isnan(result)


def test_wrmae_pooled_nonfinite_benchmark_returns_nan(arr):
    """Non-finite value in benchmark_errors -> nan."""
    result = wrmae_pooled(arr([1.0, 1.0]), arr([np.nan, 1.0]), arr([1.0, 1.0]))
    assert np.isnan(result)


def test_wrmae_pooled_nonfinite_weights_returns_nan(arr):
    """Non-finite value in weights -> nan."""
    result = wrmae_pooled(arr([1.0, 1.0]), arr([1.0, 1.0]), arr([np.inf, 1.0]))
    assert np.isnan(result)


def test_wrmae_pooled_zero_benchmark_returns_nan(arr):
    """All-zero benchmark -> denominator zero -> nan."""
    result = wrmae_pooled(arr([1.0, 2.0]), arr([0.0, 0.0]), arr([1.0, 1.0]))
    assert np.isnan(result)


def test_wrmae_pooled_near_zero_benchmark_returns_nan(arr, eps_tiny):
    """Benchmark below _SMALL_NUM_BOUND -> _scale_is_invalid fires -> nan."""
    result = wrmae_pooled(arr([1.0, 1.0]), arr([eps_tiny, eps_tiny]), arr([1.0, 1.0]))
    assert np.isnan(result)


def test_wrmae_pooled_shape_mismatch_raises(arr):
    """Mismatched array lengths raise ValueError."""
    with pytest.raises(ValueError, match="shapes must match"):
        wrmae_pooled(arr([1.0, 2.0]), arr([1.0, 2.0]), arr([1.0]))


# ================================================
# wrmae_per_series
# ================================================


def test_wrmae_per_series_known_value_equal_weights(arr):
    """Per-series WRMAE: w_norm * ratios; uniform 2:1 ratios equal the pooled result."""
    # c=[1,2,3], b=[2,4,6], w=[1,1,1] -> w_norm=[1/3]*3, ratios=[0.5]*3 -> 0.5
    result = wrmae_per_series(
        arr([1.0, 2.0, 3.0]), arr([2.0, 4.0, 6.0]), arr([1.0, 1.0, 1.0])
    )
    np.testing.assert_allclose(result, 0.5, rtol=1e-7)


def test_wrmae_per_series_diverges_from_pooled_with_heterogeneous_ratios(arr):
    """V2 and V1 diverge when per-series error ratios are heterogeneous."""
    # c=[1,2], b=[1,8], w=[3,1]
    # V2: w_norm=[0.75,0.25], ratios=[1.0,0.25] -> 0.75+0.0625=0.8125
    # V1: num=(3+2)=5, den=(3+8)=11 -> 0.4545 — clearly different
    result = wrmae_per_series(arr([1.0, 2.0]), arr([1.0, 8.0]), arr([3.0, 1.0]))
    np.testing.assert_allclose(result, 0.8125, rtol=1e-7)


def test_wrmae_per_series_empty_returns_nan(arr):
    """Empty arrays -> nan."""
    assert np.isnan(wrmae_per_series(arr([]), arr([]), arr([])))


def test_wrmae_per_series_nonfinite_input_returns_nan(arr):
    """Non-finite value in any input -> nan."""
    result = wrmae_per_series(arr([np.nan, 1.0]), arr([1.0, 2.0]), arr([1.0, 1.0]))
    assert np.isnan(result)


def test_wrmae_per_series_shape_mismatch_raises(arr):
    """Mismatched array lengths raise ValueError."""
    with pytest.raises(ValueError, match="shapes must match"):
        wrmae_per_series(arr([1.0, 2.0]), arr([1.0, 2.0]), arr([1.0]))


def test_wrmae_per_series_zero_benchmark_excluded(arr):
    """Zero-benchmark series excluded by mask; remaining series computed normally."""
    # c=[1.0, 2.0, 3.0], b=[0.0, 4.0, 6.0], w=[10.0, 1.0, 1.0]
    # Series 0 excluded (b[0]=0.0 fails mask). Remaining: c=[2,3], b=[4,6], w=[1,1]
    # total_weight=2.0, w_norm=[0.5,0.5], ratios=[0.5,0.5] -> result=0.5
    result = wrmae_per_series(
        arr([1.0, 2.0, 3.0]), arr([0.0, 4.0, 6.0]), arr([10.0, 1.0, 1.0])
    )
    np.testing.assert_allclose(result, 0.5, rtol=1e-7)


def test_wrmae_per_series_weights_renormalized_after_exclusion(arr):
    """Excluded weight is discarded; remaining weights renormalize over survivors."""
    #   c=[9.0, 1.0, 1.0], b=[1.0, 0.0, 4.0], w=[1.0, 100.0, 1.0]
    #   Series 1 excluded (b[1]=0.0). Remaining: c=[9,1], b=[1,4], w=[1,1]
    #   total_weight=2.0, w_norm=[0.5,0.5], ratios=[9.0,0.25] -> result=4.625
    result = wrmae_per_series(
        arr([9.0, 1.0, 1.0]), arr([1.0, 0.0, 4.0]), arr([1.0, 100.0, 1.0])
    )
    np.testing.assert_allclose(result, 4.625, rtol=1e-7)


def test_wrmae_per_series_all_benchmark_zero_returns_nan(arr):
    """All benchmark values zero → mask excludes every series -> nan."""
    # c=[1.0,2.0], b=[0.0,0.0], w=[1.0,1.0] → all series masked out -> nan
    result = wrmae_per_series(arr([1.0, 2.0]), arr([0.0, 0.0]), arr([1.0, 1.0]))
    assert np.isnan(result)


# ================================================
# signed_bias_pooled
# ================================================


def test_signed_bias_pooled_positive_bias_over_forecast(arr):
    """Positive bias when y_pred > y_true (over-forecast)."""
    # y_true=[10,20], y_pred=[11,21], w=[1,1] -> num=2, den=30 -> 1/15
    result = signed_bias_pooled(arr([10.0, 20.0]), arr([11.0, 21.0]), arr([1.0, 1.0]))
    np.testing.assert_allclose(result, 1.0 / 15.0, rtol=1e-7)


def test_signed_bias_pooled_negative_bias_under_forecast(arr):
    """Negative bias when y_pred < y_true (under-forecast)."""
    # y_true=[10,20], y_pred=[9,19], w=[1,1] -> num=-2, den=30 -> -1/15
    result = signed_bias_pooled(arr([10.0, 20.0]), arr([9.0, 19.0]), arr([1.0, 1.0]))
    np.testing.assert_allclose(result, -1.0 / 15.0, rtol=1e-7)


def test_signed_bias_pooled_perfect_forecast_returns_zero(arr):
    """Perfect forecast -> numerator is zero -> result is 0.0."""
    result = signed_bias_pooled(arr([10.0, 20.0]), arr([10.0, 20.0]), arr([1.0, 1.0]))
    np.testing.assert_allclose(result, 0.0, rtol=1e-7, atol=1e-10)


def test_signed_bias_pooled_negative_actuals(arr):
    """Denominator uses abs(y_true) — negative actuals are handled correctly."""
    # y_true=[-10,-20], y_pred=[-9,-19] -> num=2, den=|-10|+|-20|=30 -> 1/15
    result = signed_bias_pooled(
        arr([-10.0, -20.0]), arr([-9.0, -19.0]), arr([1.0, 1.0])
    )
    np.testing.assert_allclose(result, 1.0 / 15.0, rtol=1e-7)


def test_signed_bias_pooled_all_zero_y_true_returns_nan(arr):
    """All-zero actuals -> denominator zero -> nan."""
    result = signed_bias_pooled(arr([0.0, 0.0]), arr([1.0, 2.0]), arr([1.0, 1.0]))
    assert np.isnan(result)


def test_signed_bias_pooled_empty_returns_nan(arr):
    """Empty arrays -> nan."""
    assert np.isnan(signed_bias_pooled(arr([]), arr([]), arr([])))


def test_signed_bias_pooled_nonfinite_returns_nan(arr):
    """Non-finite value in y_true -> nan."""
    result = signed_bias_pooled(arr([np.nan, 10.0]), arr([1.0, 11.0]), arr([1.0, 1.0]))
    assert np.isnan(result)


def test_signed_bias_pooled_shape_mismatch_raises(arr):
    """Mismatched array lengths raise ValueError."""
    with pytest.raises(ValueError, match="shapes must match"):
        signed_bias_pooled(arr([1.0, 2.0]), arr([1.0, 2.0]), arr([1.0]))


# ================================================
# signed_bias_per_series
# ================================================


def test_signed_bias_per_series_known_value(arr):
    """Per-series bias: w_norm * (y_pred-y_true)/|y_true| gives weighted mean."""
    # y_true=[10,20], y_pred=[11,21], w=[1,1]
    # w_norm=[0.5,0.5], per_series=[1/10,1/20]=[0.1,0.05] -> 0.5*0.1+0.5*0.05=0.075
    result = signed_bias_per_series(
        arr([10.0, 20.0]), arr([11.0, 21.0]), arr([1.0, 1.0])
    )
    np.testing.assert_allclose(result, 0.075, rtol=1e-7)


def test_signed_bias_per_series_negative_bias_under_forecast(arr):
    """Negative per-series bias when y_pred < y_true."""
    # y_true=[10,10], y_pred=[9,9], w=[1,1]
    # per_series=[-0.1,-0.1], w_norm=[0.5,0.5] -> -0.1
    result = signed_bias_per_series(arr([10.0, 10.0]), arr([9.0, 9.0]), arr([1.0, 1.0]))
    np.testing.assert_allclose(result, -0.1, rtol=1e-7)


def test_signed_bias_per_series_zero_actual_excluded(arr):
    """Series with zero actual excluded by mask; computed over remaining series only."""
    # y_true=[0,20], y_pred=[1,21], w=[5,1]
    # Series 0 excluded (|0| ≤ _SMALL_NUM_BOUND). Remaining: y_true=[20], w=[1]
    # w_norm=[1.0], per_series=[1/20]=0.05 -> result=0.05
    result = signed_bias_per_series(arr([0.0, 20.0]), arr([1.0, 21.0]), arr([5.0, 1.0]))
    np.testing.assert_allclose(result, 0.05, rtol=1e-7)


def test_signed_bias_per_series_all_zero_y_true_returns_nan(arr):
    """All actuals zero -> all series excluded by mask -> nan."""
    result = signed_bias_per_series(arr([0.0, 0.0]), arr([1.0, 2.0]), arr([1.0, 1.0]))
    assert np.isnan(result)


def test_signed_bias_per_series_empty_returns_nan(arr):
    """Empty arrays -> nan."""
    assert np.isnan(signed_bias_per_series(arr([]), arr([]), arr([])))


def test_signed_bias_per_series_nonfinite_returns_nan(arr):
    """Non-finite value in y_pred -> nan."""
    result = signed_bias_per_series(
        arr([10.0, 20.0]), arr([np.inf, 21.0]), arr([1.0, 1.0])
    )
    assert np.isnan(result)


def test_signed_bias_per_series_shape_mismatch_raises(arr):
    """Mismatched array lengths raise ValueError."""
    with pytest.raises(ValueError, match="shapes must match"):
        signed_bias_per_series(arr([1.0, 2.0]), arr([1.0, 2.0]), arr([1.0]))
