# Forecast Backtest and Evaluation System — V1 Product Specification

## 1. Overview

This document specifies the V1 design of a forecast backtesting and evaluation system built for enterprise-scale time series forecasting workflows. The system enables data scientists to run cross-validated backtests with any forecast model, evaluate results using standard and custom metrics, apply preprocessing transforms, and produce structured artifacts suitable for experiment tracking via Vertex AI Experiments.

The system is model-agnostic, configuration-driven, and designed for use within Vertex AI pipelines, Vertex AI notebook instances, and local development environments.

---

## 2. Design Principles

- **Model-agnostic.** The system is decoupled from any specific forecasting library. Models are user-provided callables.
- **Configuration-driven.** A single YAML config file specifies the forecast horizon, cross-validation strategy, transforms, model hyperparameters, metrics, parallelization, and artifact storage.
- **One model per run.** Each config file defines a single model configuration. Comparisons across models happen via experiment tracking across runs.
- **User as adult.** The system does not impose minimum training sizes or other guardrails that restrict valid experimentation. Models that cannot handle the data will raise their own errors.
- **Resilient execution.** Failed series and folds are skipped and logged, not halted on. The system produces results for everything that succeeds.
- **Structured output.** Results are returned as a typed Python dataclass. The user controls how artifacts are logged to Vertex AI Experiments.

---

## 3. Data Format

### 3.1 Input Convention

The system accepts pandas DataFrames in Nixtla long format:

| Column | Description |
|---|---|
| `ds` | Timestamp column |
| `unique_id` | Unique identifier for each time series |
| `y` | Target variable |

Exogenous variables (`X_exogenous`) are provided as a separate DataFrame with `ds`, `unique_id`, and one or more exogenous feature columns. The exogenous DataFrame may be `None` if no exogenous variables are used.

### 3.2 Data Loading

The user loads data externally and passes DataFrames into the system programmatically. The system does not perform file I/O for data loading in V1.

```python
import pandas as pd
from backtest_system import run_backtest

target_df = pd.read_parquet("gs://my-bucket/data/sales.parquet")
exog_df = pd.read_parquet("gs://my-bucket/data/features.parquet")

results = run_backtest(
    config_path="config.yaml",
    target_df=target_df,
    exogenous_df=exog_df,
)
```

### 3.3 Panel and Single Series Support

The system supports both single time series and panel data (multiple `unique_id` values). The cross-validation, transform, and evaluation logic applies identically in both cases; a single series is treated as panel data with one `unique_id`.

### 3.4 Temporal Granularity

The system supports daily, weekly, and monthly time series. The granularity is not explicitly configured; it is inferred from the `ds` column and the user's cross-validation and horizon settings.

---

## 4. Cross-Validation

### 4.1 Windowing Strategy

V1 uses an **expanding window** approach. For each fold, the training set is anchored at the start of the data and extends to the fold's forecast origin. The validation set begins immediately after the forecast origin and extends for the fixed forecast horizon.

The system is designed to accommodate sliding window in a future version.

### 4.2 Fold Specification Modes

The user specifies folds in one of two mutually exclusive modes:

**Explicit forecast origins.** The user provides a list of forecast origin dates. Each date defines where the training set ends and the forecast begins.

```yaml
cross_validation:
  mode: explicit
  horizon: 12
  forecast_origins:
    - "2023-01-01"
    - "2023-04-01"
    - "2023-07-01"
    - "2023-10-01"
```

**Parametric.** The user specifies the number of folds and the step size (increment to slide the forecast origin). The system computes forecast origin dates by working backward from the end of the data.

```yaml
cross_validation:
  mode: parametric
  horizon: 12
  n_folds: 4
  step_size: 3  # periods to shift origin between folds
```

### 4.3 Fixed Forecast Horizon

The forecast horizon is constant across all folds within a run. It is specified in the `cross_validation` section of the config.

### 4.4 Internal Representation

Both modes produce the same internal representation: an ordered list of forecast origin dates. All downstream logic (splitting, fitting, evaluating) operates on this list without knowledge of which mode produced it.

---

## 5. Transforms

### 5.1 Overview

Transforms are a small, ordered chain of operations applied to the data before modeling and (for invertible transforms on `y`) inverted after forecasting. Transforms are applied sequentially in the order specified in the config.

### 5.2 Transform Object Interface

Each transform is a Python object implementing the following methods:

```python
class SomeTransform:
    def fit_transform(self, series, **params):
        """Fit parameters from data and return transformed series.
        For fixed-parameter transforms, reads params without fitting."""
        ...

    def transform(self, series):
        """Apply already-fitted parameters to new data (e.g., validation set)."""
        ...

    def inverse_transform(self, series):
        """Reverse the transform. Used on forecasted y."""
        ...

    def get_fitted_params(self):
        """Return dict of fitted or fixed parameters for artifact logging."""
        ...
```

The `fit_transform` method is called on each fold's training data. The `transform` method is called on the validation set using the parameters fitted from the training set. The `inverse_transform` method is applied to the model's forecasted `y` to return predictions to the original scale.

Inverse transforms are only applied to `y`. Exogenous variables are model inputs and do not require inverse transformation after forecasting.

### 5.3 Scope

Each transform declares a scope:

- **`per_series`**: A separate transform instance is created (or refit) for each `unique_id`. Fitted parameters may differ across series. Examples: Box-Cox, Z-score standardization.
- **`global`**: A single transform instance is shared across all series. Examples: trading day normalization using a shared calendar.

### 5.4 Targets

Each transform declares which columns it operates on via the `targets` field. The reserved keyword `y` refers to the target variable. Any other name refers to a specific column in `X_exogenous`. Different exogenous variables can receive different transforms.

```yaml
transforms:
  - name: trading_day_normalization
    class: mypackage.transforms.TradingDayNormalization
    scope: global
    targets: [y, revenue]
    invertible: true
    params:
      calendar: NYSE

  - name: log_transform
    class: mypackage.transforms.LogTransform
    scope: per_series
    targets: [price]
    invertible: true
```

### 5.5 Fixed vs. Fitted Parameters

Transforms support both modes:

- **Fitted parameters.** The `fit_transform` method learns parameters from the training data (e.g., Box-Cox lambda, Z-score mean and std). These parameters are refit on each fold's training data to avoid data leakage.
- **Fixed parameters.** The user supplies a parameter value in the config (e.g., `power: 0.3333` for a cube root). The `fit_transform` method uses the supplied value without fitting.

The transform object decides internally whether to fit or use a fixed value based on the `**params` it receives. The system treats all transforms identically, always calling `fit_transform` on each fold's training data.

```yaml
# Fitted from data
- name: box_cox
  class: mypackage.transforms.BoxCoxTransform
  scope: per_series
  targets: [y]
  params:
    lambda_range: [0, 2]

# Fixed cube root
- name: cube_root
  class: mypackage.transforms.PowerTransform
  scope: per_series
  targets: [y]
  params:
    power: 0.3333
    fixed: true

# Per-series overrides: fixed for some, fitted for others
- name: box_cox
  class: mypackage.transforms.BoxCoxTransform
  scope: per_series
  targets: [y]
  params:
    lambda_range: [0, 2]
    series_overrides:
      SKU_001: {fixed_lambda: 0.3333}
      SKU_002: {fixed_lambda: 0.5}
```

### 5.6 Model-Native Transforms

Some forecast models handle certain transforms internally (e.g., AutoTBATS accepts `box_cox: true` with `bc_lower_bound` and `bc_upper_bound`; AutoARIMA accepts a `lambda` parameter). When `model_native: true` is set on a transform, the system skips external application of that transform and instead merges the transform's parameters into the model callable's `**kwargs`.

The `model_params_mapping` field translates between transform parameter names and the model's expected parameter names.

```yaml
- name: box_cox
  model_native: true
  model_params_mapping:
    box_cox: true
    bc_lower_bound: 0
    bc_upper_bound: 1
```

### 5.7 Built-in Transforms

The system ships with built-in transform classes for common operations: Box-Cox, power transforms (including cube root), Z-score standardization, add-constant, and trading day normalization. Users can provide custom transforms following the same four-method interface.

---

## 6. Model Interface

### 6.1 Simple Callable Convention

Models are user-provided callables with the following signature:

```python
def my_model(train_df: pd.DataFrame, horizon: int, **kwargs) -> pd.DataFrame:
    """
    Args:
        train_df: Training data in long format (ds, unique_id, y, and
                  optionally exogenous columns).
        horizon: Number of periods to forecast.
        **kwargs: Hyperparameters and other arguments from the config.

    Returns:
        DataFrame with forecasted values containing ds, unique_id,
        and y_pred columns.
    """
    ...
```

The system imports the callable dynamically from the import path specified in the config and passes hyperparameters as `**kwargs`.

### 6.2 Residuals

The system supports three cases for in-sample residuals:

**Case 1: No residuals available.** Some models (e.g., pretrained models like Amazon Chronos) do not produce in-sample fitted values. The model callable returns only a forecast DataFrame. The `residuals` field in `BacktestResults` is `None` for that series/fold.

**Case 2: Model exposes fitted values.** Some models (e.g., Nixtla statsforecast, mlforecast) allow extraction of in-sample fitted values as a natural byproduct of the fitting process. In this case, the model callable optionally returns a tuple of `(forecast_df, residuals_df)` instead of just `forecast_df`.

The system detects the return type. If the callable returns a DataFrame, there are no residuals. If it returns a tuple, the system unpacks the forecast and residuals.

```python
# Case 1 — no residuals
def chronos_model(train_df, horizon, **kwargs):
    ...
    return forecast_df

# Case 2 — model provides fitted values
def arima_model(train_df, horizon, **kwargs):
    ...
    residuals_df = ...  # extracted from the fitted model
    return forecast_df, residuals_df
```

**Case 3: System-computed residuals (out of scope for V1).** Computing in-sample fitted values by re-running the model on the training data is deferred to a future version due to the complexity of varying model semantics across different forecasting libraries.

### 6.3 Model Configuration

```yaml
model:
  callable: mypackage.models.my_arima_model
  hyperparameters:
    order: [1, 1, 1]
    seasonal_order: [1, 1, 1, 12]
  model_n_jobs: 4
```

The `model_n_jobs` parameter is passed through to the model callable for models that support internal parallelization (e.g., `n_jobs` in statsforecast). Models that do not support it (e.g., Chronos) simply ignore it.

### 6.4 Model Serialization

Model serialization is opt-in via the config. When enabled, the system attempts to serialize the fitted model on a best-effort basis after each fold. If serialization fails, the system logs a warning and continues.

The user specifies the serialization method, which can be a built-in (`pickle`, `cloudpickle`, `joblib`) or a user-provided callable (e.g., wrapping `statsforecast.save()`).

```yaml
model:
  callable: mypackage.models.my_model
  hyperparameters: {}
  serialization:
    enabled: true
    method: joblib  # or pickle, cloudpickle, or a callable import path
```

---

## 7. Metrics

### 7.1 Two Callable Types

Metrics are user-provided callables in one of two types:

**Simple metrics.** Require only actuals and predictions.

```python
def mae(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
    ...
```

**Context-aware metrics.** Additionally receive the training data for computing scale-dependent measures.

```python
def rmsse(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, **kwargs) -> float:
    ...
```

The system determines which arguments to pass based on the `type` field in the config. Extra parameters (e.g., `m` for lag differences, `fallback_scale`) are passed via `**kwargs` from the config.

### 7.2 Tuple Returns for Diagnostics

Metrics may return either a `float` or a `Tuple[float, bool]` where the boolean is an instability flag (e.g., degenerate scale in RMSSE). The system detects the return type and logs instability flags as metadata alongside metric values.

### 7.3 Aggregation Modes

Each metric specifies an aggregation mode in the config:

- **`per_fold_mean`**: The metric is computed per fold, then averaged across folds.
- **`pooled`**: Actuals and predictions (and training data for context-aware metrics) are concatenated across folds before computing the metric once.

### 7.4 Per-Series and Global Metrics

The system supports both per-series metrics (e.g., RMSSE computed for each `unique_id`) and global metrics (e.g., WRMSSE computed across all series with aggregation weights). Both can coexist in a single run. The metric callable itself determines whether it operates per-series or globally; the system routes data accordingly based on the `scope` field.

### 7.5 Metric Grouping

Metrics can be computed at intermediate grouping levels (e.g., by product class, country) in addition to per-series and global. Grouping columns are specified in the config.

### 7.6 Multiple Metrics Per Run

A single run can specify multiple metrics. All metrics are computed on the same backtest results.

### 7.7 Metrics Configuration

```yaml
metrics:
  - name: rmsse
    callable: mypackage.metrics.rmsse
    type: context_aware
    scope: per_series
    aggregation: per_fold_mean
    params:
      m: 1
      return_components: false

  - name: scaled_bias
    callable: mypackage.metrics.difference_scaled_bias
    type: context_aware
    scope: per_series
    aggregation: per_fold_mean
    params:
      m: 1
      scale_stat: rms

  - name: wrmsse
    callable: mypackage.metrics.wrmsse
    type: context_aware
    scope: global
    aggregation: pooled
    params:
      weights_path: gs://my-bucket/data/weights.parquet

  - name: mae
    callable: mypackage.metrics.mae
    type: simple
    scope: per_series
    aggregation: per_fold_mean

  grouping_columns: [product_class, country]
```

### 7.8 Built-in Metrics

The system ships with built-in implementations of common metrics: MAE, RMSE, MAPE, RMSSE, and difference-scaled bias. Users can provide custom metric callables following the same conventions.

---

## 8. Parallelization

### 8.1 Separation of Concerns

Parallelization is split between the model and the evaluation system:

- **Model fitting and prediction**: Parallelization is the model's responsibility. The system passes `model_n_jobs` through to the model callable via `**kwargs`. Models that support internal parallelization (e.g., statsforecast's `n_jobs`) use it; models that do not (e.g., Chronos) ignore it.
- **Forecast evaluation**: The system owns parallelization using Python's `multiprocessing` module.

### 8.2 Evaluation Parallelization Strategies

The user specifies the evaluation parallelization strategy in the config:

- **`across_series`**: Each `unique_id` is evaluated independently in parallel. Primary win for panel data.
- **`across_folds`**: Each cross-validation fold is evaluated in parallel for a given series.
- **`nested`**: Parallelization across both series and folds.

### 8.3 Configuration

```yaml
parallelization:
  parallel_eval_strategy: across_series
  eval_n_workers: 4       # -1 for all available cores
  model_n_jobs: 4          # passed through to the model callable
```

### 8.4 Environment Compatibility

The parallelization approach is compatible with Vertex AI pipeline worker nodes, Vertex AI notebook instances, and local laptops with multiple CPU cores.

---

## 9. Configuration File

### 9.1 Format

YAML. One model per config file. One run per config file.

### 9.2 Full Config Structure

```yaml
# --- Data ---
data:
  target_col: y
  date_col: ds
  id_col: unique_id
  exogenous_columns: [price, promotions, temperature]

# --- Cross-Validation ---
cross_validation:
  mode: parametric          # or "explicit"
  horizon: 12
  n_folds: 4               # parametric mode only
  step_size: 3             # parametric mode only
  # forecast_origins:       # explicit mode only
  #   - "2023-01-01"
  #   - "2023-04-01"

# --- Transforms ---
transforms:
  - name: trading_day_normalization
    class: mypackage.transforms.TradingDayNormalization
    scope: global
    targets: [y]
    invertible: true
    params:
      calendar: NYSE

  - name: box_cox
    class: mypackage.transforms.BoxCoxTransform
    scope: per_series
    targets: [y]
    invertible: true
    model_native: false
    params:
      lambda_range: [0, 2]

# --- Model ---
model:
  callable: mypackage.models.auto_arima_forecast
  hyperparameters:
    order: [1, 1, 1]
    seasonal_order: [1, 1, 1, 12]
  model_n_jobs: 4
  serialization:
    enabled: false
    method: joblib

# --- Metrics ---
metrics:
  definitions:
    - name: rmsse
      callable: mypackage.metrics.rmsse
      type: context_aware
      scope: per_series
      aggregation: per_fold_mean
      params:
        m: 1
    - name: wrmsse
      callable: mypackage.metrics.wrmsse
      type: context_aware
      scope: global
      aggregation: pooled
      params: {}
    - name: mae
      callable: mypackage.metrics.mae
      type: simple
      scope: per_series
      aggregation: per_fold_mean
  grouping_columns: [product_class, country]

# --- Parallelization ---
parallelization:
  parallel_eval_strategy: across_series
  eval_n_workers: 4
  model_n_jobs: 4

# --- Artifact Storage ---
artifact_storage:
  output_path: gs://my-bucket/experiments/run_001/
  uv_lock_path: ./uv.lock

# --- Experiment Tracking ---
experiment_tracking:
  experiment_name: demand_forecasting_v1
  run_name_prefix: auto_arima
```

---

## 10. Output Structure

### 10.1 BacktestResults Dataclass

The system returns a frozen dataclass containing all artifacts from the run.

```python
from dataclasses import dataclass, field
import pandas as pd

@dataclass(frozen=True)
class BacktestResults:
    # --- Always present ---
    forecasts_per_fold: dict[str, pd.DataFrame]
    metrics: pd.DataFrame
    fold_origins: list[pd.Timestamp]
    horizon: int
    config: dict
    train_val_splits_per_fold: dict[str, dict[str, pd.DataFrame]]
    git_hash: str
    uv_lock: str
    run_summary: dict

    # --- Present depending on model/config ---
    residuals: dict[str, pd.DataFrame] | None = None
    transform_params: dict[str, dict[str, dict]] | None = None
    metric_instability_flags: pd.DataFrame | None = None
    metric_groups: dict[str, pd.DataFrame] | None = None
    fitted_models: dict[str, bytes] | None = None

    # --- Escape hatch ---
    extra: dict | None = None
```

### 10.2 Metrics DataFrame Schema

The `metrics` DataFrame uses a long format that accommodates per-series, grouped, and global metrics in a single table:

| Column | Description |
|---|---|
| `metric_name` | Name of the metric (e.g., `rmsse`, `wrmsse`, `mae`) |
| `unique_id` | Series identifier, or `None` for global/grouped metrics |
| `fold` | Fold identifier (e.g., `fold_1`) or `pooled` |
| `aggregation` | One of `per_series`, `group`, `global` |
| `value` | Computed metric value |
| *(grouping columns)* | Optional columns (e.g., `product_class`, `country`) for grouped metrics |

### 10.3 Run Summary

The `run_summary` dict provides a structured overview of run health:

```python
{
    "total_series_attempted": 500,
    "successful_series": 487,
    "failed_series": 13,
    "warned_series": 52,
    "failure_rate": 0.026,
    "warning_rate": 0.104,
    "total_folds": 5,
    "failed_folds": 2,
    "failures": [
        {"unique_id": "SKU_042", "fold": "fold_3", "error": "ValueError: ..."},
        ...
    ],
    "warnings": [
        {"unique_id": "SKU_007", "fold": "fold_1", "warning": "ConvergenceWarning: ..."},
        ...
    ],
}
```

`warned_series` counts unique series with at least one warning, not total warning instances. The detailed `warnings` list captures every instance. Warnings are captured from model fitting using Python's `warnings.catch_warnings()` context manager.

---

## 11. Experiment Tracking

### 11.1 Approach

The system prepares structured artifacts in the `BacktestResults` dataclass. The user controls logging to Vertex AI Experiments directly using the Vertex AI SDK. The system does not abstract or wrap the Vertex AI Experiments API in V1.

### 11.2 Helper Utilities

The system provides helper utilities that make it convenient to extract artifacts from `BacktestResults` in formats suitable for Vertex AI Experiments logging (e.g., flattened metric dicts, serialized config, DataFrames as parquet bytes).

### 11.3 Artifact Manifest

The following artifacts are available for logging per run:

| Artifact | Source Field | Description |
|---|---|---|
| YAML config | `config` | Full configuration for reproducibility |
| Train/val splits | `train_val_splits_per_fold` | Per-fold training and validation DataFrames |
| Forecasts | `forecasts_per_fold` | Per-fold y_pred aligned to y_true |
| Residuals | `residuals` | In-sample residuals (when available) |
| Metric values | `metrics` | Per-series, grouped, and global metrics |
| Instability flags | `metric_instability_flags` | Flags for degenerate metric computations |
| Grouped metrics | `metric_groups` | Metrics by grouping columns |
| Forecast origins | `fold_origins` | Dates defining each CV fold |
| Forecast horizon | `horizon` | Fixed horizon for the run |
| Transform params | `transform_params` | Fitted/fixed parameters per fold per series |
| Fitted models | `fitted_models` | Serialized model per fold (when enabled) |
| uv.lock | `uv_lock` | Dependency lockfile contents |
| Git hash | `git_hash` | Code version identifier |
| Run summary | `run_summary` | Success/failure/warning counts and details |

---

## 12. Error Handling and Validation

### 12.1 Validation Implementation

- **Config validation** uses Pydantic models to parse and validate the YAML config, providing clear error messages for missing fields, wrong types, and invalid values.
- **Internal data structures** (e.g., `BacktestResults`) use standard dataclasses since they are constructed by the system with already-validated data.

### 12.2 Upfront Validation

The system validates as much as possible before beginning computation:

- YAML config parses without error.
- Required config sections and fields are present.
- Callable import paths for model, metrics, and transforms resolve successfully.
- Required columns (`ds`, `unique_id`, `y`) exist in the target DataFrame.
- Exogenous columns specified in the config exist in the exogenous DataFrame.
- Forecast origins (explicit mode) fall within the data's date range.
- Horizon is a positive integer.
- Parametric CV settings produce valid fold origins.
- No contradictory settings (e.g., `model_native: true` with `invertible: true` on the same transform).
- Serialization method is valid if serialization is enabled.

### 12.3 Series-Level Failures

If a model callable or transform raises an exception for a specific `unique_id`, the system skips that series, logs the error (including the `unique_id`, fold, and exception details), and continues processing remaining series. Warnings emitted during model fitting are captured and logged but do not cause the series to be skipped.

### 12.4 Fold-Level Failures

If an entire fold fails (e.g., a global transform raises an exception), the system logs the error (including the fold identifier and exception details) and continues with remaining folds.

### 12.5 Run Summary

All errors and warnings are aggregated into the `run_summary` field of `BacktestResults`, providing counts, rates, and detailed failure/warning records for post-run triage.

---

## 13. Testing

### 13.1 Framework and Style

All tests use pytest in functional format. No class-based tests. Fixtures are defined in `conftest.py`. No setup/teardown methods.

### 13.2 Scope

V1 includes unit tests for each component covering critical paths only. Tests are focused and limited in number (under 10 per function), tailored to the complexity of each component rather than targeting a coverage percentage.

### 13.3 Components Under Test

- **Config parsing and validation.** Valid configs parse correctly; invalid configs raise clear errors for each validation rule.
- **Cross-validation fold generation.** Both explicit and parametric modes produce correct forecast origin lists. Expanding window splits are correct.
- **Transforms.** `fit_transform`, `transform`, `inverse_transform`, and `get_fitted_params` produce correct results for built-in transforms. Fixed vs. fitted parameter paths work correctly. Per-series and global scope behave correctly.
- **Model callable loading.** Dynamic import from string path resolves correctly. Hyperparameters are passed through.
- **Metrics computation.** Simple and context-aware metrics produce correct values. Tuple returns with instability flags are handled. Pooled and per-fold aggregation produce correct results. Per-series and global scope work correctly.
- **Evaluation parallelization.** Results are equivalent whether run with 1 worker or multiple workers.
- **Error handling.** Failed series are skipped and logged. Failed folds are skipped and logged. Run summary accurately reflects successes, failures, and warnings.
- **BacktestResults construction.** Required fields must be present. Optional fields default to None. The `extra` dict works as a catch-all.

### 13.4 Fixtures

`conftest.py` provides shared fixtures including: synthetic single-series and panel DataFrames, sample YAML configs, simple model callables, simple and context-aware metric callables, and transform instances.

---

## 14. Future Considerations (Out of Scope for V1)

The following capabilities are acknowledged but deferred beyond V1:

- **Sliding window cross-validation.** The expanding window implementation is designed to accommodate this without architectural changes.
- **Integration tests.** End-to-end tests running `run_backtest` with synthetic data and verifying the full pipeline.
- **System-managed data loading.** Reading input data from file paths, GCS URIs, or bucket+prefix configurations specified in the YAML config.
- **Experiment tracking abstraction.** A generic logging interface with pluggable backends for MLflow and Vertex AI Experiments.
- **Model adapter pattern.** A formal adapter interface with `fit`, `predict`, and `get_residuals` methods, replacing or extending the simple callable convention.
- **BigQuery data source support.**
- **System-computed residuals.** Computing in-sample fitted values by re-running the model on the training data, for models that do not natively expose fitted values through their API.
