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

The system accepts pandas DataFrames in Nixtla long format. Polars DataFrames are also accepted at the `run_backtest` entry point and are converted to pandas internally. All internal processing uses pandas.

| Column | Description |
|---|---|
| `ds` | Timestamp column |
| `unique_id` | Unique identifier for each time series |
| `y` | Target variable |

Exogenous variables (`X_exogenous`) are provided as additional columns in the same DataFrame that contains `ds`, `unique_id`, and `y`. The config's `exogenous_columns` field specifies which columns are exogenous features. If no exogenous variables are used, `exogenous_columns` is `None` or an empty list. The user is responsible for joining exogenous data into the target DataFrame before passing it to the system.

### 3.2 Data Loading

The user loads data externally and passes DataFrames into the system programmatically. The system does not perform file I/O for data loading in V1.

The `run_backtest` function accepts the configuration as either a YAML file path or a Python dictionary. The two are mutually exclusive. **The YAML file is the preferred approach** for reproducibility and readability. The dictionary option is provided for programmatic use cases such as hyperparameter optimization with Optuna, where configs are generated dynamically.

```python
import pandas as pd
from backtest_system import run_backtest

df = pd.read_parquet("gs://my-bucket/data/sales_with_features.parquet")

# Preferred: from YAML file
results = run_backtest(config_path="config.yaml", df=df)

# Alternative: from dict (e.g., generated programmatically by Optuna)
results = run_backtest(config=config_dict, df=df)
```

### 3.3 Panel and Single Series Support

The system supports both single time series and panel data (multiple `unique_id` values). The cross-validation, transform, and evaluation logic applies identically in both cases; a single series is treated as panel data with one `unique_id`.

### 3.4 Temporal Granularity

The system supports hourly, daily, weekly, and monthly time series. The granularity is not explicitly configured; it is inferred from the `ds` column and the user's cross-validation and horizon settings.

---

## 4. Cross-Validation

### 4.1 Windowing Strategy

V1 uses an **expanding window** approach. For each fold, the training set is anchored at the start of the data and extends to the fold's forecast origin. The validation set begins immediately after the forecast origin and extends for the fixed forecast horizon.

Each time series, identified by a `unique_id`, may have a different initial (oldest) data point. However, the most recent data point must be the same across all time series in the panel. The shared most recent date is what the system uses when computing forecast origins in parametric mode.

If a computed forecast origin falls before a particular series' initial data point, that series is excluded from that fold with a logged error for that series and fold combination. This follows the resilient execution pattern described in section 12 — it does not halt the run.

The system is designed to accommodate sliding window in a future version.

### 4.2 Fold Specification Modes

The user specifies folds in one of two mutually exclusive modes:

**Explicit forecast origins.** The user provides a list of forecast origin dates. Each date defines where the training set ends and the forecast begins. These forecast origin dates are the date of the last and most recent data point in the cross-validation training sets.

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

The forecast horizon is constant across all folds within a run, including the test fold. The forecast horizon is the number of time steps to forecast for each cross-validation fold. It is specified in the `cross_validation` section of the config.

### 4.4 Test Fold

The system supports a separate test fold that is structurally isolated from the cross-validation folds. The test fold is processed as a completely separate phase after cross-validation: the model is trained on all data up to the test origin, forecasts are generated for the same horizon, the same transforms and inverse transforms are applied, and the same metrics are computed. Test results are stored in a separate dataclass (`TestResults`) to prevent accidental mixing with CV results during model selection.

The test fold is enabled by default but can be disabled by the user. When disabled, test result fields are `None`.

```yaml
test:
  enabled: true
  test_origin: "2024-01-01"
```

### 4.5 Internal Representation

Both the "explicit forecast origin" and the "parametric" modes produce the same internal representation: an ordered list of forecast origin dates. All downstream logic (splitting, fitting, evaluating) operates on this list without knowledge of which mode produced it. The test fold origin is handled separately from the CV fold origins.

---

## 5. Transforms

### 5.1 Overview

Transforms are a small, ordered chain of operations applied to the data before modeling and (for invertible transforms on `y`) inverted after forecasting and before metric evaluation. Transforms are applied sequentially in the order specified in the config.

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

The `fit_transform` method is called on each fold's training data. The `transform` method is called on the validation set using the parameters fitted from the training set. The `inverse_transform` method is applied to the model's forecasted `y` to return predictions to the original scale before metric evaluation.

Inverse transforms are only applied to `y`. Exogenous variables are model inputs and do not require inverse transformation after forecasting.

### 5.3 Scope

Each transform declares a scope:

- **`per_series`**: A separate transform instance is created (or refit) for each `unique_id`. Fitted parameters may differ across series. Examples: Box-Cox, Z-score standardization.
- **`global`**: A single transform instance is shared across all series. Examples: trading day normalization using a shared calendar, company working day normalization using company holidays, Z-score standardization applied once over the entire panel dataset, or a transformation from nominal dollars to real dollars (V1 assumes all series share the same currency).

The transform scope implementation is designed to accommodate a future `per_group` scope (e.g., applying different transforms to subsets of series grouped by currency or product class) without architectural changes.

### 5.4 Targets

Each transform declares which columns it operates on via the `targets` field. The reserved keyword `y` refers to the target variable. Any other name refers to a specific column in `X_exogenous` or the columns corresponding to the exogenous variables. Different exogenous variables can receive different transforms.

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

- **Fitted parameters.** The `fit_transform` method learns parameters from the training data (e.g., Box-Cox lambda, Z-score mean and std). These parameters are refit on each fold's training data to avoid data leakage. For fitted parameters, the user may optionally provide an allowed range within which the fitted parameter must be found. For some fitted parameters, the user may also specify the objective for the parameter fitting such as for the Box Cox transform the `loglik` (log likelihood) or `guerrero` may be specified by the user.
- **Fixed parameters.** The user supplies a parameter value in the config (e.g., `power: 0.3333` for a cube root). The `fit_transform` method uses the supplied value without fitting.

There is also the ability to have overrides per-series where the user specifies the `unique_id` of the individual time series and the parameter overrides.

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

Some forecast models handle certain transforms internally (e.g., AutoTBATS accepts `box_cox: true` with `bc_lower_bound` and `bc_upper_bound`; AutoARIMA accepts a Box-Cox `lambda` parameter). When `model_native: true` is set on a transform, the system skips external application of that transform and instead merges the transform's parameters into the model callable's `**kwargs`.

The `model_params_mapping` field translates between transform parameter names and the model's expected parameter names. The parameter names should be named identically as the model's expected parameter names.

```yaml
- name: box_cox
  model_native: true
  model_params_mapping:
    box_cox: true
    bc_lower_bound: 0
    bc_upper_bound: 1
```

### 5.7 Built-in Transforms

The system ships with built-in transform classes for common operations. Users can provide custom transforms following the same four-method interface.

**Mathematical transforms (wrap NumPy directly):**

- **NaturalLogTransform**: Applies `np.log` / `np.exp`. Strict — raises an error if the series contains values ≤ 0. The user should chain an `AddConstantTransform` before this transform if the series contains non-positive values.
- **Log1pTransform**: Applies `np.log1p` (i.e., `log(1 + x)`) / `np.expm1`. Numerically stable for small values and naturally handles zeros.
- **PowerTransform**: Applies `series ** power` / `series ** (1/power)`. Supports any user-specified power (e.g., `power: 0.3333` for cube root).
- **AddConstantTransform**: Adds/subtracts a user-specified constant.

**Statistical transforms:**

- **BoxCoxTransform**: Wraps Nixtla's coreforecast (`coreforecast.scalers.boxcox` and `coreforecast.scalers.inv_boxcox`). Supports a `method` parameter (`guerrero` or `loglik`) for lambda selection and a `season_length` parameter required by the Guerrero method. These are simple params passthroughs — the additional parameters in the config are passed to the transform object's `fit_transform` via `**params`, and the system does not require any special handling for them.
- **ZScoreTransform**: Fits mean and standard deviation from training data.

**Domain-specific transforms:**

- **TradingDayNormalization**: Normalizes by trading day count using a shared calendar.

```yaml
# Natural log (strict — errors on non-positive values)
- name: natural_log
  class: mypackage.transforms.NaturalLogTransform
  scope: per_series
  targets: [y]
  invertible: true

# Log1p (handles zeros)
- name: log1p
  class: mypackage.transforms.Log1pTransform
  scope: per_series
  targets: [y]
  invertible: true

# Box-Cox with Guerrero method
- name: box_cox
  class: mypackage.transforms.BoxCoxTransform
  scope: per_series
  targets: [y]
  invertible: true
  params:
    method: guerrero
    season_length: 12
    lower_bound: 0
    upper_bound: 2
```

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
        One of the following:
        - DataFrame: Forecast only.
        - Tuple[DataFrame, DataFrame]: Forecast + in-sample fitted values.
        - Tuple[DataFrame, DataFrame | None, object]: Forecast + optional
          fitted values + fitted model object (for serialization).
    """
    ...
```

The system imports the callable dynamically from the import path specified in the config and passes hyperparameters as `**kwargs`.

The system detects the return type by length:

```python
# Forecast only
return forecast_df

# Forecast + fitted values
return forecast_df, fitted_values_df

# Forecast + fitted values + model object (for serialization)
return forecast_df, fitted_values_df, model_object

# Forecast + model object, no fitted values
return forecast_df, None, model_object
```

The model object (third element) is used by the serialization system when enabled. If the model object is `None` and serialization is enabled, the system logs a warning and skips serialization for that fold.

### 6.2 In-Sample Fitted Values

The system supports three cases for in-sample fitted values:

**Case 1: No fitted values available.** Some models (e.g., pretrained models like Amazon Chronos) do not produce in-sample fitted values. The model callable returns only a forecast DataFrame. The `fitted_values` field in `CVResults` and `TestResults` is `None`.

**Case 2: Model exposes fitted values.** Some models (e.g., Nixtla statsforecast, mlforecast) allow extraction of in-sample fitted values as a natural byproduct of the fitting process. In this case, the model callable optionally returns a tuple of `(forecast_df, fitted_values_df)` instead of just `forecast_df`.

The system detects the return type. If the callable returns a DataFrame, there are no fitted values. If it returns a tuple, the system unpacks the forecast and fitted values.

```python
# Case 1 — no fitted values
def chronos_model(train_df, horizon, **kwargs):
    ...
    return forecast_df

# Case 2 — model provides fitted values
def arima_model(train_df, horizon, **kwargs):
    ...
    fitted_values_df = ...  # extracted from the fitted model
    return forecast_df, fitted_values_df
```

The user can compute residuals from fitted values in whichever convention they prefer (e.g., `y - ŷ` or `ŷ - y`) since the training `y` is available in the train/validation split DataFrames.

**Case 3: System-computed fitted values (out of scope for V1).** Computing in-sample fitted values by re-running the model on the training data is deferred to a future version due to the complexity of varying model semantics across different forecasting libraries.

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

The user specifies the serialization method:

- **`pickle`**, **`cloudpickle`**, **`joblib`**: The system serializes the model object externally using the specified library.
- **`model_method`**: The system calls a method on the fitted model object itself (e.g., `.save()`). This is the recommended approach for Nixtla packages (statsforecast, mlforecast, neuralforecast), which all provide `.save()` and `.load()` methods on fitted models. The `save_method` field specifies the method name and defaults to `save`.
- **User-provided callable**: An import path to a custom serialization function.

All methods require the model callable to return the fitted model object as the third element of the return tuple (see section 6.1). If the model object is not returned and serialization is enabled, the system logs a warning and skips serialization.

```yaml
# Using Nixtla's built-in .save() method
model:
  callable: mypackage.models.my_statsforecast_model
  hyperparameters: {}
  model_n_jobs: 4
  serialization:
    enabled: true
    method: model_method
    save_method: save       # method name on the model object, defaults to "save"

# Using joblib
model:
  callable: mypackage.models.my_model
  hyperparameters: {}
  serialization:
    enabled: true
    method: joblib

# Using a custom serialization callable
model:
  callable: mypackage.models.my_model
  hyperparameters: {}
  serialization:
    enabled: true
    method: mypackage.serializers.custom_save
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

### 7.4 Per-Series, Group, and Global Metrics

The system supports per-series metrics (e.g., RMSSE computed for each `unique_id`), group metrics (e.g., WAPE computed across a product category), and global metrics (e.g., WRMSSE computed across all series with aggregation weights). All three can coexist in a single run. The metric callable itself determines whether it operates per-series, per-group, or globally; the system routes data accordingly based on the `scope` field.

A single metric callable can be defined multiple times with different scopes to compute the same metric at different levels of granularity within a single run:

```yaml
metrics:
  definitions:
    - name: wape_per_series
      callable: mypackage.metrics.wape
      type: simple
      scope: per_series
      aggregation: per_fold_mean

    - name: wape_by_category
      callable: mypackage.metrics.wape
      type: simple
      scope: group
      aggregation: per_fold_mean
      grouping_columns: [product_category]
```

### 7.5 Metric Grouping

Metrics can be computed at intermediate grouping levels (e.g., by product class, country) in addition to per-series and global. Grouping columns can be specified at two levels:

- **Per metric definition**: A `grouping_columns` field on an individual metric definition applies only to that metric. This is required when the metric's `scope` is `group`.
- **Top-level default**: A `grouping_columns` field at the top level of the `metrics` config applies to all metric definitions that have `scope: group` but do not specify their own `grouping_columns`. Per-definition values override the top-level default.

### 7.6 Multiple Metrics Per Run

A single run can specify multiple metrics. All metrics are computed on the same backtest results.

### 7.7 Metrics Configuration

```yaml
metrics:
  definitions:
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

    - name: wape_by_category
      callable: mypackage.metrics.wape
      type: simple
      scope: group
      aggregation: per_fold_mean
      grouping_columns: [product_category]  # per-definition override

    - name: wape_by_country
      callable: mypackage.metrics.wape
      type: simple
      scope: group
      aggregation: per_fold_mean
      # uses top-level default grouping_columns

  grouping_columns: [country]  # default for group-scoped metrics
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

# --- Test Fold ---
test:
  enabled: true
  test_origin: "2024-01-01"

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
    method: model_method
    save_method: save

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
  grouping_columns: [country]  # default for group-scoped metrics

# --- Parallelization ---
parallelization:
  parallel_eval_strategy: across_series
  eval_n_workers: 4

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

### 10.1 Output Dataclasses

The system returns a `BacktestResults` dataclass composed of separate dataclasses for CV results, test results, and shared metadata. This structural separation ensures test results are isolated from CV results during model selection.

```python
from dataclasses import dataclass, field
import pandas as pd


@dataclass(frozen=True)
class CVResults:
    """Cross-validation results. Used during model selection."""
    # --- Always present ---
    forecasts_per_fold: dict[str, pd.DataFrame]
    metrics: pd.DataFrame
    fold_origins: list[pd.Timestamp]
    train_val_splits_per_fold: dict[str, dict[str, pd.DataFrame]]

    # --- Present depending on model/config ---
    fitted_values: dict[str, pd.DataFrame] | None = None
    transform_params: dict[str, dict[str, dict]] | None = None
    metric_instability_flags: pd.DataFrame | None = None
    metric_groups: dict[str, pd.DataFrame] | None = None
    fitted_models: dict[str, bytes] | None = None


@dataclass(frozen=True)
class TestResults:
    """Test fold results. Structurally isolated from CV results."""
    forecasts: pd.DataFrame
    metrics: pd.DataFrame
    test_origin: pd.Timestamp
    train_test_split: dict[str, pd.DataFrame]

    # --- Present depending on model/config ---
    fitted_values: pd.DataFrame | None = None
    transform_params: dict[str, dict] | None = None
    metric_instability_flags: pd.DataFrame | None = None
    metric_groups: pd.DataFrame | None = None
    fitted_model: bytes | None = None


@dataclass(frozen=True)
class BacktestResults:
    """Top-level results containing CV results, test results, and shared metadata."""
    # --- CV results ---
    cv: CVResults

    # --- Shared metadata ---
    horizon: int
    config: dict
    git_hash: str
    uv_lock: str
    run_summary: dict

    # --- Test results (None if test fold disabled) ---
    test: TestResults | None = None

    # --- Escape hatch ---
    extra: dict | None = None
```

Note that `CVResults` uses per-fold structures (e.g., `dict[str, pd.DataFrame]`) because there are multiple CV folds, while `TestResults` uses single DataFrames because there is only one test fold. This makes the types precise to each context.

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

| Artifact | Source | Description |
|---|---|---|
| YAML config | `config` | Full configuration for reproducibility |
| **CV artifacts** | | |
| CV train/val splits | `cv.train_val_splits_per_fold` | Per-fold training and validation DataFrames |
| CV forecasts | `cv.forecasts_per_fold` | Per-fold y_pred aligned to y_true |
| CV fitted values | `cv.fitted_values` | In-sample fitted values (when available) |
| CV metric values | `cv.metrics` | Per-series, grouped, and global metrics |
| CV instability flags | `cv.metric_instability_flags` | Flags for degenerate metric computations |
| CV grouped metrics | `cv.metric_groups` | Metrics by grouping columns |
| CV forecast origins | `cv.fold_origins` | Dates defining each CV fold |
| CV transform params | `cv.transform_params` | Fitted/fixed parameters per fold per series |
| CV fitted models | `cv.fitted_models` | Serialized model per fold (when enabled) |
| **Test artifacts** | | |
| Test train/test split | `test.train_test_split` | Training and test DataFrames |
| Test forecasts | `test.forecasts` | y_pred aligned to y_true |
| Test fitted values | `test.fitted_values` | In-sample fitted values (when available) |
| Test metric values | `test.metrics` | Per-series, grouped, and global metrics |
| Test instability flags | `test.metric_instability_flags` | Flags for degenerate metric computations |
| Test grouped metrics | `test.metric_groups` | Metrics by grouping columns |
| Test origin | `test.test_origin` | Forecast origin date for test fold |
| Test transform params | `test.transform_params` | Fitted/fixed parameters per series |
| Test fitted model | `test.fitted_model` | Serialized model (when enabled) |
| **Shared metadata** | | |
| Forecast horizon | `horizon` | Fixed horizon for the run |
| uv.lock | `uv_lock` | Dependency lockfile contents |
| Git hash | `git_hash` | Code version identifier |
| Run summary | `run_summary` | Success/failure/warning counts and details |

---

## 12. Error Handling and Validation

### 12.1 Logging

The system uses Python's standard `logging` module. Each module creates a logger via `logging.getLogger(__name__)`. The system emits log records for errors, warnings, and informational events (e.g., skipped series, skipped folds, serialization failures). The caller controls log handler configuration, formatting, and destinations. This follows the standard Python convention for library code and integrates seamlessly with Vertex AI pipeline logging and GCP Cloud Logging.

### 12.2 Validation Implementation

- **Config validation** uses Pydantic models to parse and validate the YAML config, providing clear error messages for missing fields, wrong types, and invalid values.
- **Internal data structures** (e.g., `BacktestResults`) use standard dataclasses since they are constructed by the system with already-validated data.

### 12.3 Upfront Validation

The system validates as much as possible before beginning computation:

- YAML config parses without error.
- Required config sections and fields are present.
- Callable import paths for model, metrics, and transforms resolve successfully.
- Required columns (`ds`, `unique_id`, `y`) exist in the target DataFrame.
- Exogenous columns specified in the config exist in the DataFrame.
- Forecast origins (explicit mode) fall within the data's date range.
- Horizon is a positive integer.
- Parametric CV settings produce valid fold origins.
- No contradictory settings (e.g., `model_native: true` with `invertible: true` on the same transform).
- Serialization method is valid if serialization is enabled.
- Test fold origin (if test is enabled) falls within the data's date range.
- Test fold origin (if test is enabled) is after the last CV forecast origin, ensuring no overlap between CV validation data and test data.
- If `config_path` and `config` are both provided, or neither is provided, raise an error.

### 12.4 Series-Level Failures

If a model callable or transform raises an exception for a specific `unique_id`, the system skips that series, logs the error (including the `unique_id`, fold, and exception details), and continues processing remaining series. Warnings emitted during model fitting are captured and logged but do not cause the series to be skipped.

### 12.5 Fold-Level Failures

If an entire fold fails (e.g., a global transform raises an exception), the system logs the error (including the fold identifier and exception details) and continues with remaining folds.

### 12.6 Run Summary

All errors and warnings are aggregated into the `run_summary` field of `BacktestResults`, providing counts, rates, and detailed failure/warning records for post-run triage.

---

## 13. Testing

### 13.1 Framework and Style

All tests use pytest in functional format. No class-based tests. Fixtures are defined in `conftest.py`. Minimize setup/teardown methods in favor of fixtures wherever possible.

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
- **Test fold.** Test fold splitting is correct and isolated from CV folds. Test fold uses the same transforms and metrics as CV. Test results are stored in `TestResults` and structurally separate from `CVResults`. Test fold is `None` when disabled.
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
- **Model adapter pattern.** A formal adapter interface with `fit`, `predict`, and `get_fitted_values` methods, replacing or extending the simple callable convention.
- **BigQuery data source support.**
- **System-computed fitted values.** Computing in-sample fitted values by re-running the model on the training data, for models that do not natively expose fitted values through their API.
- **Per-group transform scope.** A `per_group` scope for transforms that apply to subsets of series grouped by a column (e.g., currency, product class), enabling different transform parameters per group.
- **Metric evaluation on transformed scale.** Option to compute metrics on the transformed scale (skipping inverse transforms) rather than the original scale of `y`. This is relevant for cases like nominal-to-real currency transforms where evaluation on the transformed scale may be preferred. Deferred due to complexity: it affects the transform pipeline, metric computation (context-aware metrics need `y_train` on the same scale), residual scale consistency, and artifact metadata tracking.
- **Native Polars support.** Full native Polars support throughout the internal pipeline (transforms, metrics, model callables) for performance benefits, replacing the V1 approach of converting Polars to pandas at the entry point.
- **Custom company working day normalization.** A built-in transform that supports working day normalization based on a user-provided company working day calendar, accommodating company-specific holidays and non-standard schedules.
- **Transform parallelization.** Parallelization of per-series transform fitting across `unique_id` values, which is especially important when transforms involve a fitting process to find a parameter (e.g., Box-Cox lambda via Guerrero method). V1 parallelizes model fitting (via model's own `n_jobs`) and evaluation (via the system's `multiprocessing`), but transform fitting is sequential.
- **Native hyperparameter optimization.** Built-in integration with Optuna or similar frameworks, allowing the config to specify hyperparameter search spaces instead of fixed values and having the system manage the optimization loop internally. V1 supports hyperparameter optimization via external orchestration using the dict-based config option.
