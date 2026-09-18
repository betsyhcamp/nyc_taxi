"""The weekly fold loop, ported from `backtest_weekly.py`'s `evaluate_model`.

Near-verbatim: the port's only proof is a manual diff against the notebook.
"""

import json
import logging
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yaml
from tsbricks.backtesting import evaluate_metrics, generate_folds
from tsbricks.backtesting.schema import AggregationConfig, BacktestConfig
from tsbricks.runner import (
    apply_transforms,
    fit_transforms,
    inverse_transforms,
    invoke_model,
)

from fcstnyctaxi.lib.backtest_results import build_backtest_results, build_cv_results
from fcstnyctaxi.lib.column_checks import (
    require_matching_feature_run_id,
    trim_to_allowlist,
)
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
from fcstnyctaxi.lib.storage_layout import composed_config_filename
from fcstnyctaxi.schemas.config.train import TrainModelingConfig
from fcstnyctaxi.schemas.run_identity import TrainRunIdentity
from fcstnyctaxi.schemas.run_outputs import (
    CALENDAR_ALLOWED_COLUMNS,
    CALENDAR_REQUIRED_COLUMNS,
    PANEL_REQUIRED_COLUMNS,
)

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

# BacktestOutputs field to sidecar file; the write loop and output_rows share it.
_OUTPUT_FILENAMES = {
    "monthly_series": "monthly_series.parquet",
    "monthly_forecast_components": "monthly_forecast_components.parquet",
    "metrics": "metrics.parquet",
    "raw_cv_forecasts": "raw_cv_forecasts.parquet",
}
_MANIFEST_FILENAME = "backtest_manifest.json"


@dataclass(frozen=True)
class BacktestOutputs:
    """The four frames the sidecar persists; the field names are the filenames."""

    monthly_series: pd.DataFrame
    monthly_forecast_components: pd.DataFrame
    metrics: pd.DataFrame
    raw_cv_forecasts: pd.DataFrame


@dataclass(frozen=True)
class BacktestSummary:
    """What this run backtested, for a wrapper that never opens the sidecar.

    Carries both run ids, unlike `ComposeConfigsSummary`: that impl is handed its
    `train_run_id` and this one reads both out of `run_identity.json`, so the return
    value is the only place either discovered value exists.
    """

    n_origins: int
    first_origin: str
    last_origin: str
    n_series: int
    train_run_id: str
    feature_run_id: str
    output_rows: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        """Coerce to JSON-safe primitives: a numpy scalar off a frame breaks KFP."""
        return {
            "n_origins": int(self.n_origins),
            "first_origin": str(self.first_origin),
            "last_origin": str(self.last_origin),
            "n_series": int(self.n_series),
            "train_run_id": str(self.train_run_id),
            "feature_run_id": str(self.feature_run_id),
            "output_rows": {name: int(rows) for name, rows in self.output_rows.items()},
        }


def _require_unique_origins(origin_horizon_pairs: list[tuple]) -> None:
    """Replaces the notebook's `seen_origins` skip; outputs are keyed by fold."""
    repeated = sorted(
        origin
        for origin, count in Counter(o for o, _ in origin_horizon_pairs).items()
        if count > 1
    )
    if repeated:
        raise ValueError(
            f"forecast origins repeat {repeated}: two folds would share one "
            "forecast_origin_date and double-key every output file."
        )


def _require_matching_fold_count(
    cv_folds: dict, origin_horizon_pairs: list[tuple]
) -> None:
    """generate_folds emits one fold per pair, so only an upstream change breaks it."""
    if len(cv_folds) != len(origin_horizon_pairs):
        raise ValueError(
            f"fold count {len(cv_folds)} against {len(origin_horizon_pairs)} "
            "origin/horizon pairs, indexed by fold position: more folds is an "
            "IndexError mid-loop, fewer scores each fold against a prefix."
        )


def _require_unique_monthly_series_keys(monthly_series_df: pd.DataFrame) -> None:
    """Caused by a duplicated unique_id in the panel, which nothing upstream rejects."""
    duplicated = monthly_series_df.duplicated(_MONTHLY_SERIES_KEYS)
    if duplicated.any():
        sample = monthly_series_df.loc[duplicated, _MONTHLY_SERIES_KEYS].head(3)
        raise ValueError(
            f"monthly_series has {int(duplicated.sum())} duplicate key row(s), which "
            f"inflate every weighted sum: {sample.to_dict('records')}"
        )


def _require_mapped_forecast_origins(cv_forecasts_df: pd.DataFrame) -> None:
    """.map() returns NaT for a missing key, silently nulling an output's key column."""
    unmapped = int(cv_forecasts_df["forecast_origin_date"].isna().sum())
    if unmapped:
        raise ValueError(
            f"{unmapped} raw forecast row(s) have a null forecast_origin_date: "
            "fold_id_to_origin does not cover every fold."
        )


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

    _require_unique_origins(origin_horizon_pairs)
    _require_matching_fold_count(cv_folds, origin_horizon_pairs)

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

    _require_unique_monthly_series_keys(monthly_series_df)

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

    _require_mapped_forecast_origins(cv_forecasts_df)

    # Explicit reindex: validate_sidecar asserts the first column is the origin.
    cv_forecasts_df = cv_forecasts_df[_RAW_CV_FORECAST_COLUMNS]

    return BacktestOutputs(
        monthly_series=monthly_series_df,
        monthly_forecast_components=monthly_forecast_components_df,
        metrics=backtest_results.cv.metrics,
        raw_cv_forecasts=cv_forecasts_df,
    )


def _origin_label(origin: pd.Timestamp | int) -> str:
    """The spelling `compose_configs` records, so one run states an origin one way."""
    return (
        origin.date().isoformat() if isinstance(origin, pd.Timestamp) else str(origin)
    )


def _build_manifest(
    model_name: str,
    summary: BacktestSummary,
    identity: TrainRunIdentity,
    modeling: TrainModelingConfig,
) -> dict[str, Any]:
    """Explicit mapping from three single sources, so nothing here is counted twice.

    The summary owns what was measured, the identity provenance, and `modeling` the
    tiering and weighting that change every score and sit in no other sidecar file.
    """
    return {
        "model_name": model_name,
        "lineage": {
            "train_run_id": identity.train_run_id,
            "feature_run_id": identity.feature_run_id,
            "git_hash": identity.git_hash,
            "panel_uri": identity.panel_uri,
            "calendar_uri": identity.calendar_uri,
        },
        "config": {
            "tiering": modeling.tiering.model_dump(),
            "weighting": modeling.weighting.model_dump(),
        },
        "origins": {
            "n_origins": summary.n_origins,
            "first_origin": summary.first_origin,
            "last_origin": summary.last_origin,
        },
        "n_series": summary.n_series,
        "output_rows": summary.output_rows,
    }


def backtest_impl(
    *,
    panel_path: Path,
    calendar_path: Path,
    compose_configs_dir: Path,
    model_name: str,
    out_dir: Path,
) -> BacktestSummary:
    """Back one model over one run's origins and write its eight-file sidecar.

    Keyword-only: three adjacent `Path` parameters transpose without a type error, and
    a swapped panel and calendar surfaces much later as a missing column. No provenance
    scalars: reading `run_identity.json` makes a disagreeing parameter unrepresentable.
    The completion marker is deleted once the guards pass, so a failed run leaves no
    marker even when it has overwritten part of an earlier sidecar.

    Args:
        panel_path: The weekly actuals, stamped with a `feature_run_id`.
        calendar_path: The fiscal calendar, same stamping.
        compose_configs_dir: Holds this model's composed config and `modeling.yaml`,
            with `run_identity.json` beside it.
        model_name: Selects the composed config, and names `out_dir`.
        out_dir: This model's sidecar directory, created if missing.

    Raises:
        ValueError: If `out_dir` is not this model's directory under that run root, if
            `run_identity.json` is absent, or on a failed lineage or column check.
        ValidationError: If a config or the identity fails to revalidate on read.

    Returns:
        BacktestSummary: What was backtested, for a wrapper or the local runner.
    """
    if out_dir.name != model_name:
        raise ValueError(
            f"out_dir {out_dir} must be named for its model {model_name!r}: the "
            "directory name is what tells two models' sidecars apart."
        )

    identity_path = compose_configs_dir.parent / "run_identity.json"
    if not identity_path.is_file():
        raise ValueError(
            f"No run_identity.json at {identity_path}: compose_configs writes it "
            "beside its step directory, so that path is wrong or the step failed."
        )
    identity = TrainRunIdentity.model_validate_json(identity_path.read_text())

    # Against the parsed id, not compose_configs_dir.parent as paths: the two arrive by
    # different channels, so two spellings of one place would fire on correct wiring.
    if out_dir.parent.parent.name != identity.train_run_id:
        raise ValueError(
            f"out_dir {out_dir} must sit under the run root "
            f"{identity.train_run_id!r}: the manifest would name a run it is not under."
        )

    # Otherwise a failed rerun leaves the old manifest over a mix of two runs' files.
    (out_dir / _MANIFEST_FILENAME).unlink(missing_ok=True)

    composed_config_path = compose_configs_dir / composed_config_filename(model_name)
    cfg = BacktestConfig.model_validate(
        yaml.safe_load(composed_config_path.read_text())
    )
    modeling = TrainModelingConfig.model_validate(
        yaml.safe_load((compose_configs_dir / "modeling.yaml").read_text())
    )

    panel_df = pd.read_parquet(panel_path)
    calendar_df = pd.read_parquet(calendar_path)
    # Catches a DAG wiring a different panel here than compose_configs read.
    require_matching_feature_run_id(panel_df, calendar_df, identity.feature_run_id)

    # Before the snapshots: a moved upstream timestamp must not fail input equivalence.
    panel_df = trim_to_allowlist(
        panel_df, required=PANEL_REQUIRED_COLUMNS, frame_name="panel"
    )
    calendar_df = trim_to_allowlist(
        calendar_df,
        required=CALENDAR_REQUIRED_COLUMNS,
        allowed=CALENDAR_ALLOWED_COLUMNS,
        frame_name="calendar",
    )

    outputs = compute_backtest_outputs(
        cfg=cfg, modeling=modeling, ts_df=panel_df, calendar_df=calendar_df
    )

    # Not sorted(str(...)): 10 sorts before 2, and `2025-4-27` after `2025-05-04`.
    origins = [
        _origin_label(origin)
        for origin, _ in normalized_origin_horizon_pairs(
            cfg.cross_validation.origin_horizon_pairs(), cfg.data.freq
        )
    ]
    summary = BacktestSummary(
        n_origins=len(origins),
        first_origin=origins[0],
        last_origin=origins[-1],
        n_series=int(panel_df["unique_id"].nunique()),
        train_run_id=identity.train_run_id,
        feature_run_id=identity.feature_run_id,
        output_rows={
            filename: len(getattr(outputs, field))
            for field, filename in _OUTPUT_FILENAMES.items()
        },
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    for field, filename in _OUTPUT_FILENAMES.items():
        getattr(outputs, field).to_parquet(out_dir / filename, index=False)
    calendar_df.to_parquet(out_dir / "fiscal_calendar.parquet", index=False)
    panel_df.to_parquet(out_dir / "time_series_snapshot.parquet", index=False)
    # A byte copy, not a re-dump: a re-dump reimplements save_config's formatting.
    shutil.copyfile(composed_config_path, out_dir / "composed_config.yaml")

    manifest = _build_manifest(model_name, summary, identity, modeling)
    # backtest_manifest.json written is the step's completion marker. Keep this last.
    (out_dir / _MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2) + "\n")

    _log.info(
        "backtest complete: model=%s origins=%d series=%d out_dir=%s",
        model_name,
        summary.n_origins,
        summary.n_series,
        out_dir,
    )
    return summary
