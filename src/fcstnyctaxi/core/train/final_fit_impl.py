import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from tsbricks.backtesting.schema import BacktestConfig
from tsbricks.runner import dynamic_import

from fcstnyctaxi.lib.column_checks import (
    require_matching_feature_run_id,
    trim_to_allowlist,
)
from fcstnyctaxi.lib.exog import build_exog_frame
from fcstnyctaxi.lib.storage_layout import (
    BUNDLE_MODEL_DIR_NAME,
    composed_config_filename,
)
from fcstnyctaxi.schemas.config.train import ModelSettings, TrainModelingConfig
from fcstnyctaxi.schemas.run_identity import TrainRunIdentity
from fcstnyctaxi.schemas.run_outputs import (
    CALENDAR_ALLOWED_COLUMNS,
    CALENDAR_REQUIRED_COLUMNS,
    PANEL_REQUIRED_COLUMNS,
)

_log = logging.getLogger(__name__)

_MANIFEST_FILENAME = "final_fit_manifest.json"


@dataclass(frozen=True)
class FinalFitSummary:
    """What the bundle was trained on, for a wrapper that never opens it. That makes
    this the only channel from the bundle to the metadata a registry entry carries."""

    train_end_ds: str
    n_series: int
    n_obs: int
    train_run_id: str
    feature_run_id: str

    def as_dict(self) -> dict[str, Any]:
        """Coerce what a frame can hand over as a Timestamp or numpy scalar. The run
        ids stay bare: `TrainRunIdentity` has already validated them as `str`."""
        return {
            "train_end_ds": str(self.train_end_ds),
            "n_series": int(self.n_series),
            "n_obs": int(self.n_obs),
            "train_run_id": self.train_run_id,
            "feature_run_id": self.feature_run_id,
        }


def _require_no_transforms(cfg: BacktestConfig) -> None:
    """A fitted transform has no place in the bundle and no reader to invert it."""
    if cfg.transforms:
        raise ValueError(
            f"final_fit takes no transforms, got {[t.name for t in cfg.transforms]}."
        )


def _require_written_model(model_dir: Path) -> None:
    """A save that writes nothing, or an empty file, would register an empty model."""
    files = [path for path in model_dir.rglob("*") if path.is_file()]
    if not files:
        raise ValueError(f"save wrote no file to {model_dir}.")

    empty = sorted(
        str(path.relative_to(model_dir)) for path in files if path.stat().st_size == 0
    )
    if empty:
        raise ValueError(f"save wrote zero-byte file(s) {empty} to {model_dir}.")


def _build_manifest(
    model_name: str,
    summary: FinalFitSummary,
    identity: TrainRunIdentity,
    settings: ModelSettings,
) -> dict[str, Any]:
    """Explicit mapping from three single sources, so nothing is counted twice. The
    config block is what a reader holding only the bundle needs to load and predict."""
    return {
        "model_name": model_name,
        "lineage": {
            "train_run_id": identity.train_run_id,
            "feature_run_id": identity.feature_run_id,
            "git_hash": identity.git_hash,
            "panel_uri": identity.panel_uri,
            "calendar_uri": identity.calendar_uri,
            "additional_exog_uri": identity.additional_exog_uri,
        },
        "config": {
            "fit_callable": settings.fit_callable,
            "save_callable": settings.save_callable,
            "exog_features": settings.exog_features,
        },
        "training_data": {
            "train_end_ds": summary.train_end_ds,
            "n_series": summary.n_series,
            "n_obs": summary.n_obs,
        },
    }


def final_fit_impl(
    *,
    panel_path: Path,
    calendar_path: Path,
    compose_configs_dir: Path,
    model_name: str,
    out_dir: Path,
) -> FinalFitSummary:
    """Fit one model on the whole panel and write the bundle a registry consumes.

    Keyword-only: four `Path` parameters transpose without a type error. No
    provenance scalars: reading `run_identity.json` makes a disagreeing parameter
    unrepresentable. The completion marker is deleted once the directory checks pass
    and written last, so a failed rerun leaves none over a partly rewritten bundle.

    Args:
        panel_path: The weekly actuals, stamped with a `feature_run_id`.
        calendar_path: The fiscal calendar, unstamped: Feature stamps the panel alone.
        compose_configs_dir: Holds this model's composed config and `modeling.yaml`,
            with `run_identity.json` beside it.
        model_name: The model to fit, which must declare its callable pair. Selects
            the composed config and names `out_dir`.
        out_dir: This model's bundle directory, created if missing.

    Raises:
        ValueError: If `out_dir` is not this model's directory under that run root,
            if `run_identity.json` is absent, if the config declares transforms or
            the model no callables, on a failed lineage, column or exogenous-join
            check, or if the save wrote nothing or a zero-byte file.
        ValidationError: If a config or the identity fails to revalidate on read.

    Returns:
        FinalFitSummary: What the bundle was trained on, for a wrapper or the local
            runner.
    """
    # A registered URI ending in the model name is what makes it self-describing.
    if out_dir.name != model_name:
        raise ValueError(f"out_dir {out_dir} is not named for model {model_name!r}.")

    identity_path = compose_configs_dir.parent / "run_identity.json"
    if not identity_path.is_file():
        raise ValueError(
            f"No run_identity.json at {identity_path}, which compose_configs writes."
        )
    identity = TrainRunIdentity.model_validate_json(identity_path.read_text())

    # Against the parsed id, not compose_configs_dir.parent as paths: the two arrive by
    # different channels, so two spellings of one place would fire on correct wiring.
    if out_dir.parent.parent.name != identity.train_run_id:
        raise ValueError(
            f"out_dir {out_dir} is not under run root {identity.train_run_id!r}."
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
    settings = modeling.model_settings[model_name]

    _require_no_transforms(cfg)
    # The schema requires the pair of the challenger alone, and both-or-neither of
    # every model, so one None check covers any model a caller names.
    if settings.fit_callable is None:
        raise ValueError(
            f"model {model_name!r} declares no fit_callable and save_callable."
        )
    # Both resolved before the fit, so a mistyped save path costs no training.
    fit_fn = dynamic_import(settings.fit_callable)
    save_fn = dynamic_import(settings.save_callable)

    panel_df = pd.read_parquet(panel_path)
    calendar_df = pd.read_parquet(calendar_path)
    # Catches a panel from another Feature run, or wrong bytes at the path handed in.
    require_matching_feature_run_id(panel_df, identity.feature_run_id)

    panel_df = trim_to_allowlist(
        panel_df, required=PANEL_REQUIRED_COLUMNS, frame_name="panel"
    )
    calendar_df = trim_to_allowlist(
        calendar_df,
        required=CALENDAR_REQUIRED_COLUMNS,
        allowed=CALENDAR_ALLOWED_COLUMNS,
        frame_name="calendar",
    )

    # The backtest assembles from the same entry, so the registered model trains on
    # the feature set that was scored.
    exog_df = build_exog_frame(
        panel_df, calendar_df, exog_features=tuple(settings.exog_features)
    )

    # hyperparameters alone: predict_params belong to the fit-predict callable.
    model_obj = fit_fn(panel_df, exog_df=exog_df, **(cfg.model.hyperparameters or {}))

    # Recreated rather than reused: a rerun's files would pass the write check for a
    # save that wrote nothing. Created here, not by the save, for the same reason.
    model_dir = out_dir / BUNDLE_MODEL_DIR_NAME
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True)
    save_fn(model_obj, model_dir)
    _require_written_model(model_dir)

    # A byte copy, not a re-dump: a re-dump reimplements save_config's formatting.
    shutil.copyfile(composed_config_path, out_dir / "composed_config.yaml")

    summary = FinalFitSummary(
        train_end_ds=pd.Timestamp(panel_df["ds"].max()).date().isoformat(),
        n_series=int(panel_df["unique_id"].nunique()),
        n_obs=len(panel_df),
        train_run_id=identity.train_run_id,
        feature_run_id=identity.feature_run_id,
    )
    manifest = _build_manifest(model_name, summary, identity, settings)
    # final_fit_manifest.json written is the step's completion marker. Keep this last.
    (out_dir / _MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2) + "\n")

    _log.info(
        "final_fit complete: model=%s train_end_ds=%s series=%d obs=%d out_dir=%s",
        model_name,
        summary.train_end_ds,
        summary.n_series,
        summary.n_obs,
        out_dir,
    )
    return summary
