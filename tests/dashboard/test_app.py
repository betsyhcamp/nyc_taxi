"""The real page, run offline with its four remote reads pointed at local runs."""

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml
from fsspec.implementations.local import LocalFileSystem
from streamlit.testing.v1 import AppTest

from dashboard.shared import load, paths, runs
from fcstnyctaxi.core.train.evaluate_impl import (
    _METRIC_NAMES,
    _PERIOD_KEYS,
    _SCORE_KEYS,
    GLOBAL_TIER,
    EvaluateOutputs,
    _build_manifest,
    _build_summary,
    _derive,
    _stamp_lineage,
)
from fcstnyctaxi.lib.utils import get_project_root_dir
from fcstnyctaxi.schemas.config.train import TrainModelingConfig
from fcstnyctaxi.schemas.run_identity import TrainRunIdentity

APP = str(get_project_root_dir() / "dashboard" / "app.py")
SELECTED = "20260929t045735959418z"
OTHER = "20260929t045146753815z"
FEATURE_RUN_ID = "20260923t011300349443z"
ORIGIN = pd.Timestamp("2025-03-23")
PREDICTED_MONTH = {"horizon_1": 202504, "horizon_2": 202505}


def _modeling() -> TrainModelingConfig:
    """The committed modeling config, which supplies the roles and tier labels."""
    return TrainModelingConfig.model_validate(
        yaml.safe_load(
            (get_project_root_dir() / "config/train/modeling.yaml").read_text()
        )
    )


def _identity(run_id: str) -> TrainRunIdentity:
    """One run's provenance, as `compose_configs` writes it."""
    base = f"gs://BUCKET/dev/feature/{FEATURE_RUN_ID}/data_prep"
    return TrainRunIdentity(
        git_hash="0" * 40,
        feature_run_id=FEATURE_RUN_ID,
        train_run_id=run_id,
        panel_uri=f"{base}/time_series.parquet",
        calendar_uri=f"{base}/fiscal_calendar.parquet",
        additional_exog_uri=f"{base}/exogenous_features.parquet",
    )


def _write_run(root: Path, run_id: str) -> None:
    """One run's evaluate directory. One series per tier, so every tier is populated
    at the one origin, as it is on a real run."""
    modeling = _modeling()
    roles = modeling.model_roles
    tiers = modeling.tiering.tier_labels
    series = [f"time_series_{chr(ord('a') + index)}" for index in range(len(tiers))]
    identity = _identity(run_id)

    folds = pd.DataFrame(
        [
            {
                "model": model,
                "horizon": horizon,
                "tier": tier,
                "forecast_origin_date": ORIGIN,
                "predicted_fiscal_year_month": month,
                "metric": metric,
                "value": 0.9,
                "n_obs": len(series) if tier == GLOBAL_TIER else 1,
            }
            for model in (roles.challenger, roles.benchmark)
            for horizon, month in PREDICTED_MONTH.items()
            for tier in (GLOBAL_TIER, *tiers)
            for metric in _METRIC_NAMES
        ]
    )
    per_series = pd.DataFrame(
        [
            {
                "forecast_origin_date": ORIGIN,
                "predicted_fiscal_year_month": month,
                "unique_id": unique_id,
            }
            for month in PREDICTED_MONTH.values()
            for unique_id in series
        ]
    )
    outputs = EvaluateOutputs(
        per_series_comparison=_stamp_lineage(per_series, identity),
        fold_metrics=_stamp_lineage(folds, identity),
        period_metrics=_stamp_lineage(_derive(folds, _PERIOD_KEYS), identity),
        summary_metrics=_stamp_lineage(_derive(folds, _SCORE_KEYS), identity),
    )
    manifest = _build_manifest(
        _build_summary(outputs, identity, roles), identity, modeling
    )

    evaluate = root / run_id / "evaluate"
    evaluate.mkdir(parents=True)
    outputs.fold_metrics.to_parquet(evaluate / paths.FOLD_METRICS)
    outputs.summary_metrics.to_parquet(evaluate / paths.SUMMARY_METRICS)
    (evaluate / paths.EVALUATE_MANIFEST).write_text(json.dumps(manifest))


@pytest.fixture
def page_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Both runs on local disk, and the page's four remote reads pointed there. The
    listing is the real gate, globbed over the local root."""
    for run_id in (SELECTED, OTHER):
        _write_run(tmp_path, run_id)

    def list_run_ids(slice_root: str) -> tuple[str, ...]:
        return runs.discover_run_ids(f"{tmp_path}/", LocalFileSystem())

    def pointer_run_id(pointer_uri: str) -> str:
        return SELECTED

    def evaluate_uri(config_dir: str, env: str, run_id: str) -> str:
        return f"{tmp_path / run_id / 'evaluate'}/"

    def run_outputs_uri(config_dir: str, env: str, run_id: str) -> str:
        return f"{tmp_path / run_id}/run_output.json"

    monkeypatch.setattr(runs, "list_run_ids", list_run_ids)
    monkeypatch.setattr(runs, "pointer_run_id", pointer_run_id)
    monkeypatch.setattr(load, "evaluate_uri", evaluate_uri)
    monkeypatch.setattr(load, "run_outputs_uri", run_outputs_uri)
    return tmp_path


def test_a_run_whose_files_are_its_own_draws_the_whole_page(page_root: Path) -> None:
    """Every pipeline-written run is in this state; also the fixture's self-check."""
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception, app.exception[0].message if app.exception else ""
    assert app.selectbox[0].value == SELECTED
    assert app.get("plotly_chart")


@pytest.mark.parametrize(
    "filename", [paths.EVALUATE_MANIFEST, paths.SUMMARY_METRICS, paths.FOLD_METRICS]
)
def test_another_runs_file_stops_the_page_before_it_draws(
    page_root: Path, filename: str
) -> None:
    """Checked any later, the strip or the charts describe another run above the
    error. The message naming the planted run shows the plant tripped it."""
    shutil.copy(
        page_root / OTHER / "evaluate" / filename,
        page_root / SELECTED / "evaluate" / filename,
    )
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert app.exception
    assert OTHER in app.exception[0].message
    assert not app.markdown
    assert not app.get("plotly_chart")
