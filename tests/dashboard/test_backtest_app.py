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
from dashboard.train import origin_skill
from fcstnyctaxi.core.train.evaluate_impl import (
    _OUTPUT_FILENAMES,
    _build_manifest,
    _build_summary,
    compute_evaluate_outputs,
)
from fcstnyctaxi.lib.utils import get_project_root_dir
from fcstnyctaxi.schemas.config.train import TrainModelingConfig
from fcstnyctaxi.schemas.run_identity import TrainRunIdentity

APP = str(get_project_root_dir() / "dashboard" / "backtest_app.py")
SELECTED = "train-20260929t045735959418z"
OTHER = "train-20260929t045146753815z"
FEATURE_RUN_ID = "20260923t011300349443z"

# Half a 4/4/5 fiscal year of Sunday weeks. Three origins in the 5-week month:
# two mid-month, forecasting it and the next, and one at its end, which has
# completed it and forecasts the two after.
WEEKS = pd.date_range("2025-01-05", periods=26, freq="W-SUN")
WEEKS_IN_MONTH = {202501: 4, 202502: 4, 202503: 5, 202504: 4, 202505: 4, 202506: 5}
ORIGIN_TARGETS = (
    (WEEKS[10], (202503, 202504)),
    (WEEKS[11], (202503, 202504)),
    (WEEKS[12], (202504, 202505)),
)


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


def _calendar() -> pd.DataFrame:
    """The calendar columns evaluate reads, each week's fraction of its month."""
    months = [month for month, weeks in WEEKS_IN_MONTH.items() for _ in range(weeks)]
    elapsed = [
        week / weeks
        for weeks in WEEKS_IN_MONTH.values()
        for week in range(1, weeks + 1)
    ]
    return pd.DataFrame(
        {
            "ds": WEEKS,
            "fiscal_year_month": months,
            "origin_month_fraction_elapsed": elapsed,
        }
    )


def _monthly_series(model: str, modeling: TrainModelingConfig) -> pd.DataFrame:
    """One sidecar's monthly_series, one series per tier. The benchmark errs by 20%
    and up, the challenger by 8% and up, rising by fold, series and horizon."""
    tiers = modeling.tiering.tier_labels
    base_error = 0.20 if model == modeling.model_roles.benchmark else 0.08
    rows = []
    for fold_index, (origin, months) in enumerate(ORIGIN_TARGETS):
        for step, month in enumerate(months):
            for index, tier in enumerate(tiers):
                actual = 100.0 * (index + 1) ** 2 * (1 + 0.05 * (month - 202503))
                error = base_error + 0.01 * fold_index + 0.02 * index + 0.03 * step
                rows.append(
                    {
                        "forecast_origin_date": origin,
                        "predicted_fiscal_year_month": month,
                        "unique_id": f"time_series_{chr(ord('a') + index)}",
                        "tier": tier,
                        "series_weight": 2.0 * (index + 1),
                        "monthly_forecast": actual * (1 + (-1) ** index * error),
                        "actual_monthly_total": actual,
                    }
                )
    frame = pd.DataFrame(rows)
    frame["tier"] = pd.Categorical(frame["tier"], categories=tiers, ordered=True)
    return frame


def _write_run(root: Path, run_id: str) -> None:
    """One run's evaluate directory, every table and the manifest from evaluate's
    own builders, written as `evaluate_impl` writes them."""
    modeling = _modeling()
    roles = modeling.model_roles
    identity = _identity(run_id)
    outputs = compute_evaluate_outputs(
        challenger_ms=_monthly_series(roles.challenger, modeling),
        benchmark_ms=_monthly_series(roles.benchmark, modeling),
        calendar_df=_calendar(),
        challenger_model=roles.challenger,
        benchmark_model=roles.benchmark,
        tier_labels=tuple(modeling.tiering.tier_labels),
        identity=identity,
    )
    # Self-check: the stated target months are the two horizons evaluate labels.
    horizons = set(outputs.per_series_comparison["horizon"])
    assert horizons == {"horizon_1", "horizon_2"}, horizons

    evaluate = root / run_id / "evaluate"
    evaluate.mkdir(parents=True)
    for field, filename in _OUTPUT_FILENAMES.items():
        getattr(outputs, field).to_parquet(evaluate / filename, index=False)
    manifest = _build_manifest(
        _build_summary(outputs, identity, roles), identity, modeling
    )
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


def test_the_page_draws_the_origin_skill_panel(page_root: Path) -> None:
    """Below the scorecard rows, found by the annotation only that panel carries."""
    app = AppTest.from_file(APP, default_timeout=30).run()
    names = {
        annotation.get("name")
        for chart in app.get("plotly_chart")
        for annotation in json.loads(chart.proto.spec)["layout"].get("annotations", [])
    }
    assert origin_skill.COUNTS_ANNOTATION in names


@pytest.mark.parametrize(
    "filename",
    [
        paths.EVALUATE_MANIFEST,
        paths.SUMMARY_METRICS,
        paths.FOLD_METRICS,
        paths.PER_SERIES_COMPARISON,
    ],
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
