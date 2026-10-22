"""The real page, run offline with its four remote reads pointed at local runs."""

import base64
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from fsspec.implementations.local import LocalFileSystem
from streamlit.testing.v1 import AppTest

from dashboard.shared import load, paths, runs
from dashboard.train import cohort, contribution, line_grid, origin_skill
from fcstnyctaxi.core.train import backtest_impl
from fcstnyctaxi.core.train.compose_configs_impl import _derive_origins
from fcstnyctaxi.core.train.evaluate_impl import evaluate_impl
from fcstnyctaxi.lib.config.bindings import train_backtest_bindings
from fcstnyctaxi.lib.config.composition import compose_config, save_config
from fcstnyctaxi.lib.storage_layout import composed_config_filename
from fcstnyctaxi.lib.utils import get_project_root_dir
from fcstnyctaxi.schemas.config.train import TrainModelingConfig
from fcstnyctaxi.schemas.run_identity import TrainRunIdentity
from fcstnyctaxi.schemas.storage.common import RUN_IDENTITY_FILENAME
from fcstnyctaxi.schemas.storage.train import (
    BACKTEST_DIR,
    COMPOSE_CONFIGS_DIR,
    EVALUATE_DIR,
)

APP = str(get_project_root_dir() / "dashboard" / "backtest_app.py")
CONFIG_DIR = get_project_root_dir() / "config"
SELECTED = "train-20260929t045735959418z"
OTHER = "train-20260929t045146753815z"
FEATURE_RUN_ID = "20260923t011300349443z"

# A 4/4/5 fiscal year of Sunday weeks, so xgboost's 52-week lag has a year behind
# every origin, then half the next. The two start months give origins mid-month
# and at a month's end, in a 5-week and a 4-week target month.
WEEKS_IN_MONTH = {
    year * 100 + month: (4, 4, 5)[(month - 1) % 3]
    for year, months in ((2024, 12), (2025, 6))
    for month in range(1, months + 1)
}
WEEKS = pd.date_range("2024-01-07", periods=sum(WEEKS_IN_MONTH.values()), freq="W-SUN")
START_MONTHS = [202503, 202504]


def _modeling() -> TrainModelingConfig:
    """The committed modeling config, its start months pinned to `START_MONTHS`."""
    modeling = TrainModelingConfig.model_validate(
        yaml.safe_load((CONFIG_DIR / "train/modeling.yaml").read_text())
    )
    periods = modeling.evaluation_periods.model_copy(
        update={"start_months": START_MONTHS}
    )
    return modeling.model_copy(update={"evaluation_periods": periods})


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
    """The fiscal calendar, with every column Feature's contract allows."""
    rows = [
        {
            "fiscal_year_month": month,
            "fiscal_year": month // 100,
            "fiscal_month": month % 100,
            "fiscal_week_of_month": week,
            "weeks_in_month": weeks,
            "origin_month_fraction_elapsed": week / weeks,
            "count_workdays": 5,
        }
        for month, weeks in WEEKS_IN_MONTH.items()
        for week in range(1, weeks + 1)
    ]
    frame = pd.DataFrame(rows).assign(ds=WEEKS)
    return frame.assign(fiscal_year_week=frame.groupby("fiscal_year").cumcount() + 1)


def _snapshot(modeling: TrainModelingConfig) -> pd.DataFrame:
    """The weekly panel, one series per tier, each level the square of its rank and
    rising 1% a week."""
    return pd.DataFrame(
        [
            {
                "unique_id": _series(index),
                "ds": week,
                "y": 25.0 * (index + 1) ** 2 * (1 + 0.01 * step),
            }
            for index, _ in enumerate(modeling.tiering.tier_labels)
            for step, week in enumerate(WEEKS)
        ]
    )


def _series(index: int) -> str:
    """The synthetic series name at one tier index."""
    return f"time_series_{chr(ord('a') + index)}"


def _exogenous(panel: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Feature's exogenous file: every series over every calendar week."""
    frame = pd.MultiIndex.from_product(
        [panel["unique_id"].unique(), calendar["ds"]], names=["unique_id", "ds"]
    ).to_frame(index=False)
    angle = 2 * np.pi * frame["ds"].dt.dayofyear / 365.25
    return frame.assign(
        holiday_days_in_week=0, week_sin=np.sin(angle), week_cos=np.cos(angle)
    )


def _feature_files(directory: Path) -> dict[str, Path]:
    """Feature's three files, the panel alone stamped with its run id, keyed by
    `backtest_impl`'s parameter names."""
    panel, calendar = _snapshot(_modeling()), _calendar()
    files = {
        "panel_path": directory / "time_series.parquet",
        "calendar_path": directory / "fiscal_calendar.parquet",
        "additional_exog_path": directory / "exogenous_features.parquet",
    }
    panel.assign(feature_run_id=FEATURE_RUN_ID).to_parquet(files["panel_path"])
    calendar.to_parquet(files["calendar_path"])
    _exogenous(panel, calendar).to_parquet(files["additional_exog_path"])
    return files


def _build_run(
    root: Path, feature: dict[str, Path], run_id: str, modeling: TrainModelingConfig
) -> None:
    """One run as the pipeline leaves it: `compose_configs`' outputs, a sidecar per
    model from `backtest_impl`, and `evaluate_impl`'s tables."""
    run = root / run_id
    step = run / COMPOSE_CONFIGS_DIR
    step.mkdir(parents=True)
    (run / RUN_IDENTITY_FILENAME).write_text(_identity(run_id).model_dump_json())
    save_config(
        modeling.model_dump(by_alias=True, exclude_none=True), step / "modeling.yaml"
    )
    origins, _, _ = _derive_origins(
        _snapshot(modeling), _calendar(), modeling.evaluation_periods
    )
    roles = modeling.model_roles
    # One sidecar per model, so one when a model holds both roles.
    for model in dict.fromkeys((roles.challenger, roles.benchmark)):
        composed = compose_config(
            CONFIG_DIR,
            train_backtest_bindings(model),
            {"cross_validation": {"forecast_origins": origins}},
        )
        save_config(
            composed.config.model_dump(by_alias=True, exclude_none=True),
            step / composed_config_filename(model),
        )
        backtest_impl.backtest_impl(
            **feature,
            compose_configs_dir=step,
            model_name=model,
            out_dir=run / BACKTEST_DIR / model,
        )
    evaluate_impl(
        challenger_dir=run / BACKTEST_DIR / roles.challenger,
        benchmark_dir=run / BACKTEST_DIR / roles.benchmark,
        compose_configs_dir=step,
        out_dir=run / EVALUATE_DIR,
    )
    _check_run(run, modeling)


def _check_run(run: Path, modeling: TrainModelingConfig) -> None:
    """Self-check: the models' output exercises what the page tests read."""
    comparison = pd.read_parquet(run / EVALUATE_DIR / paths.PER_SERIES_COMPARISON)
    # Every configured tier at every origin, so every per-tier facet draws.
    tiers = comparison.groupby("forecast_origin_date")["tier"].nunique()
    assert (tiers == len(modeling.tiering.tier_labels)).all(), tiers
    assert set(comparison["horizon"].astype(str)) == {"horizon_1", "horizon_2"}
    roles = modeling.model_roles
    if roles.challenger != roles.benchmark:
        # The two models' forecasts differ, so a crossed sidecar read would show.
        challenger, benchmark = (
            pd.read_parquet(run / BACKTEST_DIR / model / paths.RAW_CV_FORECASTS)
            for model in (roles.challenger, roles.benchmark)
        )
        assert not challenger["ypred"].equals(benchmark["ypred"])


def _point_page_at(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The page's remote reads, pointed at the runs under `root`. The listing is the
    real gate, globbed over the local root."""

    def list_run_ids(slice_root: str) -> tuple[str, ...]:
        return runs.discover_run_ids(f"{root}/", LocalFileSystem())

    def pointer_run_id(pointer_uri: str) -> str:
        return SELECTED

    def evaluate_uri(config_dir: str, env: str, run_id: str) -> str:
        return f"{root / run_id / 'evaluate'}/"

    def run_outputs_uri(config_dir: str, env: str, run_id: str) -> str:
        return f"{root / run_id}/run_output.json"

    def sidecar_uri(config_dir: str, env: str, run_id: str, model_name: str) -> str:
        return f"{root / run_id / BACKTEST_DIR / model_name}/"

    monkeypatch.setattr(runs, "list_run_ids", list_run_ids)
    monkeypatch.setattr(runs, "pointer_run_id", pointer_run_id)
    monkeypatch.setattr(load, "evaluate_uri", evaluate_uri)
    monkeypatch.setattr(load, "run_outputs_uri", run_outputs_uri)
    monkeypatch.setattr(load, "sidecar_uri", sidecar_uri)


@pytest.fixture(scope="module")
def built_runs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Both runs, built once and copied into each test's root."""
    root = tmp_path_factory.mktemp("runs")
    feature = _feature_files(tmp_path_factory.mktemp("feature"))
    for run_id in (SELECTED, OTHER):
        _build_run(root, feature, run_id, _modeling())
    return root


@pytest.fixture
def page_root(
    built_runs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Both runs on local disk, and the page's remote reads pointed there."""
    shutil.copytree(built_runs, tmp_path, dirs_exist_ok=True)
    _point_page_at(tmp_path, monkeypatch)
    return tmp_path


def test_a_run_whose_files_are_its_own_draws_the_whole_page(page_root: Path) -> None:
    """Every pipeline-written run is in this state; also the fixture's self-check."""
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception, app.exception[0].message if app.exception else ""
    assert app.selectbox[0].value == SELECTED
    assert app.get("plotly_chart")


def _annotations(app: AppTest) -> list[dict]:
    """Every annotation on every chart the page drew."""
    return [
        annotation
        for chart in app.get("plotly_chart")
        for annotation in json.loads(chart.proto.spec)["layout"].get("annotations", [])
    ]


def _layout_with(app: AppTest, annotation_name: str) -> dict:
    """The layout of the one chart carrying this annotation."""
    (layout,) = [
        layout
        for layout in (
            json.loads(chart.proto.spec)["layout"] for chart in app.get("plotly_chart")
        )
        if any(
            annotation.get("name") == annotation_name
            for annotation in layout.get("annotations", [])
        )
    ]
    return layout


def test_the_page_draws_the_origin_skill_panel(page_root: Path) -> None:
    """Below the scorecard rows, found by the annotation only that panel carries."""
    app = AppTest.from_file(APP, default_timeout=30).run()
    names = {annotation.get("name") for annotation in _annotations(app)}
    assert origin_skill.COUNTS_ANNOTATION in names


def test_the_page_draws_the_cohort_strip(page_root: Path) -> None:
    """From the challenger's sidecar calendar, found by its row labels."""
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception, app.exception[0].message if app.exception else ""
    names = {annotation.get("name") for annotation in _annotations(app)}
    assert cohort.ROWS_ANNOTATION in names


def test_the_page_draws_the_contribution_scatter_and_a_table_per_facet(
    page_root: Path,
) -> None:
    """Found by the sums annotation only the scatter carries, one sum per facet."""
    app = AppTest.from_file(APP, default_timeout=30).run()
    sums = [
        annotation
        for annotation in _annotations(app)
        if annotation.get("name") == contribution.SUMS_ANNOTATION
    ]
    tables = [table for table in app.dataframe if "series" in table.value.columns]
    assert sums
    assert len(tables) == len(sums)


def test_the_page_switches_the_scatter_to_log_log(page_root: Path) -> None:
    """The toggle reaches the figure the page draws, found by its sums annotation."""
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.radio(key=contribution.SCALE_KEY).set_value("log-log").run()
    assert not app.exception, app.exception[0].message if app.exception else ""
    layout = _layout_with(app, contribution.SUMS_ANNOTATION)
    axis_types = {
        axis.get("type")
        for name, axis in layout.items()
        if name.startswith(("xaxis", "yaxis"))
    }
    assert axis_types == {"log"}


def test_the_page_draws_both_line_grids_in_a_collapsed_expander(
    page_root: Path,
) -> None:
    """The tallest panel, and the one that answers nothing until a series is named."""
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception, app.exception[0].message if app.exception else ""
    (expander,) = [e for e in app.expander if e.label == line_grid.EXPANDER]
    assert not expander.proto.expanded
    assert len(_grids(app)) == line_grid.GRIDS


def _grids(app: AppTest) -> list[dict]:
    """Each line grid's figure JSON, in page order, found by its facet titles."""
    figures = [json.loads(chart.proto.spec) for chart in app.get("plotly_chart")]
    return [
        figure
        for figure in figures
        if any(
            annotation.get("name") == line_grid.TITLE_ANNOTATION
            for annotation in figure["layout"].get("annotations", [])
        )
    ]


def _decoded(values) -> np.ndarray:
    """A trace's values from chart JSON, which Plotly writes as a typed array."""
    if isinstance(values, dict):
        return np.frombuffer(base64.b64decode(values["bdata"]), dtype=values["dtype"])
    return np.asarray(values, dtype=float)


def test_each_roles_weekly_forecast_comes_from_its_own_sidecar(page_root: Path) -> None:
    """Read from the challenger's for both, every benchmark line would be the
    challenger's, on a complete and plausible page."""
    roles = _modeling().model_roles
    app = AppTest.from_file(APP, default_timeout=30).run()
    app.radio(key=line_grid.grain_key(1)).set_value(line_grid.WEEKLY).run()
    assert not app.exception, app.exception[0].message if app.exception else ""
    first_grid = _grids(app)[0]
    drawn = {}
    for model in (roles.challenger, roles.benchmark):
        raw = pd.read_parquet(
            page_root / SELECTED / BACKTEST_DIR / model / paths.RAW_CV_FORECASTS
        )
        # The facet's default origin, the run's last.
        origin = raw["forecast_origin_date"].max()
        expected = raw[
            (raw["forecast_origin_date"] == origin) & (raw["unique_id"] == _series(0))
        ].sort_values("ds")["ypred"]
        # The first facet's line for each horizon, in horizon order. The final
        # month's origins, the last among them, forecast horizon_1 alone.
        lines = [
            next((t for t in first_grid["data"] if t.get("name") == name), None)
            for name in (f"{model} horizon_1", f"{model} horizon_2")
        ]
        drawn[model] = np.concatenate([_decoded(line["y"]) for line in lines if line])
        assert drawn[model].tolist() == pytest.approx(expected.tolist())
    # Self-check: the two models' weeks differ, so a crossed read would show.
    assert drawn[roles.challenger].tolist() != pytest.approx(
        drawn[roles.benchmark].tolist()
    )


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


@pytest.mark.parametrize("role", ["challenger", "benchmark"])
def test_another_runs_sidecar_stops_the_page_before_it_draws(
    page_root: Path, role: str
) -> None:
    """Each role's sidecar, whose tables carry no run id, so its manifest is the
    witness. The message naming the planted run shows the plant tripped it."""
    manifest = (
        Path(BACKTEST_DIR)
        / getattr(_modeling().model_roles, role)
        / paths.BACKTEST_MANIFEST
    )
    shutil.copy(page_root / OTHER / manifest, page_root / SELECTED / manifest)
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert app.exception
    assert OTHER in app.exception[0].message
    assert not app.markdown
    assert not app.get("plotly_chart")


def test_one_model_in_both_roles_reads_one_sidecar(
    tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ModelRoles permits it, and the pipeline then writes one sidecar for both."""
    modeling = _modeling()
    shared = modeling.model_copy(
        update={
            "model_roles": modeling.model_roles.model_copy(
                update={"benchmark": modeling.model_roles.challenger}
            )
        }
    )
    feature = _feature_files(tmp_path_factory.mktemp("feature"))
    _build_run(tmp_path, feature, SELECTED, shared)
    # Self-check: the run holds exactly one sidecar.
    (sidecar,) = (tmp_path / SELECTED / BACKTEST_DIR).iterdir()
    _point_page_at(tmp_path, monkeypatch)

    read = []
    load_backtest_manifest = load.load_backtest_manifest

    def recording_load(prefix: str) -> dict:
        read.append(prefix)
        return load_backtest_manifest(prefix)

    monkeypatch.setattr(load, "load_backtest_manifest", recording_load)
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception, app.exception[0].message if app.exception else ""
    assert read == [f"{sidecar}/"]
