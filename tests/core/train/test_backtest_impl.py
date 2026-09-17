import dataclasses
import json
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
from tsbricks.backtesting import generate_folds
from tsbricks.backtesting.schema import BacktestConfig

from fcstnyctaxi.core.train.backtest_impl import (
    _MONTHLY_SERIES_KEYS,
    _OUTPUT_FILENAMES,
    BacktestOutputs,
    BacktestSummary,
    _build_manifest,
    backtest_impl,
    compute_backtest_outputs,
)
from fcstnyctaxi.lib.backtest_results import build_cv_results
from fcstnyctaxi.lib.config.bindings import (
    train_backtest_bindings,
    train_modeling_bindings,
)
from fcstnyctaxi.lib.config.composition import (
    compose_config,
    merge_configs,
    save_config,
)
from fcstnyctaxi.lib.monthly_aggregation import attach_tier_and_weight
from fcstnyctaxi.lib.period_utils import (
    derive_start_months,
    generate_origins_for_periods,
    last_complete_actual_month,
)
from fcstnyctaxi.lib.storage_layout import composed_config_filename
from fcstnyctaxi.lib.utils import get_project_root_dir
from fcstnyctaxi.schemas.config.train import TrainModelingConfig
from fcstnyctaxi.schemas.run_identity import TrainRunIdentity
from fcstnyctaxi.schemas.run_outputs import (
    CALENDAR_ALLOWED_COLUMNS,
    PANEL_REQUIRED_COLUMNS,
)

CONFIG_DIR = get_project_root_dir() / "config"
FEATURE_RUN_ID = "f-2026-09-17"
TRAIN_RUN_ID = "t-2026-09-17"
MODEL_NAME = "naive"

# ================================================
# Fixtures
#
# 16 weeks, W-SUN, four fiscal months of four each (202501..202504). Both origins
# sit at a month end with horizon 2, so every fold forecasts into the next month.
#
# The model is a fake resolved by dotted path, as in tests/lib/test_calibration.py:
# these tests prove the assertions fire, not that any number is right.
# ================================================

_WEEKS = pd.date_range("2025-01-05", periods=16, freq="W-SUN")
_SERIES = {"a": 10.0, "b": 5.0, "c": 1.0}

_fit_calls: list[int] = []
"""One entry per fake-model call, so a test can assert no fold ran."""


@pytest.fixture(autouse=True)
def _reset_fit_calls() -> None:
    """Module state, so every test starts from empty."""
    _fit_calls.clear()


def _fake_model_callable(
    train_df: pd.DataFrame, horizon: int, future_x_df: pd.DataFrame, **kwargs: Any
) -> pd.DataFrame:
    """Repeat each series' last observed value, and record that a fold ran."""
    _fit_calls.append(horizon)

    last_ds = train_df["ds"].max()
    future = (
        future_x_df.loc[future_x_df["ds"] > last_ds, "ds"]
        .sort_values()
        .head(horizon)
        .tolist()
    )
    last_value = train_df.sort_values("ds").groupby("unique_id")["y"].last()
    ids = sorted(train_df["unique_id"].unique())

    return pd.DataFrame(
        {
            "unique_id": [uid for uid in ids for _ in future],
            "ds": future * len(ids),
            "ypred": [float(last_value[uid]) for uid in ids for _ in future],
        }
    )


@pytest.fixture
def calendar_df() -> pd.DataFrame:
    """Only the columns the fold loop reads, not the full calendar contract."""
    week_of_month = [i % 4 + 1 for i in range(len(_WEEKS))]
    return pd.DataFrame(
        {
            "ds": _WEEKS,
            "fiscal_year_month": [202501 + i // 4 for i in range(len(_WEEKS))],
            "origin_month_fraction_elapsed": [w / 4 for w in week_of_month],
        }
    )


@pytest.fixture
def ts_df() -> pd.DataFrame:
    """Three series with distinct trailing means, so qcut gets three bins."""
    return pd.DataFrame(
        {
            "unique_id": [uid for uid in _SERIES for _ in _WEEKS],
            "ds": list(_WEEKS) * len(_SERIES),
            "y": [value for value in _SERIES.values() for _ in _WEEKS],
        }
    )


def _shipped_modeling() -> TrainModelingConfig:
    """The committed tiering, weighting and evaluation periods."""
    return cast(
        TrainModelingConfig,
        compose_config(CONFIG_DIR, train_modeling_bindings()).config,
    )


@pytest.fixture
def modeling() -> TrainModelingConfig:
    """The shipped tiering and weighting, composed rather than hand-built."""
    return _shipped_modeling()


def _cfg(origins: list[tuple[int, int]]) -> BacktestConfig:
    """Compose off the production tree; origins are (week index, horizon)."""
    return merge_configs(
        CONFIG_DIR / "base" / "data.yaml",
        CONFIG_DIR / "train" / "backtest.yaml",
        {
            "model": {
                "fit_predict_callable": "test_backtest_impl._fake_model_callable",
                "hyperparameters": {"freq": "W-SUN"},
            },
            "cross_validation": {
                "forecast_origins": [
                    {"origin": str(_WEEKS[index].date()), "horizon": horizon}
                    for index, horizon in origins
                ]
            },
        },
    )


_TWO_ORIGINS = [(7, 2), (11, 2)]


# ================================================
# compute_backtest_outputs: the loop runs
# ================================================


def test_the_fold_loop_produces_one_origin_per_configured_origin(
    ts_df: pd.DataFrame, calendar_df: pd.DataFrame, modeling: TrainModelingConfig
) -> None:
    """The fixture's own check: without it a broken fixture reads as a broken check."""
    outputs = compute_backtest_outputs(
        cfg=_cfg(_TWO_ORIGINS), modeling=modeling, ts_df=ts_df, calendar_df=calendar_df
    )

    assert len(_fit_calls) == len(_TWO_ORIGINS)
    assert outputs.raw_cv_forecasts["forecast_origin_date"].nunique() == len(
        _TWO_ORIGINS
    )
    assert not outputs.monthly_series.empty


# ================================================
# The four assertions
#
# Only the repeated-origins check is reachable through cfg and data. The other three
# replace a collaborator in the impl's namespace, as tests/lib/test_calibration.py
# does. All four run the real function: the defect they catch is an unwired check,
# which a direct call to the predicate would pass.
# ================================================


def test_repeated_forecast_origins_raise_before_the_loop(
    ts_df: pd.DataFrame, calendar_df: pd.DataFrame, modeling: TrainModelingConfig
) -> None:
    """BacktestConfig rejects a repeated pair, so the reachable case is two horizons."""
    with pytest.raises(ValueError, match="forecast origins repeat"):
        compute_backtest_outputs(
            cfg=_cfg([(7, 2), (7, 3)]),
            modeling=modeling,
            ts_df=ts_df,
            calendar_df=calendar_df,
        )

    assert _fit_calls == []


def test_a_fold_count_disagreeing_with_the_pairs_raises_before_the_loop(
    ts_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    modeling: TrainModelingConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """generate_folds emits one fold per pair, so only an upstream change does this."""

    def _drop_the_last_fold(
        df: pd.DataFrame, cv_config: Any, data_config: Any, test_config: Any = None
    ) -> tuple[dict, dict | None]:
        folds, test_split = generate_folds(df, cv_config, data_config, test_config)
        return dict(list(folds.items())[:-1]), test_split

    monkeypatch.setattr(
        "fcstnyctaxi.core.train.backtest_impl.generate_folds", _drop_the_last_fold
    )

    with pytest.raises(ValueError, match="origin/horizon"):
        compute_backtest_outputs(
            cfg=_cfg(_TWO_ORIGINS),
            modeling=modeling,
            ts_df=ts_df,
            calendar_df=calendar_df,
        )

    assert _fit_calls == []


def test_an_unmapped_fold_id_raises_instead_of_nulling_the_origin(
    ts_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    modeling: TrainModelingConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """.map() returns NaT for a missing key, silently nulling an output's key column."""

    def _cv_results_missing_one_origin(**kwargs: Any) -> Any:
        cv_results = build_cv_results(**kwargs)
        first, *_ = cv_results.fold_id_to_origin
        return dataclasses.replace(
            cv_results,
            fold_id_to_origin={first: cv_results.fold_id_to_origin[first]},
        )

    monkeypatch.setattr(
        "fcstnyctaxi.core.train.backtest_impl.build_cv_results",
        _cv_results_missing_one_origin,
    )

    with pytest.raises(ValueError, match="null forecast_origin_date"):
        compute_backtest_outputs(
            cfg=_cfg(_TWO_ORIGINS),
            modeling=modeling,
            ts_df=ts_df,
            calendar_df=calendar_df,
        )


def test_duplicate_monthly_series_keys_raise(
    ts_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    modeling: TrainModelingConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A duplicated key inflates every weighted sum while the run looks healthy."""

    def _attach_twice(**kwargs: Any) -> pd.DataFrame:
        rows = attach_tier_and_weight(**kwargs)
        return pd.concat([rows, rows], ignore_index=True)

    monkeypatch.setattr(
        "fcstnyctaxi.core.train.backtest_impl.attach_tier_and_weight", _attach_twice
    )

    # One origin, so the duplication is this test's and not a repeated origin.
    with pytest.raises(ValueError, match="duplicate key row"):
        compute_backtest_outputs(
            cfg=_cfg([(7, 2)]),
            modeling=modeling,
            ts_df=ts_df,
            calendar_df=calendar_df,
        )


# ================================================
# backtest_impl: the guards above the first write
#
# Each fires before the panel and calendar are opened, so a run identity is the most
# any needs staged. The end-to-end run belongs with the fixture that anchors a golden.
# ================================================


def _identity() -> TrainRunIdentity:
    """The provenance record compose_configs leaves at the run root."""
    return TrainRunIdentity(
        git_hash="abc1234-dirty",
        feature_run_id=FEATURE_RUN_ID,
        train_run_id=TRAIN_RUN_ID,
        panel_uri=f"gs://bucket/dev/feature/{FEATURE_RUN_ID}/time_series.parquet",
        calendar_uri=f"gs://bucket/dev/feature/{FEATURE_RUN_ID}/fiscal_calendar.parquet",
    )


def _stage_compose_configs(tmp_path: Path, with_identity: bool = True) -> Path:
    """A compose_configs step directory, optionally with its identity beside it."""
    step_dir = tmp_path / TRAIN_RUN_ID / "compose_configs"
    step_dir.mkdir(parents=True)
    if with_identity:
        (step_dir.parent / "run_identity.json").write_text(
            _identity().model_dump_json(indent=2)
        )
    return step_dir


def test_an_out_dir_not_named_for_its_model_is_refused(tmp_path: Path) -> None:
    """Two models writing one directory would interleave two sidecars silently."""
    step_dir = _stage_compose_configs(tmp_path)
    out_dir = tmp_path / TRAIN_RUN_ID / "backtest" / "some_other_model"

    with pytest.raises(ValueError, match="must be named for its model"):
        backtest_impl(
            panel_path=tmp_path / "absent.parquet",
            calendar_path=tmp_path / "absent.parquet",
            compose_configs_dir=step_dir,
            model_name=MODEL_NAME,
            out_dir=out_dir,
        )

    assert not out_dir.exists()


def test_an_out_dir_outside_the_declared_run_root_is_refused(tmp_path: Path) -> None:
    """Otherwise a sidecar sits under one run while its manifest names another."""
    step_dir = _stage_compose_configs(tmp_path)
    out_dir = tmp_path / "a-different-run" / "backtest" / MODEL_NAME

    with pytest.raises(ValueError, match="must sit under the run root"):
        backtest_impl(
            panel_path=tmp_path / "absent.parquet",
            calendar_path=tmp_path / "absent.parquet",
            compose_configs_dir=step_dir,
            model_name=MODEL_NAME,
            out_dir=out_dir,
        )

    assert not out_dir.exists()


def test_a_missing_run_identity_names_what_should_have_written_it(
    tmp_path: Path,
) -> None:
    """A bare FileNotFoundError would not say which step failed to produce it."""
    step_dir = _stage_compose_configs(tmp_path, with_identity=False)

    with pytest.raises(ValueError, match="No run_identity.json"):
        backtest_impl(
            panel_path=tmp_path / "absent.parquet",
            calendar_path=tmp_path / "absent.parquet",
            compose_configs_dir=step_dir,
            model_name=MODEL_NAME,
            out_dir=tmp_path / TRAIN_RUN_ID / "backtest" / MODEL_NAME,
        )


# ================================================
# BacktestSummary and the manifest
# ================================================


def _summary() -> BacktestSummary:
    """Counts as they arrive off frames, so nunique's int64 is what as_dict sees."""
    return BacktestSummary(
        n_origins=2,
        first_origin="2025-02-23",
        last_origin="2025-03-23",
        n_series=pd.Series(["a", "b", "c"]).nunique(),
        feature_run_id=FEATURE_RUN_ID,
        output_rows={"monthly_series.parquet": pd.Series([1, 2]).size},
    )


def test_summary_as_dict_survives_json_serialisation() -> None:
    """KFP serialises artifact metadata, so a numpy scalar would break a run."""
    json.dumps(_summary().as_dict())


def test_the_manifest_records_the_effective_settings_not_the_defaults() -> None:
    """No sidecar has ever recorded these, because the notebook took the defaults."""
    shipped = cast(
        TrainModelingConfig,
        compose_config(CONFIG_DIR, train_modeling_bindings()).config,
    )
    tuned = shipped.model_copy(
        update={"tiering": shipped.tiering.model_copy(update={"trailing_weeks": 13})}
    )

    manifest = _build_manifest(MODEL_NAME, _summary(), _identity(), tuned)

    assert manifest["config"]["tiering"]["trailing_weeks"] == 13
    assert manifest["lineage"]["train_run_id"] == TRAIN_RUN_ID
    assert manifest["lineage"]["git_hash"] == _identity().git_hash


def test_the_manifest_survives_json_serialisation() -> None:
    """It is written with json.dumps, so a numpy count would fail the whole step."""
    modeling = cast(
        TrainModelingConfig,
        compose_config(CONFIG_DIR, train_modeling_bindings()).config,
    )

    json.dumps(_build_manifest(MODEL_NAME, _summary(), _identity(), modeling))


# ================================================
# The shaped fixture, and the structural assertions
#
# A second fixture, deliberately: the pair above is 16 weeks on a fake model, sized to
# keep the assertion tests fast. This is 80 weeks on the real naive model, shaped so
# the regression golden it will anchor is not a smoke test. Five series with distinct
# trailing means so every tier label is used, one at zero revenue for the lowest tier,
# 52 weeks of history before the first origin, two fiscal months per horizon, and one
# series activating after the first origin so a ragged series set is exercised. Origins
# come from the shipped evaluation_periods, as compose_configs derives them.
# ================================================

_FULL_WEEKS_PER_MONTH = 4
_FULL_N_WEEKS = 80
_FULL_WEEKS = pd.date_range("2024-01-07", periods=_FULL_N_WEEKS, freq="W-SUN")
_FULL_MONTHS = [202401 + i for i in range(12)] + [202501 + i for i in range(8)]
# Weekly level, and the week the series becomes active.
_FULL_SERIES = {
    "high": (1000.0, 0),
    "mid": (300.0, 0),
    "low": (80.0, 0),
    "tiny": (5.0, 0),
    "zero": (0.0, 0),
    "late": (200.0, 66),
}

_SIDECAR_FILENAMES = frozenset(_OUTPUT_FILENAMES.values()) | {
    "fiscal_calendar.parquet",
    "time_series_snapshot.parquet",
    "composed_config.yaml",
    "backtest_manifest.json",
}


@pytest.fixture(scope="module")
def full_calendar() -> pd.DataFrame:
    """Every column the contract declares, so the impl's trim has something to keep."""
    month_index = [week // _FULL_WEEKS_PER_MONTH for week in range(_FULL_N_WEEKS)]
    week_of_month = [week % _FULL_WEEKS_PER_MONTH + 1 for week in range(_FULL_N_WEEKS)]
    return pd.DataFrame(
        {
            "ds": _FULL_WEEKS,
            "fiscal_year_month": [_FULL_MONTHS[m] for m in month_index],
            "fiscal_month": [m % 12 + 1 for m in month_index],
            "fiscal_week_of_month": week_of_month,
            "weeks_in_month": _FULL_WEEKS_PER_MONTH,
            "origin_month_fraction_elapsed": [
                week / _FULL_WEEKS_PER_MONTH for week in week_of_month
            ],
            "count_workdays": 5,
            "fiscal_year": [_FULL_MONTHS[m] // 100 for m in month_index],
            "fiscal_year_week": list(range(1, 49)) + list(range(1, 33)),
        }
    )


@pytest.fixture(scope="module")
def full_panel() -> pd.DataFrame:
    """Deterministic levels on a five-week cycle, so naive is not trivially exact."""
    return pd.DataFrame(
        [
            {
                "unique_id": uid,
                "ds": _FULL_WEEKS[week],
                "y": level * (1 + 0.1 * (week % 5)),
            }
            for uid, (level, first_week) in _FULL_SERIES.items()
            for week in range(first_week, _FULL_N_WEEKS)
        ]
    )


@pytest.fixture(scope="module")
def full_cfg(full_panel: pd.DataFrame, full_calendar: pd.DataFrame) -> BacktestConfig:
    """The naive config compose_configs would emit for this panel."""
    periods = _shipped_modeling().evaluation_periods
    last_complete = last_complete_actual_month(
        max_actual_date=full_panel["ds"].max(), calendar_df=full_calendar
    )
    origins = generate_origins_for_periods(
        start_months=derive_start_months(
            last_complete_actual_month=last_complete,
            n_start_months=periods.n_start_months,
            start_month_step=periods.start_month_step,
            forecast_horizon_months=periods.forecast_horizon_months,
            calendar_df=full_calendar,
        ),
        forecast_horizon_months=periods.forecast_horizon_months,
        calendar_df=full_calendar,
        last_complete_actual_month=last_complete,
    )
    return cast(
        BacktestConfig,
        compose_config(
            CONFIG_DIR,
            train_backtest_bindings(MODEL_NAME),
            {"cross_validation": {"forecast_origins": origins}},
        ).config,
    )


@pytest.fixture(scope="module")
def full_outputs(
    full_cfg: BacktestConfig, full_panel: pd.DataFrame, full_calendar: pd.DataFrame
) -> BacktestOutputs:
    """One real backtest, shared by every structural assertion below."""
    return compute_backtest_outputs(
        cfg=full_cfg,
        modeling=_shipped_modeling(),
        ts_df=full_panel,
        calendar_df=full_calendar,
    )


def _merge_components(outputs: BacktestOutputs) -> pd.DataFrame:
    """Outer, not inner: an inner merge verifies reconstruction on the intersection
    alone, so a components file missing one key while carrying a spurious one passes."""
    return outputs.monthly_series.merge(
        outputs.monthly_forecast_components,
        on=_MONTHLY_SERIES_KEYS,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )


def test_the_fixture_uses_every_tier_and_a_ragged_series_set(
    full_outputs: BacktestOutputs,
) -> None:
    """A fixture that collapses to fewer tiers weakens its golden without saying so."""
    monthly_series = full_outputs.monthly_series

    assert set(monthly_series["tier"].astype(str)) == set(
        _shipped_modeling().tiering.tier_labels
    )
    per_origin = monthly_series.groupby("forecast_origin_date")["unique_id"].nunique()
    assert per_origin.nunique() > 1


def test_components_cover_the_same_keys_as_monthly_series(
    full_outputs: BacktestOutputs,
) -> None:
    """Two files a consumer joins, so a key in one and not the other is a drop."""
    assert (_merge_components(full_outputs)["_merge"] == "both").all()


def test_monthly_forecast_reconstructs_exactly_from_its_components(
    full_outputs: BacktestOutputs,
) -> None:
    """Exact, not approximate: a tolerance would hide a dropped or swapped addend."""
    merged = _merge_components(full_outputs)

    residual = (
        merged["mtd_revenue"]
        + merged["predicted_remaining"]
        - merged["monthly_forecast"]
    )
    assert (residual == 0).all()


def test_forecast_origin_date_carries_the_calendars_unit(
    full_outputs: BacktestOutputs, full_calendar: pd.DataFrame
) -> None:
    """Anchored to the calendar, since all three frames derive theirs from one variable
    and so agree even when all three are wrong together."""
    calendar_unit = full_calendar["ds"].dt.unit

    for frame in (
        full_outputs.monthly_series,
        full_outputs.monthly_forecast_components,
        full_outputs.raw_cv_forecasts,
    ):
        assert frame["forecast_origin_date"].dt.unit == calendar_unit


def test_raw_cv_forecasts_lead_with_the_origin_and_carry_no_nulls(
    full_outputs: BacktestOutputs,
) -> None:
    """Column order is load-bearing: compare tools read the first column by position."""
    raw_cv_forecasts = full_outputs.raw_cv_forecasts

    assert list(raw_cv_forecasts.columns)[0] == "forecast_origin_date"
    assert not raw_cv_forecasts["forecast_origin_date"].isna().any()


def test_monthly_series_keys_are_unique_on_good_data(
    full_outputs: BacktestOutputs,
) -> None:
    """The raising case is crafted above; this pins that real data does not trip it."""
    assert not full_outputs.monthly_series.duplicated(_MONTHLY_SERIES_KEYS).any()


# ================================================
# backtest_impl end to end
# ================================================


def _stage(
    tmp_path: Path,
    full_panel: pd.DataFrame,
    full_calendar: pd.DataFrame,
    full_cfg: BacktestConfig,
) -> dict[str, Any]:
    """Land the inputs and compose_configs outputs, as backtest_impl's arguments.

    Both frames are stamped the way Feature delivers them, with a metadata column
    beyond the lineage one, so the trims have something to drop.
    """
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    panel_path = inputs / "time_series.parquet"
    calendar_path = inputs / "fiscal_calendar.parquet"
    stamps = {
        "feature_run_id": FEATURE_RUN_ID,
        "executed_at": pd.Timestamp("2026-09-17"),
    }
    full_panel.assign(**stamps).to_parquet(panel_path)
    full_calendar.assign(**stamps).to_parquet(calendar_path)

    step_dir = _stage_compose_configs(tmp_path)
    save_config(
        full_cfg.model_dump(by_alias=True, exclude_none=True),
        step_dir / composed_config_filename(MODEL_NAME),
    )
    save_config(
        _shipped_modeling().model_dump(by_alias=True, exclude_none=True),
        step_dir / "modeling.yaml",
    )
    return {
        "panel_path": panel_path,
        "calendar_path": calendar_path,
        "compose_configs_dir": step_dir,
        "model_name": MODEL_NAME,
        "out_dir": tmp_path / TRAIN_RUN_ID / "backtest" / MODEL_NAME,
    }


@pytest.fixture
def staged(
    tmp_path: Path,
    full_panel: pd.DataFrame,
    full_calendar: pd.DataFrame,
    full_cfg: BacktestConfig,
) -> dict[str, Any]:
    """A run root staged but not yet backtested, for the one test that must fail."""
    return _stage(tmp_path, full_panel, full_calendar, full_cfg)


@pytest.fixture(scope="module")
def completed_run(
    tmp_path_factory: pytest.TempPathFactory,
    full_panel: pd.DataFrame,
    full_calendar: pd.DataFrame,
    full_cfg: BacktestConfig,
) -> Path:
    """One real run, shared by the sidecar assertions; returns its out_dir."""
    staged = _stage(
        tmp_path_factory.mktemp("backtest"), full_panel, full_calendar, full_cfg
    )
    backtest_impl(**staged)
    return staged["out_dir"]


def test_a_run_writes_the_whole_sidecar(completed_run: Path) -> None:
    """Asserted in full because the file set is the contract two compare tools read,
    and a missing file is not detectable from inside the run that omitted it."""
    assert {path.name for path in completed_run.iterdir()} == _SIDECAR_FILENAMES


def test_the_manifest_is_absent_when_an_earlier_step_fails(
    staged: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Its presence is the completion marker, so it must not survive a failed run."""

    def _fail(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("manifest build failed")

    monkeypatch.setattr("fcstnyctaxi.core.train.backtest_impl._build_manifest", _fail)

    with pytest.raises(RuntimeError):
        backtest_impl(**staged)

    written = {path.name for path in staged["out_dir"].iterdir()}
    assert written == _SIDECAR_FILENAMES - {"backtest_manifest.json"}


def test_the_manifest_agrees_with_the_files_beside_it(completed_run: Path) -> None:
    """A manifest a reader cannot check against the directory records nothing."""
    manifest = json.loads((completed_run / "backtest_manifest.json").read_text())

    for filename, rows in manifest["output_rows"].items():
        assert len(pd.read_parquet(completed_run / filename)) == rows
    panel_snapshot = pd.read_parquet(completed_run / "time_series_snapshot.parquet")
    assert manifest["n_series"] == panel_snapshot["unique_id"].nunique()
    monthly_series = pd.read_parquet(completed_run / "monthly_series.parquet")
    assert (
        manifest["origins"]["n_origins"]
        == monthly_series["forecast_origin_date"].nunique()
    )


def test_dtypes_survive_the_parquet_round_trip(
    completed_run: Path, full_outputs: BacktestOutputs
) -> None:
    """Against the frames as computed, not against named dtypes: `tier` is categorical
    or object depending on whether every fold binned the same number of tiers, so
    pinning either would fail on data rather than on a defect."""
    calendar_unit = pd.read_parquet(completed_run / "fiscal_calendar.parquet")[
        "ds"
    ].dt.unit

    for field, filename in _OUTPUT_FILENAMES.items():
        written = pd.read_parquet(completed_run / filename)
        assert written.dtypes.equals(getattr(full_outputs, field).dtypes)
        if "forecast_origin_date" in written.columns:
            assert written["forecast_origin_date"].dt.unit == calendar_unit


def test_the_snapshots_are_trimmed_to_the_contract(completed_run: Path) -> None:
    """An untrimmed panel reaches the model and crashes it three layers down."""
    panel_snapshot = pd.read_parquet(completed_run / "time_series_snapshot.parquet")
    calendar_snapshot = pd.read_parquet(completed_run / "fiscal_calendar.parquet")

    assert tuple(panel_snapshot.columns) == PANEL_REQUIRED_COLUMNS
    assert tuple(calendar_snapshot.columns) == CALENDAR_ALLOWED_COLUMNS
