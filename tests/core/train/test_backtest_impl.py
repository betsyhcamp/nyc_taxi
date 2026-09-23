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
    _origin_label,
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
    ADDITIONAL_EXOG_REQUIRED_COLUMNS,
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

_exog_columns_seen: list[list[str]] = []
"""The columns each fake-model call was handed, so a test can assert the trim."""


@pytest.fixture(autouse=True)
def _reset_fit_calls() -> None:
    """Module state, so every test starts from empty."""
    _fit_calls.clear()
    _exog_columns_seen.clear()


def _fake_model_callable(
    train_df: pd.DataFrame, horizon: int, future_x_df: pd.DataFrame, **kwargs: Any
) -> pd.DataFrame:
    """Repeat each series' last observed value, and record that a fold ran."""
    _fit_calls.append(horizon)
    _exog_columns_seen.append(list(future_x_df.columns))

    last_ds = train_df["ds"].max()
    # drop_duplicates before head: future_x_df is keyed (unique_id, ds), so taking
    # rows would return one date repeated rather than `horizon` distinct dates.
    future = (
        future_x_df.loc[future_x_df["ds"] > last_ds, "ds"]
        .drop_duplicates()
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

# The 16-week calendar carries only what the fold loop reads, so there is nothing
# for a model to consume; the 80-week fixture below declares a real feature set.
_NO_EXOG: tuple[str, ...] = ()


# ================================================
# compute_backtest_outputs: the loop runs
# ================================================


def test_the_fold_loop_produces_one_origin_per_configured_origin(
    ts_df: pd.DataFrame, calendar_df: pd.DataFrame, modeling: TrainModelingConfig
) -> None:
    """The fixture's own check: without it a broken fixture reads as a broken check."""
    outputs = compute_backtest_outputs(
        cfg=_cfg(_TWO_ORIGINS),
        modeling=modeling,
        ts_df=ts_df,
        calendar_df=calendar_df,
        exog_features=_NO_EXOG,
    )

    assert len(_fit_calls) == len(_TWO_ORIGINS)
    assert outputs.raw_cv_forecasts["forecast_origin_date"].nunique() == len(
        _TWO_ORIGINS
    )
    assert not outputs.monthly_series.empty


def test_only_the_configured_columns_reach_the_model(
    ts_df: pd.DataFrame, calendar_df: pd.DataFrame, modeling: TrainModelingConfig
) -> None:
    """The impl decides the feature set now, so a calendar column outside it must
    not reach the model, where an extra column is adopted as a feature at fit."""
    selected = ("fiscal_year_month",)

    compute_backtest_outputs(
        cfg=_cfg([(7, 2)]),
        modeling=modeling,
        ts_df=ts_df,
        calendar_df=calendar_df,
        exog_features=selected,
    )

    assert _exog_columns_seen == [["unique_id", "ds", *selected]]


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
            exog_features=_NO_EXOG,
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
            exog_features=_NO_EXOG,
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
            exog_features=_NO_EXOG,
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
            exog_features=_NO_EXOG,
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
        additional_exog_uri=(
            f"gs://bucket/dev/feature/{FEATURE_RUN_ID}/exogenous_features.parquet"
        ),
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
            additional_exog_path=tmp_path / "absent.parquet",
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
            additional_exog_path=tmp_path / "absent.parquet",
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
            additional_exog_path=tmp_path / "absent.parquet",
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
        train_run_id=TRAIN_RUN_ID,
        feature_run_id=FEATURE_RUN_ID,
        output_rows={"monthly_series.parquet": pd.Series([1, 2]).size},
    )


def test_summary_as_dict_survives_json_serialisation() -> None:
    """KFP serializes artifact metadata, so a numpy scalar would break a run."""
    json.dumps(_summary().as_dict())


@pytest.mark.parametrize(
    ("origin", "expected"),
    [
        (pd.Timestamp("2025-4-27"), "2025-04-27"),
        (pd.Timestamp("2025-05-04T00:00:00"), "2025-05-04"),
        (5, "5"),
    ],
)
def test_an_origin_is_labelled_the_way_compose_configs_spells_it(
    origin: Any, expected: str
) -> None:
    """The schema admits mixed spellings with only a warning, and freq=1 admits ints."""
    assert _origin_label(origin) == expected


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
    # Whole-object, as register_model compares the bundle's block: every field of
    # the identity is echoed, so a slot left out of the block is a silent drift.
    assert manifest["lineage"] == _identity().model_dump()


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
# keep the assertion tests fast. This is conftest.py's 80 weeks on the real naive model,
# shaped so the regression golden it will anchor is not a smoke test, with two fiscal
# months per horizon. Origins come from the shipped evaluation_periods, as
# compose_configs derives them.
# ================================================

# Literal, not derived from the writer's own map: deriving it moves the expectation
# with any deletion from that map, so dropping a file from the sidecar stays green.
_SIDECAR_FILENAMES = frozenset(
    {
        "monthly_series.parquet",
        "monthly_forecast_components.parquet",
        "metrics.parquet",
        "raw_cv_forecasts.parquet",
        "fiscal_calendar.parquet",
        "time_series_snapshot.parquet",
        "additional_exog.parquet",
        "composed_config.yaml",
        "backtest_manifest.json",
    }
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
        exog_features=tuple(
            _shipped_modeling().model_settings[MODEL_NAME].exog_features
        ),
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


def _raise_attempt_failed(*args: Any, **kwargs: Any) -> Any:
    """Stand in for a collaborator that fails partway through a run."""
    raise RuntimeError("attempt failed")


def _additional_exog(panel: pd.DataFrame) -> pd.DataFrame:
    """The exogenous features file, on the keys Feature publishes it against."""
    return panel[["unique_id", "ds"]].assign(
        holiday_days_in_week=0, week_sin=0.0, week_cos=1.0
    )


def _stage(
    tmp_path: Path,
    full_panel: pd.DataFrame,
    full_calendar: pd.DataFrame,
    full_cfg: BacktestConfig,
) -> dict[str, Any]:
    """Land the inputs and compose_configs outputs, as backtest_impl's arguments.

    Stamped as Feature delivers them: `feature_run_id` on the panel alone, and a
    metadata column on all three, so the trims have something to drop.
    """
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    panel_path = inputs / "time_series.parquet"
    calendar_path = inputs / "fiscal_calendar.parquet"
    additional_exog_path = inputs / "exogenous_features.parquet"
    executed_at = pd.Timestamp("2026-09-17")
    full_panel.assign(
        feature_run_id=FEATURE_RUN_ID, executed_at=executed_at
    ).to_parquet(panel_path)
    full_calendar.assign(executed_at=executed_at).to_parquet(calendar_path)
    _additional_exog(full_panel).assign(executed_at=executed_at).to_parquet(
        additional_exog_path
    )

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
        "additional_exog_path": additional_exog_path,
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


def test_the_summary_carries_the_run_ids_it_discovered(
    staged: dict[str, Any],
) -> None:
    """Neither id is a parameter: both are read off run_identity.json, so a wrapper
    that never opens the sidecar has the return value as its only source."""
    summary = backtest_impl(**staged)

    assert summary.train_run_id == TRAIN_RUN_ID
    assert summary.feature_run_id == FEATURE_RUN_ID


def test_a_panel_from_another_feature_run_is_refused(
    staged: dict[str, Any], full_panel: pd.DataFrame
) -> None:
    """Wrong bytes at a path compose_configs already read, past its own check."""
    full_panel.assign(feature_run_id="f-another-run").to_parquet(staged["panel_path"])

    with pytest.raises(ValueError, match="not the declared"):
        backtest_impl(**staged)


@pytest.mark.parametrize(
    "failing_collaborator", ["compute_backtest_outputs", "_build_manifest"]
)
def test_a_failed_rerun_leaves_no_completion_marker(
    staged: dict[str, Any], monkeypatch: pytest.MonkeyPatch, failing_collaborator: str
) -> None:
    """Both failure positions: before the writes, and partway through them."""
    backtest_impl(**staged)
    assert (staged["out_dir"] / "backtest_manifest.json").is_file()

    monkeypatch.setattr(
        f"fcstnyctaxi.core.train.backtest_impl.{failing_collaborator}",
        _raise_attempt_failed,
    )
    with pytest.raises(RuntimeError):
        backtest_impl(**staged)

    assert not (staged["out_dir"] / "backtest_manifest.json").exists()


def test_every_sidecar_file_lands_before_the_completion_marker(
    staged: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A marker over a half-written sidecar reads as a finished one."""
    monkeypatch.setattr(
        "fcstnyctaxi.core.train.backtest_impl._build_manifest", _raise_attempt_failed
    )
    with pytest.raises(RuntimeError):
        backtest_impl(**staged)

    # _build_manifest runs after every file write and before the marker write, so
    # this partitions the writes exactly where the marker rule sits.
    written = {path.name for path in staged["out_dir"].iterdir()}
    assert written == _SIDECAR_FILENAMES - {"backtest_manifest.json"}


def test_an_exogenous_path_naming_another_artifact_is_refused_by_frame_name(
    staged: dict[str, Any],
) -> None:
    """Three staged paths transpose silently; the trim is what names which frame."""
    staged["additional_exog_path"] = staged["calendar_path"]

    with pytest.raises(
        ValueError, match="additional exogenous is missing required columns"
    ):
        backtest_impl(**staged)


def test_the_impl_passes_the_models_own_configured_features(
    staged: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wiring that read no config at all would pass an empty tuple, which the
    shipped entry for this model also is, so the staged config declares a column."""
    document = _shipped_modeling().model_dump(by_alias=True, exclude_none=True)
    document["model_settings"][MODEL_NAME]["exog_features"] = ["count_workdays"]
    save_config(document, staged["compose_configs_dir"] / "modeling.yaml")
    seen: dict[str, Any] = {}

    def _capture(**kwargs: Any) -> BacktestOutputs:
        seen["exog_features"] = kwargs["exog_features"]
        return compute_backtest_outputs(**kwargs)

    monkeypatch.setattr(
        "fcstnyctaxi.core.train.backtest_impl.compute_backtest_outputs", _capture
    )
    backtest_impl(**staged)

    assert seen["exog_features"] == ("count_workdays",)


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
    exog_snapshot = pd.read_parquet(completed_run / "additional_exog.parquet")

    assert tuple(panel_snapshot.columns) == PANEL_REQUIRED_COLUMNS
    assert tuple(calendar_snapshot.columns) == CALENDAR_ALLOWED_COLUMNS
    assert tuple(exog_snapshot.columns) == ADDITIONAL_EXOG_REQUIRED_COLUMNS
