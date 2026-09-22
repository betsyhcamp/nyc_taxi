import dataclasses
import json
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
import yaml
from pandas.testing import assert_frame_equal
from stand_in_model import (
    STAND_IN_EXOG_FEATURES,
    STAND_IN_MODEL_NAME,
    stand_in_config_dir,
    stand_in_fit,
    stand_in_load,
)

from fcstnyctaxi.core.train.compose_configs_impl import compose_configs_impl
from fcstnyctaxi.core.train.final_fit_impl import FinalFitSummary, final_fit_impl
from fcstnyctaxi.lib.config.bindings import train_modeling_bindings
from fcstnyctaxi.lib.config.composition import compose_config, save_config
from fcstnyctaxi.lib.exog import build_exog_frame
from fcstnyctaxi.lib.storage_layout import (
    BUNDLE_MODEL_DIR_NAME,
    SourcedPath,
    composed_config_filename,
)
from fcstnyctaxi.lib.utils import get_project_root_dir
from fcstnyctaxi.schemas.config.train import TrainModelingConfig
from fcstnyctaxi.schemas.run_outputs import TrainingData

CONFIG_DIR = get_project_root_dir() / "config"
FEATURE_RUN_ID = "f-2026-09-21"
TRAIN_RUN_ID = "t-2026-09-21"
MODEL_NAME = STAND_IN_MODEL_NAME
_MARKER = "final_fit_manifest.json"

# ================================================
# Fixtures
#
# conftest.py's panel and calendar, staged by running compose_configs on them, so every
# file the impl reads is one the upstream step really emits. The tree is the stand-in
# model's, so no modeling package is needed; tests then edit that output only where they
# need a state it would not produce.
# ================================================

# Weeks the calendar extends past the staged panel, as a real calendar may.
_HORIZON = 2

_BOXCOX = {
    "name": "boxcox",
    "class": "tsbricks.blocks.transforms.BoxCoxTransform",
    "targets": ["y"],
}


def _save_nothing(model_obj: Any, model_dir: Path) -> None:
    """A save callable that returns without writing."""


def _save_an_empty_file(model_obj: Any, model_dir: Path) -> None:
    """A save callable leaving one zero-byte file, in a directory it did not create."""
    (model_dir / "weights.bin").touch()


def _shipped_modeling() -> TrainModelingConfig:
    """The committed modeling config, including every model's settings."""
    return cast(
        TrainModelingConfig,
        compose_config(CONFIG_DIR, train_modeling_bindings()).config,
    )


def _edit_yaml(path: Path, edit: Callable[[dict], None]) -> None:
    """Rewrite one emitted config through the writer compose_configs uses."""
    document = yaml.safe_load(path.read_text())
    edit(document)
    save_config(document, path)


def _set_model_settings(step_dir: Path, **updates: Any) -> None:
    """Change this model's entry in the emitted modeling.yaml."""

    def _update(document: dict) -> None:
        document["model_settings"][MODEL_NAME].update(updates)

    _edit_yaml(step_dir / "modeling.yaml", _update)


def _stage(
    root: Path,
    panel: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    transforms: list[dict] | None = None,
    config_dir: Path | None = None,
) -> dict[str, Any]:
    """Run compose_configs on stamped frames and return final_fit_impl's arguments.

    Stamped as Feature delivers them: `feature_run_id` on the panel alone, and a
    metadata column on both, so the trims have something to drop.
    """
    inputs = root / "inputs"
    inputs.mkdir()
    executed_at = pd.Timestamp("2026-09-21")
    panel_path = inputs / "time_series.parquet"
    calendar_path = inputs / "fiscal_calendar.parquet"
    panel.assign(feature_run_id=FEATURE_RUN_ID, executed_at=executed_at).to_parquet(
        panel_path
    )
    calendar.assign(executed_at=executed_at).to_parquet(calendar_path)

    step_dir = root / TRAIN_RUN_ID / "compose_configs"
    compose_configs_impl(
        config_dir=config_dir or stand_in_config_dir(root),
        env="dev",
        panel=SourcedPath(path=panel_path, uri="gs://bucket/time_series.parquet"),
        calendar=SourcedPath(
            path=calendar_path, uri="gs://bucket/fiscal_calendar.parquet"
        ),
        expected_feature_run_id=FEATURE_RUN_ID,
        train_run_id=TRAIN_RUN_ID,
        git_hash="abc1234",
        out_dir=step_dir,
    )

    def _with_transforms(document: dict) -> None:
        document["transforms"] = transforms

    if transforms is not None:
        _edit_yaml(step_dir / composed_config_filename(MODEL_NAME), _with_transforms)
    return {
        "panel_path": panel_path,
        "calendar_path": calendar_path,
        "compose_configs_dir": step_dir,
        "model_name": MODEL_NAME,
        "out_dir": root / TRAIN_RUN_ID / "final_fit" / MODEL_NAME,
    }


@pytest.fixture(scope="module")
def panel(full_panel: pd.DataFrame, full_calendar: pd.DataFrame) -> pd.DataFrame:
    """The shared panel, stopping _HORIZON weeks short of the calendar."""
    last_trained = full_calendar["ds"].iloc[-1 - _HORIZON]
    return full_panel[full_panel["ds"] <= last_trained].reset_index(drop=True)


@pytest.fixture
def staged(
    tmp_path: Path, panel: pd.DataFrame, full_calendar: pd.DataFrame
) -> dict[str, Any]:
    """A run root staged but not yet fitted, for the tests that must run the impl."""
    return _stage(tmp_path, panel, full_calendar)


@pytest.fixture(scope="module")
def completed_run(
    tmp_path_factory: pytest.TempPathFactory,
    panel: pd.DataFrame,
    full_calendar: pd.DataFrame,
) -> tuple[dict[str, Any], FinalFitSummary]:
    """One real fit, shared by the assertions on the bundle it leaves."""
    staged = _stage(tmp_path_factory.mktemp("final_fit"), panel, full_calendar)
    return staged, final_fit_impl(**staged)


def test_the_fixture_is_ragged_integer_keyed_and_leaves_a_horizon(
    panel: pd.DataFrame, full_calendar: pd.DataFrame
) -> None:
    """The shapes later tests lean on, so a broken fixture reads as one."""
    assert pd.api.types.is_integer_dtype(panel["unique_id"])
    assert panel.groupby("unique_id")["ds"].min().nunique() > 1
    assert (full_calendar["ds"] > panel["ds"].max()).sum() == _HORIZON
    assert set(full_calendar.columns) - {"ds", *STAND_IN_EXOG_FEATURES}


# ================================================
# The directory guards, which fire before anything is read or written
# ================================================


def test_an_out_dir_not_named_for_its_model_is_refused(staged: dict[str, Any]) -> None:
    """A registered URI names its model, which is what makes the entry legible."""
    out_dir = staged["out_dir"].parent / "some_other_model"

    with pytest.raises(ValueError, match="not named for model"):
        final_fit_impl(**{**staged, "out_dir": out_dir})

    assert not out_dir.exists()


def test_an_out_dir_outside_the_declared_run_root_is_refused(
    staged: dict[str, Any], tmp_path: Path
) -> None:
    """Otherwise a bundle sits under one run while its manifest names another."""
    out_dir = tmp_path / "a-different-run" / "final_fit" / MODEL_NAME

    with pytest.raises(ValueError, match="not under run root"):
        final_fit_impl(**{**staged, "out_dir": out_dir})

    assert not out_dir.exists()


def test_a_missing_run_identity_names_what_should_have_written_it(
    staged: dict[str, Any],
) -> None:
    """A bare FileNotFoundError would not say which step failed to produce it."""
    (staged["compose_configs_dir"].parent / "run_identity.json").unlink()

    with pytest.raises(ValueError, match="which compose_configs writes"):
        final_fit_impl(**staged)


# ================================================
# The guards past the marker delete
#
# Every state below is one compose_configs lets through: the broken calendars pass it,
# and the rest arise after it ran.
# ================================================


def test_a_configured_transform_is_refused_before_the_fit(
    tmp_path: Path, panel: pd.DataFrame, full_calendar: pd.DataFrame
) -> None:
    """Its fitted state would have no file in the bundle and no reader to invert it."""
    staged = _stage(tmp_path, panel, full_calendar, transforms=[_BOXCOX])

    with pytest.raises(ValueError, match="takes no transforms"):
        final_fit_impl(**staged)

    assert not (staged["out_dir"] / BUNDLE_MODEL_DIR_NAME).exists()


def test_a_model_declaring_no_callables_is_refused(
    tmp_path: Path, panel: pd.DataFrame, full_calendar: pd.DataFrame
) -> None:
    """The schema requires a pair of the challenger alone, and a caller may name any."""
    staged = _stage(tmp_path, panel, full_calendar, config_dir=CONFIG_DIR)
    benchmark = _shipped_modeling().model_roles.benchmark
    assert _shipped_modeling().model_settings[benchmark].fit_callable is None

    with pytest.raises(ValueError, match="declares no fit_callable"):
        final_fit_impl(
            **{
                **staged,
                "model_name": benchmark,
                "out_dir": staged["out_dir"].parent / benchmark,
            }
        )


def test_a_panel_from_another_feature_run_is_refused(
    staged: dict[str, Any], panel: pd.DataFrame
) -> None:
    """Wrong bytes at a path compose_configs already read, past its own check."""
    panel.assign(feature_run_id="f-another-run").to_parquet(staged["panel_path"])

    with pytest.raises(ValueError, match="not the declared"):
        final_fit_impl(**staged)


def test_a_repeated_calendar_date_is_refused_before_the_fit(
    tmp_path: Path, panel: pd.DataFrame, full_calendar: pd.DataFrame
) -> None:
    """A repeated date duplicates a training row for every series."""
    repeated = pd.concat([full_calendar, full_calendar.iloc[[30]]], ignore_index=True)
    staged = _stage(tmp_path, panel, repeated)

    with pytest.raises(ValueError, match="calendar repeats ds"):
        final_fit_impl(**staged)

    assert not (staged["out_dir"] / BUNDLE_MODEL_DIR_NAME).exists()


def test_a_panel_week_missing_from_the_calendar_is_refused(
    tmp_path: Path, panel: pd.DataFrame, full_calendar: pd.DataFrame
) -> None:
    """A gap trains on NaN features, which a fit accepts without complaint."""
    gapped = full_calendar.drop(index=30).reset_index(drop=True)
    staged = _stage(tmp_path, panel, gapped)

    with pytest.raises(ValueError, match="no exogenous values"):
        final_fit_impl(**staged)


def test_a_calendar_missing_a_consumed_column_is_refused_by_name(
    tmp_path: Path, panel: pd.DataFrame, full_calendar: pd.DataFrame
) -> None:
    """compose_configs passes it, and the join would raise a bare KeyError instead."""
    consumed = STAND_IN_EXOG_FEATURES[-1]
    staged = _stage(tmp_path, panel, full_calendar.drop(columns=[consumed]))

    with pytest.raises(ValueError, match="calendar is missing required columns"):
        final_fit_impl(**staged)


def test_a_save_that_writes_nothing_is_refused(staged: dict[str, Any]) -> None:
    """The write check exists for a contributor's save that returns without writing."""
    _set_model_settings(
        staged["compose_configs_dir"], save_callable="test_final_fit_impl._save_nothing"
    )

    with pytest.raises(ValueError, match="wrote no file"):
        final_fit_impl(**staged)

    assert not (staged["out_dir"] / _MARKER).exists()


def test_a_zero_byte_file_is_refused(staged: dict[str, Any]) -> None:
    """An empty file passes a directory-not-empty check and registers nothing."""
    _set_model_settings(
        staged["compose_configs_dir"],
        save_callable="test_final_fit_impl._save_an_empty_file",
    )

    with pytest.raises(ValueError, match="zero-byte file"):
        final_fit_impl(**staged)


def test_a_rerun_does_not_count_the_previous_runs_files(staged: dict[str, Any]) -> None:
    """A reused model directory lets an old bundle pass for a save writing nothing."""
    final_fit_impl(**staged)
    marker = staged["out_dir"] / _MARKER
    assert marker.is_file()

    _set_model_settings(
        staged["compose_configs_dir"], save_callable="test_final_fit_impl._save_nothing"
    )
    with pytest.raises(ValueError, match="wrote no file"):
        final_fit_impl(**staged)

    assert not marker.exists()


@pytest.mark.parametrize(
    "failing_collaborator", ["build_exog_frame", "_build_manifest"]
)
def test_a_failed_rerun_leaves_no_completion_marker(
    staged: dict[str, Any], monkeypatch: pytest.MonkeyPatch, failing_collaborator: str
) -> None:
    """Both failure positions: before the fit, and after the bundle is written."""

    def _fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("attempt failed")

    final_fit_impl(**staged)
    marker = staged["out_dir"] / _MARKER
    assert marker.is_file()

    monkeypatch.setattr(
        f"fcstnyctaxi.core.train.final_fit_impl.{failing_collaborator}", _fail
    )
    with pytest.raises(RuntimeError):
        final_fit_impl(**staged)

    assert not marker.exists()


# ================================================
# The bundle
# ================================================


def test_the_bundle_holds_an_in_process_fit_of_its_config(
    completed_run: tuple[dict[str, Any], FinalFitSummary],
    panel: pd.DataFrame,
    full_calendar: pd.DataFrame,
) -> None:
    """The registered model must be this panel, feature set and hyperparameters,
    read back from where the layout puts it."""
    staged, _ = completed_run
    hyperparameters = yaml.safe_load(
        (
            staged["compose_configs_dir"] / composed_config_filename(MODEL_NAME)
        ).read_text()
    )["model"]["hyperparameters"]
    # Self-check: were it the default, a fit handed freq alone would record the same.
    default = stand_in_fit(panel, hyperparameters["freq"])["setting"]
    assert hyperparameters["setting"] != default
    exog_df = build_exog_frame(
        panel, full_calendar, exog_features=STAND_IN_EXOG_FEATURES
    )

    in_process = stand_in_fit(panel, exog_df=exog_df, **hyperparameters)
    loaded = stand_in_load(staged["out_dir"] / BUNDLE_MODEL_DIR_NAME)

    assert_frame_equal(loaded["train_df"], in_process["train_df"])
    assert_frame_equal(loaded["exog_df"], in_process["exog_df"])
    assert (loaded["freq"], loaded["setting"]) == (
        in_process["freq"],
        in_process["setting"],
    )


def test_only_the_configured_features_reach_the_model(
    completed_run: tuple[dict[str, Any], FinalFitSummary],
    full_calendar: pd.DataFrame,
) -> None:
    """A fit adopts every column it is handed, so the impl's selection is the model's,
    and the stamps Feature adds must reach it through neither frame."""
    staged, _ = completed_run
    handed = stand_in_load(staged["out_dir"] / BUNDLE_MODEL_DIR_NAME)
    columns = set(handed["train_df"].columns) | set(handed["exog_df"].columns)
    unselected = set(full_calendar.columns) - set(STAND_IN_EXOG_FEATURES) - {"ds"}

    assert set(STAND_IN_EXOG_FEATURES) <= set(handed["exog_df"].columns)
    assert not columns & (unselected | {"feature_run_id", "executed_at"})


def test_the_manifest_agrees_with_the_panel_it_describes(
    completed_run: tuple[dict[str, Any], FinalFitSummary], panel: pd.DataFrame
) -> None:
    """A reader holding only the bundle checks these against the panel it names."""
    staged, summary = completed_run
    manifest = json.loads((staged["out_dir"] / _MARKER).read_text())
    training_data = manifest["training_data"]

    # fromisoformat refuses the "2025-06-29 00:00:00" that str() on a Timestamp gives.
    assert date.fromisoformat(training_data["train_end_ds"]) == panel["ds"].max().date()
    assert training_data["n_obs"] == len(panel)
    assert training_data["n_series"] == panel["unique_id"].nunique()
    assert training_data == {key: summary.as_dict()[key] for key in training_data}


def test_the_manifest_training_data_passes_through_the_run_outputs_schema(
    completed_run: tuple[dict[str, Any], FinalFitSummary],
) -> None:
    """register_model copies this block into run_outputs.json through TrainingData,
    so the schema must hand back exactly what the bundle wrote."""
    staged, _ = completed_run
    manifest = json.loads((staged["out_dir"] / _MARKER).read_text())
    training_data = manifest["training_data"]

    copied = TrainingData.model_validate(training_data).model_dump(mode="json")

    assert copied == training_data


def test_the_manifest_names_what_a_reader_needs_to_load_and_predict(
    completed_run: tuple[dict[str, Any], FinalFitSummary],
) -> None:
    """The save callable says which loader pairs with the bytes, and exog_features
    which columns a predictor must assemble."""
    staged, _ = completed_run
    manifest = json.loads((staged["out_dir"] / _MARKER).read_text())
    settings = TrainModelingConfig.model_validate(
        yaml.safe_load((staged["compose_configs_dir"] / "modeling.yaml").read_text())
    ).model_settings[MODEL_NAME]

    assert manifest["model_name"] == MODEL_NAME
    assert manifest["config"]["save_callable"] == settings.save_callable
    assert manifest["config"]["exog_features"] == settings.exog_features
    assert manifest["lineage"]["train_run_id"] == TRAIN_RUN_ID


def test_the_summary_carries_the_run_ids_it_discovered(
    completed_run: tuple[dict[str, Any], FinalFitSummary],
) -> None:
    """Neither id is a parameter, and a wrapper never opens the bundle to find them."""
    _, summary = completed_run

    assert summary.train_run_id == TRAIN_RUN_ID
    assert summary.feature_run_id == FEATURE_RUN_ID


def test_the_bundle_carries_the_config_it_was_fitted_under(
    completed_run: tuple[dict[str, Any], FinalFitSummary],
) -> None:
    """Byte for byte: a re-dump reimplements save_config; another model's is wrong."""
    staged, _ = completed_run
    source = staged["compose_configs_dir"] / composed_config_filename(MODEL_NAME)

    assert (
        staged["out_dir"] / "composed_config.yaml"
    ).read_bytes() == source.read_bytes()


def test_summary_as_dict_survives_json_serialization(panel: pd.DataFrame) -> None:
    """Built from values as they come off a frame, a Timestamp and numpy counts, which
    a dataclass accepts into str and int fields and json.dumps refuses."""
    summary = FinalFitSummary(
        train_end_ds=panel["ds"].max(),  # type: ignore[arg-type]
        n_series=panel["unique_id"].drop_duplicates().count(),
        n_obs=panel["y"].count(),
        train_run_id=TRAIN_RUN_ID,
        feature_run_id=FEATURE_RUN_ID,
    )
    with pytest.raises(TypeError):
        json.dumps(dataclasses.asdict(summary))

    json.dumps(summary.as_dict(), allow_nan=False)
