import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from pydantic import BaseModel, ValidationError
from tsbricks.backtesting.schema import BacktestConfig

from fcstnyctaxi.core.train.compose_configs_impl import (
    ComposeConfigsSummary,
    SourcedPath,
    compose_configs_impl,
    compose_train_static_configs,
)
from fcstnyctaxi.lib.config.composition import RUNTIME_SOURCE
from fcstnyctaxi.lib.utils import get_project_root_dir
from fcstnyctaxi.schemas.config.environment import EnvironmentConfig
from fcstnyctaxi.schemas.config.train import TrainInfraConfig, TrainModelingConfig
from fcstnyctaxi.schemas.run_identity import TrainRunIdentity

CONFIG_DIR = get_project_root_dir() / "config"

FEATURE_RUN_ID = "f-2026-09-08"
TRAIN_RUN_ID = "t-2026-09-08"
GIT_HASH = "abc1234"

_WEEKS_PER_MONTH = 4
_N_MONTHS = 12


def _calendar_frame(feature_run_id: str | None = FEATURE_RUN_ID) -> pd.DataFrame:
    """A fiscal calendar carrying every contract column, not just the three read."""
    n_weeks = _N_MONTHS * _WEEKS_PER_MONTH
    week_of_month = [i % _WEEKS_PER_MONTH + 1 for i in range(n_weeks)]
    month = [i // _WEEKS_PER_MONTH + 1 for i in range(n_weeks)]

    return pd.DataFrame(
        {
            "ds": pd.date_range("2025-01-05", periods=n_weeks, freq="W-SUN"),
            "fiscal_year_month": [202500 + m for m in month],
            "fiscal_month": month,
            "fiscal_week_of_month": week_of_month,
            "weeks_in_month": _WEEKS_PER_MONTH,
            "origin_month_fraction_elapsed": [
                w / _WEEKS_PER_MONTH for w in week_of_month
            ],
            "count_workdays": 5,
            "feature_run_id": pd.array([feature_run_id] * n_weeks, dtype="string"),
        }
    )


def _panel_frame(feature_run_id: str | None = FEATURE_RUN_ID) -> pd.DataFrame:
    """Two series of weekly actuals spanning the whole calendar."""
    weeks = pd.date_range(
        "2025-01-05", periods=_N_MONTHS * _WEEKS_PER_MONTH, freq="W-SUN"
    )
    frame = pd.DataFrame(
        {"ds": weeks.repeat(2), "unique_id": [1, 2] * len(weeks)}
    ).assign(y=lambda df: range(len(df)))

    return frame.assign(
        feature_run_id=pd.array([feature_run_id] * len(frame), dtype="string")
    )


def _write_inputs(
    tmp_path: Path, panel_df: pd.DataFrame, calendar_df: pd.DataFrame
) -> tuple[SourcedPath, SourcedPath]:
    """Land two frames as parquet and pair each with a plausible durable URI."""
    panel_path = tmp_path / "time_series.parquet"
    calendar_path = tmp_path / "fiscal_calendar.parquet"
    panel_df.to_parquet(panel_path)
    calendar_df.to_parquet(calendar_path)

    prefix = f"gs://bucket/dev/feature/{FEATURE_RUN_ID}/data_prep"
    return (
        SourcedPath(path=panel_path, uri=f"{prefix}/time_series.parquet"),
        SourcedPath(path=calendar_path, uri=f"{prefix}/fiscal_calendar.parquet"),
    )


def _run_dir(tmp_path: Path) -> Path:
    """This run's root, where run_identity.json lands beside the step directory."""
    return tmp_path / TRAIN_RUN_ID


def _step_dir(tmp_path: Path) -> Path:
    """The step directory under this run's root, the shape the impl requires.

    Not a flat `tmp_path / "out"`: the impl writes run_identity.json to the parent,
    so a directory with no run-id level would put it wherever tmp_path happens to be.
    """
    return _run_dir(tmp_path) / "compose_configs"


def _run(
    tmp_path: Path,
    panel_df: pd.DataFrame | None = None,
    calendar_df: pd.DataFrame | None = None,
    expected_feature_run_id: str = FEATURE_RUN_ID,
) -> ComposeConfigsSummary:
    """Compose one run into its step directory, defaulting to a consistent pair."""
    panel, calendar = _write_inputs(
        tmp_path,
        _panel_frame() if panel_df is None else panel_df,
        _calendar_frame() if calendar_df is None else calendar_df,
    )
    return compose_configs_impl(
        config_dir=CONFIG_DIR,
        env="dev",
        panel=panel,
        calendar=calendar,
        expected_feature_run_id=expected_feature_run_id,
        train_run_id=TRAIN_RUN_ID,
        git_hash=GIT_HASH,
        out_dir=_step_dir(tmp_path),
    )


@pytest.fixture
def composed(tmp_path: Path) -> tuple[ComposeConfigsSummary, Path]:
    """One successful run, returning its summary and its step directory."""
    return _run(tmp_path), _step_dir(tmp_path)


# ================================================
# compose_train_static_configs
# ================================================


def test_static_configs_come_back_environment_infra_modeling() -> None:
    """The order is fixed by the docstring, so each element is checked by type."""
    environment, infra, modeling = compose_train_static_configs(CONFIG_DIR, "dev")

    assert isinstance(environment.config, EnvironmentConfig)
    assert isinstance(infra.config, TrainInfraConfig)
    assert isinstance(modeling.config, TrainModelingConfig)


def test_static_configs_reject_an_env_with_no_file() -> None:
    """The selector is guarded here so all three callers share one message."""
    with pytest.raises(ValueError, match="Unknown env 'prod'"):
        compose_train_static_configs(CONFIG_DIR, "prod")


# ================================================
# compose_configs_impl — the emitted artifacts
# ================================================


@pytest.mark.parametrize(
    ("filename", "destination"),
    [
        ("environment.yaml", EnvironmentConfig),
        ("infra.yaml", TrainInfraConfig),
        ("modeling.yaml", TrainModelingConfig),
        ("composed_config_naive.yaml", BacktestConfig),
        ("composed_config_xgboost.yaml", BacktestConfig),
    ],
)
def test_every_emitted_config_revalidates_when_loaded_back(
    composed: tuple[ComposeConfigsSummary, Path],
    filename: str,
    destination: type[BaseModel],
) -> None:
    """An emitted config a consumer cannot re-validate is not a usable record."""
    _, out_dir = composed

    destination.model_validate(yaml.safe_load((out_dir / filename).read_text()))


def test_run_identity_reloads_and_stamps_what_was_read(
    composed: tuple[ComposeConfigsSummary, Path],
) -> None:
    """The one emitted file with four downstream readers, so it is pinned hardest."""
    _, out_dir = composed

    identity = TrainRunIdentity(
        **json.loads((out_dir.parent / "run_identity.json").read_text())
    )

    assert identity.feature_run_id == FEATURE_RUN_ID
    assert identity.train_run_id == TRAIN_RUN_ID
    assert identity.git_hash == GIT_HASH
    assert identity.panel_uri.endswith("time_series.parquet")
    assert identity.calendar_uri.endswith("fiscal_calendar.parquet")


def test_run_identity_lands_at_the_run_root_not_in_the_step_directory(
    composed: tuple[ComposeConfigsSummary, Path],
) -> None:
    """Test the placement a reader outside this pipeline depends on.

    A restart guard knows the bucket, env, slice and run id and nothing else, so
    a file it must find cannot sit behind a step name.
    """
    _, out_dir = composed

    assert (out_dir.parent / "run_identity.json").is_file()
    assert not (out_dir / "run_identity.json").exists()


def test_a_step_directory_outside_its_run_root_is_refused(tmp_path: Path) -> None:
    """Test that the impl refuses to write the identity somewhere unintended.

    The run root is derived from out_dir rather than passed, so without this a
    caller handing over a flat directory silently scatters run_identity.json.
    """
    panel, calendar = _write_inputs(tmp_path, _panel_frame(), _calendar_frame())

    with pytest.raises(ValueError, match="must be a step directory under the run root"):
        compose_configs_impl(
            config_dir=CONFIG_DIR,
            env="dev",
            panel=panel,
            calendar=calendar,
            expected_feature_run_id=FEATURE_RUN_ID,
            train_run_id=TRAIN_RUN_ID,
            git_hash=GIT_HASH,
            out_dir=tmp_path / "flat",
        )

    assert not (tmp_path / "flat").exists()
    assert not (tmp_path / "run_identity.json").exists()


def test_manifest_hashes_a_shared_fragment_once_but_keeps_per_destination_order(
    composed: tuple[ComposeConfigsSummary, Path],
) -> None:
    """base/data.yaml feeds every model; order decides the winner, so lists keep it."""
    _, out_dir = composed
    manifest = json.loads((out_dir / "manifest.json").read_text())

    assert manifest["env"] == "dev"
    assert manifest["config_files"]["base/data.yaml"].startswith("sha256:")
    assert manifest["destinations"]["BacktestConfig:naive"]["config_files"] == [
        "base/data.yaml",
        "train/backtest.yaml",
        "train/models/naive.yaml",
        RUNTIME_SOURCE,
    ]


def test_manifest_records_no_hash_for_the_runtime_source(
    composed: tuple[ComposeConfigsSummary, Path],
) -> None:
    """The runtime override is a source but not a file, so it cannot be hashed."""
    _, out_dir = composed
    manifest = json.loads((out_dir / "manifest.json").read_text())

    assert RUNTIME_SOURCE not in manifest["config_files"]


@pytest.mark.parametrize("model_name", ["naive", "xgboost"])
def test_the_derived_origins_reach_every_model_config(
    composed: tuple[ComposeConfigsSummary, Path], model_name: str
) -> None:
    """Origins are the one runtime override; a model config without them is unusable."""
    summary, out_dir = composed
    emitted = yaml.safe_load(
        (out_dir / f"composed_config_{model_name}.yaml").read_text()
    )

    origins = emitted["cross_validation"]["forecast_origins"]
    assert len(origins) == summary.n_origins
    assert origins[0]["origin"] == summary.first_origin
    assert origins[-1]["origin"] == summary.last_origin


# ================================================
# ComposeConfigsSummary
# ================================================


def test_summary_survives_json_serialisation(
    composed: tuple[ComposeConfigsSummary, Path],
) -> None:
    """KFP serialises artifact metadata to JSON, so a numpy scalar would break a run."""
    summary, _ = composed

    json.dumps(summary.as_dict())


def test_summary_describes_the_run_the_caller_could_not_open(
    composed: tuple[ComposeConfigsSummary, Path],
) -> None:
    """Every field here is derived by walking a frame the wrapper never opens."""
    summary, _ = composed

    assert summary.model_names == ["naive", "xgboost"]
    assert summary.n_origins > 0
    assert summary.first_origin <= summary.last_origin
    assert summary.start_months == sorted(summary.start_months)
    assert summary.last_complete_actual_month >= max(summary.start_months)


# ================================================
# compose_configs_impl — what it requires of its inputs
# ================================================


def test_frames_carrying_only_what_this_step_reads_still_compose(
    tmp_path: Path,
) -> None:
    """A Feature change to columns this step never reads must not break composition."""
    minimal_calendar = _calendar_frame()[
        ["ds", "fiscal_year_month", "feature_run_id"]
    ].assign(some_new_exogenous_feature=1.0)

    summary = _run(tmp_path, calendar_df=minimal_calendar)

    assert summary.n_origins > 0


def test_a_panel_mixing_two_feature_runs_raises(tmp_path: Path) -> None:
    """Two ids in one frame means every table stamped from it claims a false run."""
    panel = _panel_frame()
    panel.loc[panel.index[:10], "feature_run_id"] = "another-run"

    with pytest.raises(ValueError, match="panel mixes Feature runs"):
        _run(tmp_path, panel_df=panel)


def test_a_panel_and_calendar_from_different_runs_raise(tmp_path: Path) -> None:
    """Origins from one run's calendar and actuals from another's is wrong numbers."""
    with pytest.raises(ValueError, match="!= calendar"):
        _run(tmp_path, calendar_df=_calendar_frame(feature_run_id="another-run"))


@pytest.mark.parametrize("null_rows", [slice(None), slice(0, 10)], ids=["all", "some"])
def test_a_null_feature_run_id_raises(tmp_path: Path, null_rows: slice) -> None:
    """nunique() skips nulls, so a partly-null column would otherwise pass."""
    panel = _panel_frame()
    panel.loc[panel.index[null_rows], "feature_run_id"] = None

    with pytest.raises(ValueError, match="null feature_run_id"):
        _run(tmp_path, panel_df=panel)


def test_a_consistent_pair_from_the_wrong_run_raises(tmp_path: Path) -> None:
    """The check that makes pasting explicit URIs safe: right shape, wrong run."""
    with pytest.raises(ValueError, match="not the declared"):
        _run(tmp_path, expected_feature_run_id="a-different-run")


def test_a_frame_without_the_lineage_column_names_which_frame(
    tmp_path: Path,
) -> None:
    """Two frames are read here, so a bare KeyError would not say which one failed."""
    calendar = _calendar_frame().drop(columns=["feature_run_id"])

    with pytest.raises(ValueError, match="calendar is missing required columns"):
        _run(tmp_path, calendar_df=calendar)


def test_nothing_is_written_when_the_lineage_check_fails(tmp_path: Path) -> None:
    """A half-written prefix is indistinguishable from a complete one to a reader."""
    with pytest.raises(ValueError):
        _run(tmp_path, expected_feature_run_id="a-different-run")

    assert not _run_dir(tmp_path).exists()


def test_nothing_is_written_when_an_identity_field_is_invalid(tmp_path: Path) -> None:
    """Identity is validated before the first write, not after six of seven land.

    A local path where a URI belongs is the documented slip: the runner holds
    both, and stamping the staged path would record the wrong source silently.
    """
    panel, calendar = _write_inputs(tmp_path, _panel_frame(), _calendar_frame())

    with pytest.raises(ValidationError):
        compose_configs_impl(
            config_dir=CONFIG_DIR,
            env="dev",
            panel=SourcedPath(path=panel.path, uri="/tmp/staged/time_series.parquet"),
            calendar=calendar,
            expected_feature_run_id=FEATURE_RUN_ID,
            train_run_id=TRAIN_RUN_ID,
            git_hash=GIT_HASH,
            out_dir=_step_dir(tmp_path),
        )

    assert not _run_dir(tmp_path).exists()


def test_one_artifact_passed_as_both_panel_and_calendar_raises(
    tmp_path: Path,
) -> None:
    """The calendar in both slots would write seven artifacts past the last actual."""
    _, calendar = _write_inputs(tmp_path, _panel_frame(), _calendar_frame())

    with pytest.raises(ValueError, match="same file"):
        compose_configs_impl(
            config_dir=CONFIG_DIR,
            env="dev",
            panel=calendar,
            calendar=calendar,
            expected_feature_run_id=FEATURE_RUN_ID,
            train_run_id=TRAIN_RUN_ID,
            git_hash=GIT_HASH,
            out_dir=_step_dir(tmp_path),
        )

    assert not _run_dir(tmp_path).exists()
