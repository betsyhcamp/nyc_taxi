"""Tests for the local Training runner's ordering guarantees.

`main()` is driven end to end rather than reshaped for injection. PROJECT_ROOT
already redirects `get_project_root_dir`, so a copied config tree under
`tmp_path` exercises the real function, and only `_require_git_hash` is patched
since git cannot answer for a directory that is not a repository.
"""

import shutil
import sys
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from fcstnyctaxi.core.train.compose_configs_impl import ComposeConfigsSummary
from fcstnyctaxi.lib.storage_layout import resolve_run_prefix
from fcstnyctaxi.lib.utils import get_project_root_dir
from fcstnyctaxi.pipelines import local_train_pipeline

ENV = "dev"
RUN_ID = "t-20260913T000000000000Z"
FEATURE_RUN_ID = "f-20260913T000000000000Z"
PREVIOUS_OUTPUT = "the previous attempt's output"


@pytest.fixture
def broken_config_root(tmp_path: Path) -> Path:
    """A project root whose `train/modeling.yaml` no longer satisfies its schema.

    Copied from the real tree and then corrupted, so every other destination
    still composes: the preflight has to be what raises, not a missing file
    somewhere upstream. An undeclared key rather than a wrong type, because it
    names the fragment in the message without depending on any field's type.
    """
    root = tmp_path / "project"
    shutil.copytree(get_project_root_dir() / "config", root / "config")
    modeling = root / "config" / "train" / "modeling.yaml"
    modeling.write_text(f"{modeling.read_text()}an_undeclared_key: 1\n")
    return root


def test_a_malformed_train_config_raises_before_out_dir_is_cleared(
    broken_config_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """Test that the preflight raises while the previous attempt's output survives.

    `resolve_run_prefix` composes only EnvironmentConfig, so without the explicit
    preflight the Training configs are first read by the impl, two steps after
    the rmtree has already destroyed the directory this seeds.
    """
    monkeypatch.setenv("PROJECT_ROOT", str(broken_config_root))
    mocker.patch.object(
        local_train_pipeline, "_require_git_hash", return_value="abc1234"
    )
    rmtree = mocker.spy(local_train_pipeline.shutil, "rmtree")

    scratch = tmp_path / "scratch"
    run_prefix = resolve_run_prefix(broken_config_root / "config", ENV, "train", RUN_ID)
    out_dir = local_train_pipeline._mirror_path(
        f"{run_prefix}compose_configs/", scratch
    )
    out_dir.mkdir(parents=True)
    (out_dir / "run_identity.json").write_text(PREVIOUS_OUTPUT)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "local_train_pipeline",
            "--env",
            ENV,
            "--feature-run-id",
            FEATURE_RUN_ID,
            "--panel-uri",
            f"gs://bucket/{ENV}/feature/{FEATURE_RUN_ID}/data_prep/time_series.parquet",
            "--calendar-uri",
            f"gs://bucket/{ENV}/feature/{FEATURE_RUN_ID}/data_prep/fiscal_calendar.parquet",
            "--run-id",
            RUN_ID,
            "--scratch-dir",
            str(scratch),
        ],
    )

    with pytest.raises(ValueError, match="train/modeling.yaml"):
        local_train_pipeline.main()

    assert rmtree.call_count == 0
    assert (out_dir / "run_identity.json").read_text() == PREVIOUS_OUTPUT


def test_the_identity_is_published_before_the_step_and_manifest_marks_it_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """Test the publish order and which file marks the step complete.

    run_identity.json sits outside `out_dir`, so `sync_to_gcs` no longer carries it
    and the step needs a different marker. Published second, a failed step sync
    would leave the run with no record of what it read.
    """
    root = tmp_path / "project"
    shutil.copytree(get_project_root_dir() / "config", root / "config")
    monkeypatch.setenv("PROJECT_ROOT", str(root))
    mocker.patch.object(
        local_train_pipeline, "_require_git_hash", return_value="abc1234"
    )
    mocker.patch.object(
        local_train_pipeline, "download_from_gcs", side_effect=lambda uri, d: d / "f"
    )
    mocker.patch.object(
        local_train_pipeline,
        "compose_configs_impl",
        return_value=ComposeConfigsSummary(
            n_origins=1,
            first_origin="2025-05-18",
            last_origin="2025-05-18",
            last_complete_actual_month=202505,
            start_months=[202505],
            model_names=["naive"],
        ),
    )
    # One manager, so the assertion is about order between the two calls rather
    # than each in isolation.
    publishes = mocker.MagicMock()
    publishes.attach_mock(
        mocker.patch.object(local_train_pipeline, "upload_to_gcs"), "upload"
    )
    publishes.attach_mock(
        mocker.patch.object(local_train_pipeline, "sync_to_gcs", return_value=(7, 0)),
        "sync",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "local_train_pipeline",
            "--env",
            ENV,
            "--feature-run-id",
            FEATURE_RUN_ID,
            "--panel-uri",
            f"gs://bucket/{ENV}/feature/{FEATURE_RUN_ID}/data_prep/time_series.parquet",
            "--calendar-uri",
            f"gs://bucket/{ENV}/feature/{FEATURE_RUN_ID}/data_prep/fiscal_calendar.parquet",
            "--run-id",
            RUN_ID,
            "--scratch-dir",
            str(tmp_path / "scratch"),
        ],
    )

    local_train_pipeline.main()

    run_prefix = resolve_run_prefix(root / "config", ENV, "train", RUN_ID)
    assert [call[0] for call in publishes.mock_calls] == ["upload", "sync"]
    assert publishes.mock_calls[0].args[1] == f"{run_prefix}run_identity.json"
    assert publishes.mock_calls[1].kwargs["completion_marker"] == "manifest.json"
