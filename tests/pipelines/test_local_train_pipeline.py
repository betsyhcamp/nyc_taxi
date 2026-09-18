"""Tests for the local Training runner's ordering guarantees.

`main()` runs end to end against a config tree copied under `tmp_path`, with
everything needing a repo, a bucket or real parquet patched out, so what is left
to assert is the order of the local steps.
"""

import logging
import shutil
import sys
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from fcstnyctaxi.core.train.backtest_impl import BacktestSummary
from fcstnyctaxi.core.train.compose_configs_impl import ComposeConfigsSummary
from fcstnyctaxi.lib import run_outputs
from fcstnyctaxi.lib.storage_layout import resolve_run_prefix
from fcstnyctaxi.lib.utils import get_project_root_dir
from fcstnyctaxi.pipelines import local_train_pipeline
from fcstnyctaxi.schemas.run_outputs import FeatureArtifacts, FeatureRunOutputs

ENV = "dev"
RUN_ID = "t-20260913T000000000000Z"
FEATURE_RUN_ID = "f-20260913T000000000000Z"
PREVIOUS_OUTPUT = "the previous attempt's output"
# A step segment Training never constructs, so an assertion on these proves the
# URIs came from the manifest rather than from a path this runner guessed.
RESOLVED_PANEL_URI = f"gs://bucket/{ENV}/feature/{FEATURE_RUN_ID}/step/panel.parquet"
RESOLVED_CALENDAR_URI = (
    f"gs://bucket/{ENV}/feature/{FEATURE_RUN_ID}/step/calendar.parquet"
)
# Through the shared model, which is what makes writer and reader agree on shape.
RESOLVED_MANIFEST = FeatureRunOutputs(
    feature_run_id=FEATURE_RUN_ID,
    env=ENV,
    published=FeatureArtifacts(
        panel_uri=RESOLVED_PANEL_URI, calendar_uri=RESOLVED_CALENDAR_URI
    ),
).model_dump_json()
COMPOSE_SUMMARY = ComposeConfigsSummary(
    n_origins=1,
    first_origin="2025-05-18",
    last_origin="2025-05-18",
    last_complete_actual_month=202505,
    start_months=[202505],
    model_names=["naive", "lightgbm"],
)
BACKTEST_SUMMARY = BacktestSummary(
    n_origins=1,
    first_origin="2025-05-18",
    last_origin="2025-05-18",
    n_series=3,
    feature_run_id=FEATURE_RUN_ID,
    output_rows={"monthly_series.parquet": 6},
)


@pytest.fixture
def broken_config_root(tmp_path: Path) -> Path:
    """A real config tree corrupted so that only `train/modeling.yaml` fails.

    An undeclared key rather than a wrong type: the message names the fragment.
    """
    root = tmp_path / "project"
    shutil.copytree(get_project_root_dir() / "config", root / "config")
    modeling = root / "config" / "train" / "modeling.yaml"
    modeling.write_text(f"{modeling.read_text()}an_undeclared_key: 1\n")
    return root


@pytest.mark.parametrize("uri", ["s3://bucket/panel.parquet", "/tmp/panel.parquet"])
def test_mirror_path_rejects_a_non_gcs_uri(tmp_path: Path, uri: str) -> None:
    """An absolute path absorbs the mirror root, landing outside the scratch tree."""
    with pytest.raises(ValueError, match="must be a gs://"):
        local_train_pipeline._mirror_path(uri, tmp_path)


def test_a_model_filter_naming_no_composed_model_is_rejected() -> None:
    """Filtered silently it would match nothing, back no model, and exit clean."""
    with pytest.raises(ValueError, match="xgboost"):
        local_train_pipeline._select_models(["naive", "lightgbm"], "xgboost")


def test_the_model_filter_narrows_and_its_absence_keeps_every_model() -> None:
    """Both paths, since the guard above only says which one raises."""
    assert local_train_pipeline._select_models(["naive", "lightgbm"], "naive") == [
        "naive"
    ]
    assert local_train_pipeline._select_models(["naive", "lightgbm"], None) == [
        "naive",
        "lightgbm",
    ]


def test_a_malformed_train_config_raises_before_the_resolve_and_the_clear(
    broken_config_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """Both orderings at once: below the resolve a local config error reports as a
    network error, and below the rmtree the impl reads the configs too late."""
    monkeypatch.setenv("PROJECT_ROOT", str(broken_config_root))
    mocker.patch.object(
        local_train_pipeline, "require_git_hash", return_value="abc1234"
    )
    # No URI flags below, so this runs unless the preflight raises first, and its
    # message is what the assertion distinguishes the mis-ordered case by.
    mocker.patch.object(
        run_outputs, "read_text_from_gcs", side_effect=FileNotFoundError
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


def test_each_backtest_is_published_before_the_next_one_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """Two orderings. Published second, a failed sync would leave no record of what
    the run read; and running both backtests before publishing either would lose the
    first model's finished sidecar whenever the second one failed."""
    root = tmp_path / "project"
    shutil.copytree(get_project_root_dir() / "config", root / "config")
    monkeypatch.setenv("PROJECT_ROOT", str(root))
    mocker.patch.object(
        local_train_pipeline, "require_git_hash", return_value="abc1234"
    )
    mocker.patch.object(
        local_train_pipeline, "download_from_gcs", side_effect=lambda uri, d: d / "f"
    )
    mocker.patch.object(
        local_train_pipeline, "compose_configs_impl", return_value=COMPOSE_SUMMARY
    )
    sync = mocker.patch.object(local_train_pipeline, "sync_to_gcs", return_value=(7, 0))
    # One manager, so the assertion is about order across the calls rather than each
    # in isolation. backtest_impl belongs in it for the same reason: left out, a run
    # that backtested both models before publishing either would look identical.
    publishes = mocker.MagicMock()
    publishes.attach_mock(
        mocker.patch.object(local_train_pipeline, "upload_to_gcs"), "identity"
    )
    publishes.attach_mock(sync, "sync")
    publishes.attach_mock(
        mocker.patch.object(
            local_train_pipeline, "backtest_impl", return_value=BACKTEST_SUMMARY
        ),
        "backtest",
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
    assert [call[0] for call in publishes.mock_calls] == [
        "identity",
        "sync",
        "backtest",
        "sync",
        "backtest",
        "sync",
    ]
    assert publishes.mock_calls[0].args[1] == run_prefix
    assert [call.kwargs["completion_marker"] for call in sync.call_args_list] == [
        "manifest.json",
        "backtest_manifest.json",
        "backtest_manifest.json",
    ]
    # out_dir is mirrored from these, so pinning the URIs pins both locations.
    assert [call.args[1] for call in sync.call_args_list[1:]] == [
        f"{run_prefix}backtest/{name}/" for name in COMPOSE_SUMMARY.model_names
    ]


def test_the_published_destinations_are_ones_the_transport_accepts(
    fake_gcs: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """The ordering test above patches both publishers, so it proves the calls are
    made and never that the transport would accept what they name."""

    def _impl_outputs(*, out_dir: Path, **_: object) -> ComposeConfigsSummary:
        """Write what the patched impl would have, since the publish step sends it."""
        (out_dir.parent / "run_identity.json").write_text('{"seeded": true}')
        (out_dir / "manifest.json").write_text("{}")
        return COMPOSE_SUMMARY

    def _backtest_outputs(*, out_dir: Path, **_: object) -> BacktestSummary:
        """The same, for the sidecar the per-model publish sends."""
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "backtest_manifest.json").write_text("{}")
        return BACKTEST_SUMMARY

    root = tmp_path / "project"
    shutil.copytree(get_project_root_dir() / "config", root / "config")
    monkeypatch.setenv("PROJECT_ROOT", str(root))
    mocker.patch.object(
        local_train_pipeline, "require_git_hash", return_value="abc1234"
    )
    mocker.patch.object(
        local_train_pipeline, "download_from_gcs", side_effect=lambda uri, d: d / "f"
    )
    mocker.patch.object(
        local_train_pipeline, "compose_configs_impl", side_effect=_impl_outputs
    )
    mocker.patch.object(
        local_train_pipeline, "backtest_impl", side_effect=_backtest_outputs
    )
    mocker.patch.object(
        run_outputs, "read_text_from_gcs", return_value=RESOLVED_MANIFEST
    )
    # Neither publisher is patched: that is the whole point of this test.

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "local_train_pipeline",
            "--env",
            ENV,
            "--feature-run-id",
            FEATURE_RUN_ID,
            "--run-id",
            RUN_ID,
            "--scratch-dir",
            str(tmp_path / "scratch"),
        ],
    )

    local_train_pipeline.main()

    run_prefix = resolve_run_prefix(root / "config", ENV, "train", RUN_ID)
    published = fake_gcs / run_prefix.removeprefix("gs://")
    assert (published / "run_identity.json").is_file()
    assert (published / "compose_configs" / "manifest.json").is_file()
    # The backtest prefix carries a second segment, which no patched test exercises.
    for model_name in COMPOSE_SUMMARY.model_names:
        assert (
            published / "backtest" / model_name / "backtest_manifest.json"
        ).is_file()


def test_the_local_runner_resolves_both_uris_from_the_feature_run_id_alone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test the drift the flag-name mirror test cannot see: this caller could still
    require the overrides while both parsers declare the same flag names. The
    download assertion also holds the backtest step to the already-staged inputs."""
    root = tmp_path / "project"
    shutil.copytree(get_project_root_dir() / "config", root / "config")
    monkeypatch.setenv("PROJECT_ROOT", str(root))
    mocker.patch.object(
        local_train_pipeline, "require_git_hash", return_value="abc1234"
    )
    downloads = mocker.patch.object(
        local_train_pipeline, "download_from_gcs", side_effect=lambda uri, d: d / "f"
    )
    mocker.patch.object(
        local_train_pipeline, "compose_configs_impl", return_value=COMPOSE_SUMMARY
    )
    mocker.patch.object(
        local_train_pipeline, "backtest_impl", return_value=BACKTEST_SUMMARY
    )
    mocker.patch.object(local_train_pipeline, "upload_to_gcs")
    mocker.patch.object(local_train_pipeline, "sync_to_gcs", return_value=(7, 0))
    # Patched on the importing module, which imports at module scope. Picked wrong,
    # the failure is a credentials error in CI rather than an assertion failure.
    mocker.patch.object(
        run_outputs, "read_text_from_gcs", return_value=RESOLVED_MANIFEST
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
            "--run-id",
            RUN_ID,
            "--scratch-dir",
            str(tmp_path / "scratch"),
        ],
    )

    with caplog.at_level(logging.INFO):
        local_train_pipeline.main()

    assert [call.args[0] for call in downloads.call_args_list] == [
        RESOLVED_PANEL_URI,
        RESOLVED_CALENDAR_URI,
    ]
    # The evidence the override flags are to be retired on, from this caller.
    assert "source=resolved" in caplog.text
