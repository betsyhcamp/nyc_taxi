"""Tests for the local Training runner's ordering guarantees.

`main()` runs end to end against a config tree copied under `tmp_path`, with
everything needing a repo, a bucket or real parquet patched out, so what is left
to assert is which local steps run, and in what order.
"""

import logging
import shutil
import sys
from pathlib import Path
from typing import cast

import pytest
from pytest_mock import MockerFixture

from fcstnyctaxi.core.train.backtest_impl import BacktestSummary
from fcstnyctaxi.core.train.compose_configs_impl import (
    ComposeConfigsSummary,
    compose_train_static_configs,
)
from fcstnyctaxi.core.train.evaluate_impl import EvaluateSummary
from fcstnyctaxi.core.train.final_fit_impl import FinalFitSummary
from fcstnyctaxi.lib import run_outputs
from fcstnyctaxi.lib.storage_layout import BUNDLE_MODEL_DIR_NAME, resolve_run_prefix
from fcstnyctaxi.lib.utils import get_project_root_dir
from fcstnyctaxi.pipelines import local_train_pipeline
from fcstnyctaxi.schemas.config.train import TrainModelingConfig
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
    train_run_id=RUN_ID,
    feature_run_id=FEATURE_RUN_ID,
    output_rows={"monthly_series.parquet": 6},
)
# One origin over a two month horizon is two folds, matching the summary above.
EVALUATE_SUMMARY = EvaluateSummary(
    train_run_id=RUN_ID,
    challenger_model="lightgbm",
    benchmark_model="naive",
    feature_run_id=FEATURE_RUN_ID,
    n_origins=1,
    first_origin="2025-05-18",
    last_origin="2025-05-18",
    n_series=3,
    n_folds_total=2,
    hero_metric_name="wrmae_pooled",
    hero_metric_values={"horizon_1": 0.92, "horizon_2": 0.88},
    output_rows={"fold_metrics.parquet": 4},
)
FINAL_FIT_SUMMARY = FinalFitSummary(
    train_end_ds="2025-05-18",
    n_series=3,
    n_obs=60,
    train_run_id=RUN_ID,
    feature_run_id=FEATURE_RUN_ID,
)


def _backtest_outputs(*, out_dir: Path, **_: object) -> BacktestSummary:
    """Write what the patched impl would: the marker the per-model publish sends
    and the scoring step reads to decide the sidecar is finished."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "backtest_manifest.json").write_text("{}")
    return BACKTEST_SUMMARY


def _evaluate_outputs(*, out_dir: Path, **_: object) -> EvaluateSummary:
    """The same, for the evaluate directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evaluate_manifest.json").write_text("{}")
    return EVALUATE_SUMMARY


def _final_fit_outputs(*, out_dir: Path, **_: object) -> FinalFitSummary:
    """The same, for the bundle, whose model directory is the only subdirectory any
    step publishes."""
    (out_dir / BUNDLE_MODEL_DIR_NAME).mkdir(parents=True, exist_ok=True)
    (out_dir / BUNDLE_MODEL_DIR_NAME / "weights.bin").write_bytes(b"fitted")
    (out_dir / "final_fit_manifest.json").write_text("{}")
    return FINAL_FIT_SUMMARY


def _challenger(root: Path) -> str:
    """Composed rather than named, so assertions hold whichever model the repo ships."""
    _, _, modeling = compose_train_static_configs(root / "config", ENV)
    return cast(TrainModelingConfig, modeling.config).model_roles.challenger


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
    """Four orderings, each with its own loss: no record of what the run read, a
    finished sidecar lost to a later failure, an absent pair scored, an unscored fit."""
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
    # One manager, impls included, so the assertion sees order across every call.
    publishes = mocker.MagicMock()
    publishes.attach_mock(
        mocker.patch.object(local_train_pipeline, "upload_to_gcs"), "identity"
    )
    publishes.attach_mock(sync, "sync")
    publishes.attach_mock(
        mocker.patch.object(
            local_train_pipeline, "backtest_impl", side_effect=_backtest_outputs
        ),
        "backtest",
    )
    publishes.attach_mock(
        mocker.patch.object(
            local_train_pipeline, "evaluate_impl", side_effect=_evaluate_outputs
        ),
        "evaluate",
    )
    final_fit = mocker.patch.object(
        local_train_pipeline, "final_fit_impl", side_effect=_final_fit_outputs
    )
    publishes.attach_mock(final_fit, "final_fit")

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
        "evaluate",
        "sync",
        "final_fit",
        "sync",
    ]
    assert publishes.mock_calls[0].args[1] == run_prefix
    assert [call.kwargs["completion_marker"] for call in sync.call_args_list] == [
        "manifest.json",
        "backtest_manifest.json",
        "backtest_manifest.json",
        "evaluate_manifest.json",
        "final_fit_manifest.json",
    ]
    # out_dir is mirrored from these, so pinning the URIs pins both locations.
    challenger = _challenger(root)
    assert [call.args[1] for call in sync.call_args_list[1:]] == [
        f"{run_prefix}backtest/{name}/" for name in COMPOSE_SUMMARY.model_names
    ] + [f"{run_prefix}evaluate/", f"{run_prefix}final_fit/{challenger}/"]
    # The registration target only: the benchmark is backtested and never fitted.
    assert final_fit.call_count == 1
    assert final_fit.call_args.kwargs["model_name"] == challenger


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
        local_train_pipeline, "evaluate_impl", side_effect=_evaluate_outputs
    )
    mocker.patch.object(
        local_train_pipeline, "final_fit_impl", side_effect=_final_fit_outputs
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
    assert (published / "evaluate" / "evaluate_manifest.json").is_file()
    # The bundle nests its model directory, which no other step publishes.
    bundle = published / "final_fit" / _challenger(root)
    assert (bundle / "final_fit_manifest.json").is_file()
    assert (bundle / BUNDLE_MODEL_DIR_NAME / "weights.bin").is_file()


def test_a_narrowed_rerun_scores_the_pair_the_run_id_has_accumulated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Two narrowed invocations complete one run's pair and the second scores it.
    Tie scoring to the invocation and recovering a run costs a second backtest."""
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
    mocker.patch.object(
        local_train_pipeline, "backtest_impl", side_effect=_backtest_outputs
    )
    mocker.patch.object(local_train_pipeline, "upload_to_gcs")
    mocker.patch.object(local_train_pipeline, "sync_to_gcs", return_value=(7, 0))
    mocker.patch.object(
        run_outputs, "read_text_from_gcs", return_value=RESOLVED_MANIFEST
    )
    evaluate = mocker.patch.object(
        local_train_pipeline, "evaluate_impl", side_effect=_evaluate_outputs
    )
    final_fit = mocker.patch.object(
        local_train_pipeline, "final_fit_impl", side_effect=_final_fit_outputs
    )
    # Composed here rather than named, so the assertion is about which role each
    # directory fills and not about the models this repo happens to ship.
    _, _, modeling = compose_train_static_configs(root / "config", ENV)
    roles = cast(TrainModelingConfig, modeling.config).model_roles
    # One model in both roles is a legal config, and under it the first invocation
    # below would complete the pair by itself, leaving nothing for the second.
    assert roles.challenger != roles.benchmark

    scratch = tmp_path / "scratch"
    run_prefix = resolve_run_prefix(root / "config", ENV, "train", RUN_ID)
    # What an interrupted backtest leaves behind: that impl deletes its marker
    # first and writes it last, so outputs without one are a half-written sidecar.
    partial = local_train_pipeline._mirror_path(
        local_train_pipeline._backtest_uri(run_prefix, roles.challenger), scratch
    )
    partial.mkdir(parents=True)
    (partial / "monthly_series.parquet").write_text("half of a sidecar")

    def _back(model_name: str) -> None:
        """One invocation narrowed to a single model, under a shared run id."""
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
                "--model",
                model_name,
                "--scratch-dir",
                str(scratch),
            ],
        )
        local_train_pipeline.main()

    with caplog.at_level(logging.WARNING):
        _back(roles.benchmark)

    # A present directory is not a finished one, and scoring the partial above
    # would read frames whose writer stopped partway.
    assert evaluate.call_count == 0
    # Ordering, as in the DAG: nothing is fitted for a pair that was never scored.
    assert final_fit.call_count == 0
    assert any(record.levelname == "WARNING" for record in caplog.records)

    _back(roles.challenger)

    assert evaluate.call_count == 1
    scored = evaluate.call_args.kwargs
    # A transposed pair is the one wiring error the tables never reveal.
    assert scored["challenger_dir"].name == roles.challenger
    assert scored["benchmark_dir"].name == roles.benchmark
    assert final_fit.call_count == 1
    assert final_fit.call_args.kwargs["model_name"] == roles.challenger


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
