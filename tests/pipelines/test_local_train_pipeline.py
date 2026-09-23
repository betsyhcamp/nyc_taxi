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
from fcstnyctaxi.core.train.register_model_impl import RegisterModelSummary
from fcstnyctaxi.lib import run_outputs
from fcstnyctaxi.lib.config.bindings import model_names_from_roles, resolve_model_roles
from fcstnyctaxi.lib.storage_layout import (
    BUNDLE_MODEL_DIR_NAME,
    RUN_OUTPUTS_FILENAME,
    SourcedPath,
    resolve_run_prefix,
)
from fcstnyctaxi.lib.utils import get_project_root_dir, require_path_safe_run_id
from fcstnyctaxi.pipelines import local_train_pipeline
from fcstnyctaxi.schemas.config.train import TrainModelingConfig
from fcstnyctaxi.schemas.run_outputs import FeatureArtifacts, FeatureRunOutputs

ENV = "dev"
RUN_ID = "t-20260913t000000000000z"
FEATURE_RUN_ID = "f-20260913T000000000000Z"
PREVIOUS_OUTPUT = "the previous attempt's output"
# A step segment Training never constructs, so an assertion on these proves the
# URIs came from the manifest rather than from a path this runner guessed.
RESOLVED_PANEL_URI = f"gs://bucket/{ENV}/feature/{FEATURE_RUN_ID}/step/panel.parquet"
RESOLVED_CALENDAR_URI = (
    f"gs://bucket/{ENV}/feature/{FEATURE_RUN_ID}/step/calendar.parquet"
)
RESOLVED_EXOG_URI = f"gs://bucket/{ENV}/feature/{FEATURE_RUN_ID}/step/exogenous.parquet"
# Through the shared model, which is what makes writer and reader agree on shape.
RESOLVED_MANIFEST = FeatureRunOutputs(
    feature_run_id=FEATURE_RUN_ID,
    env=ENV,
    published=FeatureArtifacts(
        panel_uri=RESOLVED_PANEL_URI,
        calendar_uri=RESOLVED_CALENDAR_URI,
        exogenous_uri=RESOLVED_EXOG_URI,
    ),
).model_dump_json()
# The shipped roles, so the patched steps back the models the runner goes on to score.
SHIPPED_ROLES = resolve_model_roles(get_project_root_dir() / "config")
COMPOSE_SUMMARY = ComposeConfigsSummary(
    n_origins=1,
    first_origin="2025-05-18",
    last_origin="2025-05-18",
    last_complete_actual_month=202505,
    start_months=[202505],
    model_names=list(model_names_from_roles(SHIPPED_ROLES)),
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
    challenger_model=SHIPPED_ROLES.challenger,
    benchmark_model=SHIPPED_ROLES.benchmark,
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
SERVING_IMAGE = "us-central1-docker.pkg.dev/p/r/train@sha256:" + "c" * 64
# The challenger's, since only a run of the shipped roles could produce this summary.
# The prefix stays fake: nothing reads it, and composing the real one would tie a
# fully patched test to infra.yaml.
_REGISTERED_MODEL_ID = f"fcst-a-{SHIPPED_ROLES.challenger}"
REGISTER_SUMMARY = RegisterModelSummary(
    model_tag=f"projects/123456789/locations/us-central1/models/{_REGISTERED_MODEL_ID}@1",
    model_id=_REGISTERED_MODEL_ID,
    version_id="1",
    uploaded=True,
    model_name=SHIPPED_ROLES.challenger,
    train_run_id=RUN_ID,
    git_hash="abc1234",
)


def _compose_outputs(*, out_dir: Path, **_: object) -> ComposeConfigsSummary:
    """Write what the patched impl would have, since the publish step sends it."""
    (out_dir.parent / "run_identity.json").write_text('{"seeded": true}')
    (out_dir / "manifest.json").write_text("{}")
    return COMPOSE_SUMMARY


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


def _register_outputs(*, run_dir: Path, **_: object) -> RegisterModelSummary:
    """The same, for the run record the impl writes at the run root."""
    (run_dir / RUN_OUTPUTS_FILENAME).write_text("{}")
    return REGISTER_SUMMARY


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


def test_a_run_id_safe_as_a_path_but_not_as_a_label_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """Test the label charset is refused here, not by Vertex at the last task."""
    run_id = RUN_ID.upper()
    # Self-check: only the stricter guard can refuse it.
    require_path_safe_run_id(run_id, "--run-id")
    # Keeps a run with the guard removed off the network.
    mocker.patch.object(
        run_outputs, "read_text_from_gcs", side_effect=FileNotFoundError
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
            run_id,
            "--scratch-dir",
            str(tmp_path / "scratch"),
        ],
    )

    with pytest.raises(ValueError, match="--run-id"):
        local_train_pipeline.main()


@pytest.mark.parametrize(
    "serving_image", [None, SERVING_IMAGE], ids=["unregistered", "registered"]
)
def test_each_backtest_is_published_before_the_next_one_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
    serving_image: str | None,
) -> None:
    """Five orderings, each with its own loss: no record of what the run read, a lost
    sidecar, an absent pair scored, an unscored fit, a record ahead of its steps."""
    root = tmp_path / "project"
    shutil.copytree(get_project_root_dir() / "config", root / "config")
    monkeypatch.setenv("PROJECT_ROOT", str(root))
    mocker.patch.object(
        local_train_pipeline, "require_git_hash", return_value="abc1234"
    )
    mocker.patch.object(
        local_train_pipeline, "download_from_gcs", side_effect=lambda uri, d: d / "f"
    )
    compose = mocker.patch.object(
        local_train_pipeline, "compose_configs_impl", return_value=COMPOSE_SUMMARY
    )
    sync = mocker.patch.object(local_train_pipeline, "sync_to_gcs", return_value=(7, 0))
    # One manager, impls included, so the assertion sees order across every call.
    publishes = mocker.MagicMock()
    publishes.attach_mock(
        mocker.patch.object(local_train_pipeline, "upload_to_gcs"), "upload"
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
    delete = mocker.patch.object(local_train_pipeline, "delete_from_gcs")
    publishes.attach_mock(delete, "delete")
    register = mocker.patch.object(
        local_train_pipeline, "register_model_impl", side_effect=_register_outputs
    )
    publishes.attach_mock(register, "register")

    registering = [] if serving_image is None else ["--serving-image", serving_image]
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
            "--additional-exog-uri",
            f"gs://bucket/{ENV}/feature/{FEATURE_RUN_ID}/data_prep/exogenous_features.parquet",
            "--run-id",
            RUN_ID,
            *registering,
            "--scratch-dir",
            str(tmp_path / "scratch"),
        ],
    )

    local_train_pipeline.main()

    run_prefix = resolve_run_prefix(root / "config", ENV, "train", RUN_ID)
    # Without the flag the run stops after the fit: no registry write, no record.
    registered = [] if serving_image is None else ["register", "upload"]
    assert [call[0] for call in publishes.mock_calls] == [
        "upload",
        "sync",
        "backtest",
        "sync",
        "backtest",
        "sync",
        "evaluate",
        "sync",
        "final_fit",
        "sync",
        "delete",
        *registered,
    ]
    assert publishes.mock_calls[0].args[1] == run_prefix
    delete.assert_called_once_with(f"{run_prefix}{RUN_OUTPUTS_FILENAME}")
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

    if serving_image is not None:
        # The wrapper's arguments: one bundle as mirror and URI, the compose step's
        # directory, and the run root above it, where the record is published from.
        kwargs = register.call_args.kwargs
        assert kwargs["bundle"] == SourcedPath(
            path=final_fit.call_args.kwargs["out_dir"],
            uri=f"{run_prefix}final_fit/{challenger}/",
        )
        assert kwargs["compose_configs_dir"] == compose.call_args.kwargs["out_dir"]
        assert kwargs["run_dir"] == kwargs["compose_configs_dir"].parent
        assert kwargs["serving_container_image_uri"] == SERVING_IMAGE
        assert publishes.mock_calls[-1].args == (
            kwargs["run_dir"] / RUN_OUTPUTS_FILENAME,
            run_prefix,
        )


def test_the_published_destinations_are_ones_the_transport_accepts(
    fake_gcs: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """The ordering test above patches both publishers, so it proves the calls are
    made and never that the transport would accept what they name."""
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
        local_train_pipeline, "compose_configs_impl", side_effect=_compose_outputs
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
        local_train_pipeline, "register_model_impl", side_effect=_register_outputs
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
            "--serving-image",
            SERVING_IMAGE,
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
    # A file to the run prefix, like run_identity.json, but last of the run.
    assert (published / RUN_OUTPUTS_FILENAME).is_file()


@pytest.mark.parametrize(
    "registering", [False, True], ids=["unregistered", "failed_registration"]
)
def test_a_rerun_leaves_no_earlier_record_published(
    fake_gcs: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
    registering: bool,
) -> None:
    """The rerun replaced the bundle, so an earlier record left standing would point
    Inference at a version whose bytes are gone; Vertex's impl deletes it too."""
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
        local_train_pipeline, "compose_configs_impl", side_effect=_compose_outputs
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
        local_train_pipeline,
        "register_model_impl",
        side_effect=ValueError("registered at another git_hash"),
    )
    mocker.patch.object(
        run_outputs, "read_text_from_gcs", return_value=RESOLVED_MANIFEST
    )
    run_prefix = resolve_run_prefix(root / "config", ENV, "train", RUN_ID)
    # What an earlier, registered run of this id left at the run root.
    earlier_record = fake_gcs / run_prefix.removeprefix("gs://") / RUN_OUTPUTS_FILENAME
    earlier_record.parent.mkdir(parents=True)
    earlier_record.write_text('{"earlier": true}')

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
            *(["--serving-image", SERVING_IMAGE] if registering else []),
            "--scratch-dir",
            str(tmp_path / "scratch"),
        ],
    )

    if registering:
        with pytest.raises(ValueError, match="another git_hash"):
            local_train_pipeline.main()
    else:
        local_train_pipeline.main()

    assert not earlier_record.exists()


def test_a_serving_image_that_is_not_digest_pinned_is_refused_before_any_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    """A tag can be repointed, so the runtime a version records would drift."""
    # Keeps a run with the guard removed off the network.
    mocker.patch.object(
        run_outputs, "read_text_from_gcs", side_effect=FileNotFoundError
    )
    downloads = mocker.patch.object(local_train_pipeline, "download_from_gcs")
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
            "--serving-image",
            "us-central1-docker.pkg.dev/p/r/train:latest",
            "--scratch-dir",
            str(tmp_path / "scratch"),
        ],
    )

    with pytest.raises(ValueError, match="--serving-image"):
        local_train_pipeline.main()

    downloads.assert_not_called()


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
    mocker.patch.object(local_train_pipeline, "delete_from_gcs")
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


def test_compose_is_handed_each_path_paired_with_the_uri_it_was_built_from(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """A path mirrored from the wrong URI still exists and still reads, so the
    pairing is the only thing that catches it. The exogenous path is mirrored and
    not downloaded, because nothing opens that file until the impls do."""
    root = tmp_path / "project"
    shutil.copytree(get_project_root_dir() / "config", root / "config")
    monkeypatch.setenv("PROJECT_ROOT", str(root))
    mocker.patch.object(
        local_train_pipeline, "require_git_hash", return_value="abc1234"
    )
    # The real return value, so a staged path equals the mirror path it lands in.
    mocker.patch.object(
        local_train_pipeline,
        "download_from_gcs",
        side_effect=lambda uri, directory: directory / uri.rsplit("/", 1)[-1],
    )
    compose = mocker.patch.object(
        local_train_pipeline, "compose_configs_impl", return_value=COMPOSE_SUMMARY
    )
    mocker.patch.object(
        local_train_pipeline, "backtest_impl", return_value=BACKTEST_SUMMARY
    )
    mocker.patch.object(local_train_pipeline, "upload_to_gcs")
    mocker.patch.object(local_train_pipeline, "sync_to_gcs", return_value=(7, 0))
    mocker.patch.object(
        run_outputs, "read_text_from_gcs", return_value=RESOLVED_MANIFEST
    )

    scratch = tmp_path / "scratch"
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

    local_train_pipeline.main()

    kwargs = compose.call_args.kwargs
    for name, uri in (
        ("panel", RESOLVED_PANEL_URI),
        ("calendar", RESOLVED_CALENDAR_URI),
        ("additional_exog", RESOLVED_EXOG_URI),
    ):
        assert kwargs[name] == SourcedPath(
            path=local_train_pipeline._mirror_path(uri, scratch), uri=uri
        )


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
