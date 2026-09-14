import os

# Precedes the imports: the compile fixture below binds base_image at import time,
# and an unset value aborts collection for this whole module.
os.environ.setdefault(
    "FCST_TRAIN_IMAGE",
    "us-central1-docker.pkg.dev/nyc-taxi-ehc/fcst-ml-containers/train@sha256:"
    + "a" * 64,
)

import ast
import hashlib
import logging
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest
import yaml
from kfp import compiler
from pytest_mock import MockerFixture

from fcstnyctaxi.pipelines import local_train_pipeline
from fcstnyctaxi.pipelines.train_pipeline import train_pipeline
from scripts import submit_train_pipeline

ENV = "dev"
RUN_ID = "t-20260913T000000000000Z"
FEATURE_RUN_ID = "f-20260913T000000000000Z"
PANEL_URI = f"gs://bucket/{ENV}/feature/{FEATURE_RUN_ID}/data_prep/time_series.parquet"
CALENDAR_URI = (
    f"gs://bucket/{ENV}/feature/{FEATURE_RUN_ID}/data_prep/fiscal_calendar.parquet"
)
SERVICE_ACCOUNT = "svc-train@nyc-taxi-ehc.iam.gserviceaccount.com"
RESOURCE_NAME = "projects/123456789/locations/us-central1/pipelineJobs/fcst-train-x"

# Flags each command line owns because of how it executes, not what it models.
# Everything else must appear on both, so a new flag has to be classified here.
LOCAL_ORCHESTRATION_FLAGS = frozenset({"--scratch-dir"})
VERTEX_ORCHESTRATION_FLAGS = frozenset({"--template-path", "--wait", "--no-caching"})


@pytest.fixture(scope="module")
def template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The real Training pipeline, compiled once against the seeded image."""
    path = tmp_path_factory.mktemp("build") / "fcst-train-pipeline.yaml"
    compiler.Compiler().compile(
        pipeline_func=train_pipeline,  # type: ignore[arg-type]
        package_path=str(path),
    )
    return path


@pytest.fixture
def importer_only_template(template: Path, tmp_path: Path) -> Path:
    """Derived from the real template, not hand-written, so it stays submittable."""
    ir = yaml.safe_load(template.read_text())
    for executor in ir["deploymentSpec"]["executors"].values():
        executor.pop("container", None)
        executor.setdefault("importer", {"artifactUri": {}})
    path = tmp_path / "importers-only.yaml"
    path.write_text(yaml.safe_dump(ir))
    return path


def _argv(template: Path, overrides: dict[str, str] | None = None) -> list[str]:
    """The flags a submission takes, with any value replaced by `overrides`."""
    values = {
        "--template-path": str(template),
        "--env": ENV,
        "--feature-run-id": FEATURE_RUN_ID,
        "--panel-uri": PANEL_URI,
        "--calendar-uri": CALENDAR_URI,
        "--run-id": RUN_ID,
    }
    values.update(overrides or {})
    argv = ["submit_train_pipeline"]
    for flag, value in values.items():
        argv += [flag, value]
    return argv


@pytest.fixture
def vertex(
    mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> MagicMock:
    """Patched, not constructed: `PipelineJob.__init__` resolves credentials, and
    `load_dotenv` would otherwise hand back the service account a test withholds."""
    mocker.patch.object(submit_train_pipeline, "load_dotenv")
    monkeypatch.setenv(submit_train_pipeline._SERVICE_ACCOUNT_VAR, SERVICE_ACCOUNT)
    monkeypatch.setattr(
        submit_train_pipeline, "_LAST_RUN_ID_PATH", tmp_path / ".last_run_id"
    )
    sdk = mocker.patch.object(submit_train_pipeline, "aiplatform")
    sdk.PipelineJob.return_value.resource_name = RESOURCE_NAME
    return sdk


def _cli_flags(module: ModuleType) -> frozenset[str]:
    """Every option string the module's parser declares, read from its source:
    argparse exposes no public way to enumerate a parser's options."""
    tree = ast.parse(Path(module.__file__ or "").read_text())
    return frozenset(
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    )


def test_submitted_parameters_and_provenance_come_from_the_real_template(
    template: Path,
    vertex: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test both halves against a real template: a literal parameter list would
    pass a rename, and a moved image key would report nothing."""
    monkeypatch.setattr("sys.argv", _argv(template))

    with caplog.at_level(logging.INFO):
        submit_train_pipeline.main()

    ir = yaml.safe_load(template.read_text())
    parameter_values = vertex.PipelineJob.call_args.kwargs["parameter_values"]
    assert set(parameter_values) == set(ir["root"]["inputDefinitions"]["parameters"])
    assert parameter_values == {
        "env": ENV,
        "train_run_id": RUN_ID,
        "feature_run_id": FEATURE_RUN_ID,
        "panel_uri": PANEL_URI,
        "calendar_uri": CALENDAR_URI,
    }
    assert os.environ["FCST_TRAIN_IMAGE"] in caplog.text
    assert hashlib.sha256(template.read_bytes()).hexdigest() in caplog.text


def test_a_missing_service_account_raises_before_the_sdk_is_touched(
    template: Path,
    vertex: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the whole mitigation for EnvironmentConfig carrying no identities."""
    monkeypatch.delenv(submit_train_pipeline._SERVICE_ACCOUNT_VAR)
    monkeypatch.setattr("sys.argv", _argv(template))

    with pytest.raises(RuntimeError, match="FCST_TRAIN_SERVICE_ACCOUNT"):
        submit_train_pipeline.main()

    assert vertex.mock_calls == []


@pytest.mark.parametrize(
    ("flags", "expected"),
    [([], True), (["--no-caching"], False)],
    ids=["default-on", "no-caching"],
)
def test_caching_is_on_unless_refused(
    flags: list[str],
    expected: bool,
    template: Path,
    vertex: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the caching polarity: the gate needs a resubmission to hit the cache."""
    monkeypatch.setattr("sys.argv", _argv(template) + flags)

    submit_train_pipeline.main()

    assert vertex.PipelineJob.call_args.kwargs["enable_caching"] is expected


def test_last_run_id_records_the_submitted_id(
    template: Path,
    vertex: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test that the id a restart needs is written where the next run can read it."""
    monkeypatch.setattr("sys.argv", _argv(template))

    submit_train_pipeline.main()

    assert (tmp_path / ".last_run_id").read_text() == f"{RUN_ID}\n"


def test_a_failed_submit_records_no_run_id(
    template: Path,
    vertex: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test the ordering: written first, the id would name a run that never ran."""
    vertex.PipelineJob.return_value.submit.side_effect = RuntimeError("rejected")
    monkeypatch.setattr("sys.argv", _argv(template))

    with pytest.raises(RuntimeError, match="rejected"):
        submit_train_pipeline.main()

    assert not (tmp_path / ".last_run_id").exists()


def test_a_malformed_feature_run_id_names_the_flag_it_arrived_on(
    template: Path,
    vertex: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that the error names the flag, since both ids arrive as argv."""
    monkeypatch.setattr("sys.argv", _argv(template, {"--feature-run-id": "../escape"}))

    with pytest.raises(ValueError, match="--feature-run-id"):
        submit_train_pipeline.main()


def test_a_template_pinning_no_container_executors_warns_and_still_submits(
    importer_only_template: Path,
    vertex: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test the only condition that warns; refusing is the compile step's job."""
    monkeypatch.setattr("sys.argv", _argv(importer_only_template))

    with caplog.at_level(logging.INFO):
        submit_train_pipeline.main()

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "zero container executors" in warnings[0].getMessage()
    vertex.PipelineJob.return_value.submit.assert_called_once()


def test_domain_flags_mirror_the_local_runners() -> None:
    """Test the drift the parameter test cannot see: a flag on one parser only."""
    local = _cli_flags(local_train_pipeline) - LOCAL_ORCHESTRATION_FLAGS
    vertex_flags = _cli_flags(submit_train_pipeline) - VERTEX_ORCHESTRATION_FLAGS

    # Without this the walk finding nothing would pass the comparison below.
    assert local
    assert vertex_flags == local
