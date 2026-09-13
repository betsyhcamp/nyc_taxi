import os

os.environ.setdefault(
    "FCST_TRAIN_IMAGE",
    "us-central1-docker.pkg.dev/nyc-taxi-ehc/fcst-ml-containers/train@sha256:"
    + "a" * 64,
)

from pathlib import Path
from typing import Any, cast

import pytest
from kfp.dsl import Artifact, Dataset
from kfp.dsl.python_component import PythonComponent
from pytest_mock import MockerFixture

from fcstnyctaxi.components.train.compose_configs_component import (
    _require_digest_ref,
    compose_configs,
)
from fcstnyctaxi.core.train.compose_configs_impl import (
    ComposeConfigsSummary,
    SourcedPath,
)
from fcstnyctaxi.lib.storage_layout import resolve_run_prefix
from fcstnyctaxi.lib.utils import get_project_root_dir

# dsl.component has no return annotation, so a checker sees the undecorated function
# and .execute() reads as unknown. Cast once rather than at each call site.
COMPONENT = cast(PythonComponent, compose_configs)

CONFIG_DIR = get_project_root_dir() / "config"

# The one input that cannot be a sentinel: composition runs against the real tree.
ENV = "dev"
TRAIN_RUN_ID = "t-sentinel"
FEATURE_RUN_ID = "f-sentinel"
GIT_HASH = "abc1234-dirty"

# Distinct, so pairing assertions also prove panel and calendar were not swapped.
PANEL_URI = "gs://sentinel-bucket/feature/f-sentinel/panel.parquet"
CALENDAR_URI = "gs://sentinel-bucket/feature/f-sentinel/calendar.parquet"

_REPO = "us-central1-docker.pkg.dev/nyc-taxi-ehc/fcst-ml-containers"

VALID_IMAGE = (
    "us-central1-docker.pkg.dev/nyc-taxi-ehc/fcst-ml-containers/train@sha256:"
    + "b" * 64
)

# Real, not a Mock: the wrapper feeds as_dict() to metadata.update(), which raises
# on a Mock for a reason unrelated to anything under test.
SUMMARY = ComposeConfigsSummary(
    n_origins=3,
    first_origin="2025-04-20",
    last_origin="2025-06-15",
    last_complete_actual_month=202503,
    start_months=[202501, 202502, 202503],
    model_names=["naive", "xgboost"],
)


@pytest.fixture
def real_config_dir(mocker: MockerFixture) -> Path:
    """Point the component's baked /app/config at the real config tree.

    A string target, so TID251 never sees an import; it reaches the component
    because KFP requires that import inside the body.
    """
    mocker.patch("fcstnyctaxi.runtime_paths.CONFIG_DIR", CONFIG_DIR)
    return CONFIG_DIR


@pytest.fixture
def mock_impl(mocker: MockerFixture) -> Any:
    """compose_configs_impl replaced at its source module, as the wrapper imports it."""
    mock = mocker.patch(
        "fcstnyctaxi.core.train.compose_configs_impl.compose_configs_impl"
    )
    mock.return_value = SUMMARY
    return mock


@pytest.fixture
def baked_git_hash(monkeypatch: pytest.MonkeyPatch) -> str:
    """The value Dockerfile.train bakes as ENV FCST_GIT_HASH."""
    monkeypatch.setenv("FCST_GIT_HASH", GIT_HASH)
    return GIT_HASH


def _artifacts() -> tuple[Dataset, Dataset, Artifact]:
    """Fresh inputs and output; composed_configs starts at uri="" so .path reads ""."""
    return (
        Dataset(name="panel", uri=PANEL_URI),
        Dataset(name="calendar", uri=CALENDAR_URI),
        Artifact(name="composed_configs", uri=""),
    )


def _expected_run_prefix() -> str:
    """Derived, never a literal: a bucket change in dev.yaml must not fail this.

    Pins that the wrapper resolves through one function; the convention itself is
    pinned by tests/lib/test_storage_layout.py.
    """
    return resolve_run_prefix(CONFIG_DIR, ENV, "train", TRAIN_RUN_ID)


def test_wrapper_wires_inputs_outputs_and_metadata(
    real_config_dir: Path, mock_impl: Any, baked_git_hash: str
) -> None:
    """Test that the wrapper places its output, pairs each input, and stamps it."""
    panel, calendar, composed_configs = _artifacts()
    assert composed_configs.path == ""  # baseline: demonstrably wrong until assigned

    result = COMPONENT.execute(
        env=ENV,
        train_run_id=TRAIN_RUN_ID,
        feature_run_id=FEATURE_RUN_ID,
        panel=panel,
        calendar=calendar,
        composed_configs=composed_configs,
    )

    expected_prefix = _expected_run_prefix()
    expected_uri = f"{expected_prefix}compose_configs/"

    # Indexed, not attribute: the wrapper returns a bare tuple KFP maps positionally.
    assert result[0] == expected_prefix
    assert composed_configs.uri == expected_uri

    kwargs = mock_impl.call_args.kwargs
    # Through KFP rather than hardcoding /gcs/: tests our ordering, not KFP's mount.
    assert kwargs["out_dir"] == Path(Artifact(uri=expected_uri).path)
    assert kwargs["panel"] == SourcedPath(path=Path(panel.path), uri=PANEL_URI)
    assert kwargs["calendar"] == SourcedPath(path=Path(calendar.path), uri=CALENDAR_URI)
    assert kwargs["config_dir"] == CONFIG_DIR
    assert kwargs["env"] == ENV
    assert kwargs["expected_feature_run_id"] == FEATURE_RUN_ID
    assert kwargs["train_run_id"] == TRAIN_RUN_ID
    assert kwargs["git_hash"] == GIT_HASH

    # Every summary field reaches metadata, without pinning which fields exist.
    for key, value in SUMMARY.as_dict().items():
        assert composed_configs.metadata[key] == value
    assert composed_configs.metadata["train_run_id"] == TRAIN_RUN_ID
    assert composed_configs.metadata["feature_run_id"] == FEATURE_RUN_ID
    assert composed_configs.metadata["git_hash"] == GIT_HASH


@pytest.mark.parametrize("baked_value", [None, ""])
def test_missing_git_hash_raises_naming_the_variable(
    real_config_dir: Path,
    mock_impl: Any,
    monkeypatch: pytest.MonkeyPatch,
    baked_value: str | None,
) -> None:
    """Test that absent or empty FCST_GIT_HASH is refused by name, before any work."""
    if baked_value is None:
        monkeypatch.delenv("FCST_GIT_HASH", raising=False)
    else:
        monkeypatch.setenv("FCST_GIT_HASH", baked_value)

    panel, calendar, composed_configs = _artifacts()

    with pytest.raises(RuntimeError, match="FCST_GIT_HASH"):
        COMPONENT.execute(
            env=ENV,
            train_run_id=TRAIN_RUN_ID,
            feature_run_id=FEATURE_RUN_ID,
            panel=panel,
            calendar=calendar,
            composed_configs=composed_configs,
        )

    mock_impl.assert_not_called()


def test_impl_failure_propagates_and_leaves_metadata_unstamped(
    real_config_dir: Path, mock_impl: Any, baked_git_hash: str
) -> None:
    """Test that an impl failure reaches the caller and stamps no metadata."""
    mock_impl.side_effect = ValueError("panel and calendar are the same filepath")
    panel, calendar, composed_configs = _artifacts()

    with pytest.raises(ValueError, match="same filepath"):
        COMPONENT.execute(
            env=ENV,
            train_run_id=TRAIN_RUN_ID,
            feature_run_id=FEATURE_RUN_ID,
            panel=panel,
            calendar=calendar,
            composed_configs=composed_configs,
        )

    assert composed_configs.metadata == {}
    # Not "untouched": .uri is assigned before the impl call, so asserting a pristine
    # artifact would assert the ordering bug back in.
    assert composed_configs.uri == f"{_expected_run_prefix()}compose_configs/"


def test_the_generated_container_module_defines_the_named_output() -> None:
    """Test that the generated container code resolves its annotation, output named
    run_prefix as PR 4's tasks consume it."""
    # run_prefix, not KFP's default "Output": a bare `-> str` compiles, runs, and
    # renames what downstream wrappers ask for.
    outputs = COMPONENT.component_spec.outputs
    assert outputs is not None
    assert "run_prefix" in outputs

    # Asserted, not cast: narrows for a checker and says which part of the IR moved
    # if a kfp upgrade reshapes it.
    container = COMPONENT.component_spec.implementation.container
    assert container is not None
    assert container.command is not None
    generated = container.command[-1]
    assert isinstance(generated, str)

    # Executing the definition is the assertion: KFP copies only the body and a fixed
    # preamble, so a module-level NamedTuple raises NameError here, in the container,
    # with every other test in this file still green.
    exec(compile(generated, "<generated>", "exec"), {"__name__": "__generated__"})


@pytest.mark.parametrize(
    "image",
    [
        VALID_IMAGE,
        # Nested names are legal in Artifact Registry, and the repository is the
        # first three segments whatever the depth.
        "us-central1-docker.pkg.dev/nyc-taxi-ehc/fcst-ml-containers/team/train"
        "@sha256:" + "c" * 64,
    ],
    ids=["flat", "nested"],
)
def test_require_digest_ref_accepts_a_digest_pinned_reference(image: str) -> None:
    """Test that a well-formed reference is returned unchanged, nesting included."""
    assert _require_digest_ref(image) == image


@pytest.mark.parametrize(
    ("image", "expected_message"),
    [
        (None, "not set"),
        ("", "not set"),
        (_REPO + "/train:abc1234", "64 hex"),
        (_REPO + "/train@sha256:zz", "64 hex"),
        (_REPO + "/train@sha256:", "64 hex"),
        (_REPO + "/@sha256:" + "c" * 64, "64 hex"),
        ("a/b/c@sha256:" + "c" * 64, "64 hex"),
    ],
    ids=[
        "unset",
        "empty",
        "tag",
        "digest-not-hex",
        "digest-empty",
        "no-name",
        "no-host",
    ],
)
def test_require_digest_ref_rejects(image: str | None, expected_message: str) -> None:
    """Test that anything but a digest-pinned registry reference is refused.

    The tag is the likely mistake; the rest the old "@sha256:" substring test allowed.
    """
    with pytest.raises(ValueError, match=expected_message):
        _require_digest_ref(image)
