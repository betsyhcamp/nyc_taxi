import os

os.environ.setdefault(
    "FCST_TRAIN_IMAGE",
    "us-central1-docker.pkg.dev/nyc-taxi-ehc/fcst-ml-containers/train@sha256:"
    + "a" * 64,
)

from pathlib import Path
from typing import Any, cast

import pytest
from kfp.dsl import Artifact, Model
from kfp.dsl.python_component import PythonComponent
from pytest_mock import MockerFixture

from fcstnyctaxi.components.train.register_model_component import register_model
from fcstnyctaxi.core.train.register_model_impl import RegisterModelSummary
from fcstnyctaxi.lib.storage_layout import SourcedPath

# dsl.component has no return annotation, so a checker reads .execute() as unknown.
COMPONENT = cast(PythonComponent, register_model)

RUN_ROOT = "gs://sentinel-bucket/dev/train/t-sentinel/"
MODEL_NAME = "model_a"

# Distinct, so a transposed pair fails the pairing assertions.
COMPOSED_CONFIGS_URI = f"{RUN_ROOT}compose_configs/"
BUNDLE_URI = f"{RUN_ROOT}final_fit/{MODEL_NAME}/"
# Not the seeded FCST_TRAIN_IMAGE, so the module's _IMAGE cannot pass for this.
SERVING_IMAGE = "us-central1-docker.pkg.dev/p/r/train@sha256:" + "b" * 64

# Real, not a Mock: metadata.update() raises on one, for a reason unrelated to
# anything under test.
SUMMARY = RegisterModelSummary(
    model_tag=f"projects/123456789/locations/us-central1/models/fcst-a-{MODEL_NAME}@3",
    model_id=f"fcst-a-{MODEL_NAME}",
    version_id="3",
    uploaded=True,
    model_name=MODEL_NAME,
    train_run_id="t-sentinel",
    git_hash="abc1234",
)


@pytest.fixture
def mock_impl(mocker: MockerFixture) -> Any:
    """register_model_impl replaced at its source module, as the wrapper imports it."""
    mock = mocker.patch(
        "fcstnyctaxi.core.train.register_model_impl.register_model_impl"
    )
    mock.return_value = SUMMARY
    return mock


def _artifacts() -> tuple[Artifact, Model, Model]:
    """Fresh inputs and output; registered starts at uri=""."""
    return (
        Artifact(name="composed_configs", uri=COMPOSED_CONFIGS_URI),
        Model(name="bundle", uri=BUNDLE_URI),
        Model(name="registered", uri=""),
    )


def _execute(composed_configs: Artifact, bundle: Model, registered: Model) -> None:
    """Run the component as KFP's local executor would."""
    COMPONENT.execute(
        serving_container_image_uri=SERVING_IMAGE,
        composed_configs=composed_configs,
        bundle=bundle,
        registered=registered,
    )


def test_wrapper_pairs_every_input_and_records_the_registered_version(
    mock_impl: Any,
) -> None:
    """The bundle's mount and URI travel as one pair, and the output names the version,
    never a gs:// path."""
    composed_configs, bundle, registered = _artifacts()

    _execute(composed_configs, bundle, registered)

    kwargs = mock_impl.call_args.kwargs
    assert kwargs["bundle"] == SourcedPath(path=Path(bundle.path), uri=BUNDLE_URI)
    assert kwargs["compose_configs_dir"] == Path(composed_configs.path)
    assert kwargs["run_dir"] == Path(composed_configs.path).parent
    assert kwargs["serving_container_image_uri"] == SERVING_IMAGE

    assert registered.uri == SUMMARY.model_tag
    # Every summary field reaches metadata, without pinning which fields exist.
    for key, value in SUMMARY.as_dict().items():
        assert registered.metadata[key] == value


def test_the_stamped_hash_is_the_labels_not_the_images(
    mock_impl: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second source for the hash could disagree with the label it describes."""
    monkeypatch.setenv("FCST_GIT_HASH", "fffffff")
    composed_configs, bundle, registered = _artifacts()

    _execute(composed_configs, bundle, registered)

    assert registered.metadata["git_hash"] == SUMMARY.git_hash


def test_impl_failure_propagates_and_leaves_the_output_untouched(
    mock_impl: Any,
) -> None:
    """The URI is the version the impl returns, so a failed impl leaves none to set."""
    mock_impl.side_effect = ValueError("registered at another git_hash")
    composed_configs, bundle, registered = _artifacts()

    with pytest.raises(ValueError, match="another git_hash"):
        _execute(composed_configs, bundle, registered)

    assert registered.uri == ""
    assert registered.metadata == {}


def test_the_program_kfp_ships_runs_with_only_its_preamble(mock_impl: Any) -> None:
    """KFP ships the function's source alone, so a module global it reads fails only
    on Vertex; execute() above runs in this module and would never notice."""
    program = COMPONENT.component_spec.implementation.container.command[-1]
    namespace: dict[str, Any] = {}
    exec(program, namespace)
    # Self-check: the namespace really lacks what the module defines.
    assert "_IMAGE" not in namespace
    composed_configs, bundle, registered = _artifacts()

    namespace["register_model"](
        serving_container_image_uri=SERVING_IMAGE,
        composed_configs=composed_configs,
        bundle=bundle,
        registered=registered,
    )

    assert registered.uri == SUMMARY.model_tag
