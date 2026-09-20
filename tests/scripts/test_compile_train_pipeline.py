import os

# Precedes the imports: @dsl.component binds base_image at import time, and an unset
# value aborts collection for this whole module.
os.environ.setdefault(
    "FCST_TRAIN_IMAGE",
    "us-central1-docker.pkg.dev/nyc-taxi-ehc/fcst-ml-containers/train@sha256:"
    + "a" * 64,
)

from pathlib import Path
from typing import Any

import pytest
import yaml
from pytest_mock import MockerFixture

from fcstnyctaxi.lib.utils import get_project_root_dir
from scripts import compile_train_pipeline

PROJECT_ROOT = get_project_root_dir()

REPO = "us-central1-docker.pkg.dev/nyc-taxi-ehc/fcst-ml-containers"
EXPECTED = f"{REPO}/train@sha256:" + "d" * 64
NESTED = f"{REPO}/team/train@sha256:" + "d" * 64
# A second Training digest, as a re-push before a fresh compile would leave.
STALE = f"{REPO}/train@sha256:" + "e" * 64
# One Docker repository per project makes another slice a sibling under one prefix.
CROSS_SLICE = f"{REPO}/inference@sha256:" + "f" * 64
PLATFORM = "gcr.io/ml-pipeline/kfp-launcher:2.15.2"


def _ir(*images: str) -> dict[str, Any]:
    """A deployment spec: one container executor per image, plus two importers."""
    executors: dict[str, Any] = {
        f"exec-step-{i}": {"container": {"image": image}}
        for i, image in enumerate(images)
    }
    executors["exec-importer"] = {"importer": {"artifactUri": {}}}
    executors["exec-importer-2"] = {"importer": {"artifactUri": {}}}
    return {"deploymentSpec": {"executors": executors}}


def test_platform_images_and_importers_need_no_allowlist() -> None:
    """Test that importers are skipped and an image outside the prefix is ignored.

    Enumerating Google's images instead would churn on every kfp bump.
    """
    ir = _ir(EXPECTED, PLATFORM)

    assert compile_train_pipeline._executor_images(ir) == {EXPECTED, PLATFORM}
    compile_train_pipeline._require_pinned_project_images(ir, EXPECTED)


@pytest.mark.parametrize(
    ("images", "expected"),
    [
        ((PLATFORM,), EXPECTED),
        ((EXPECTED, STALE), EXPECTED),
        ((EXPECTED, CROSS_SLICE), EXPECTED),
        ((NESTED, CROSS_SLICE), NESTED),
    ],
    ids=["absent", "stale-digest", "cross-slice", "nested-plus-cross-slice"],
)
def test_project_owned_images_must_equal_the_expected_digest(
    images: tuple[str, ...], expected: str
) -> None:
    """Test the four faults that reach here: absent, stale, cross-slice, and the same
    cross-slice under a nested name, which a last-segment parse stops classifying."""
    with pytest.raises(RuntimeError, match="must be exactly"):
        compile_train_pipeline._require_pinned_project_images(_ir(*images), expected)


def test_compile_builds_the_real_pipeline_and_pins_its_image(tmp_path: Path) -> None:
    """Test _compile end to end against a real compile, writing outside the tree.

    The specs above are hand-built, so without this they would pass unchanged if kfp
    moved container.image and the check stopped reading anything real. The two roots
    are separated here for the reason they exist: tmp_path has no config tree to
    read the model set from.
    """
    template = tmp_path / "build" / "fcst-train-pipeline.yaml"

    compile_train_pipeline._compile(PROJECT_ROOT, template)

    ir = yaml.safe_load(template.read_text())
    assert compile_train_pipeline._executor_images(ir) == {
        os.environ["FCST_TRAIN_IMAGE"]
    }


def test_compile_removes_the_template_when_validation_fails(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    """Test that a rejected spec is not left on disk.

    compile() has already overwritten any previous template by then, so leaving the
    rejected one is the state a later --template-path would submit successfully.
    """
    template = tmp_path / "build" / "fcst-train-pipeline.yaml"
    mocker.patch.object(
        compile_train_pipeline,
        "_require_pinned_project_images",
        side_effect=RuntimeError("rejected"),
    )

    with pytest.raises(RuntimeError, match="rejected"):
        compile_train_pipeline._compile(PROJECT_ROOT, template)

    assert not template.exists()
