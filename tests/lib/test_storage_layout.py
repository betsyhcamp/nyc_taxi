from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from fcstnyctaxi.lib.config.bindings import environment_bindings
from fcstnyctaxi.lib.config.composition import compose_config
from fcstnyctaxi.lib.storage_layout import (
    build_environment_root,
    build_object_uri,
    build_prefix_uri,
    build_run_root,
    latest_pointer_path,
)
from fcstnyctaxi.lib.utils import get_project_root_dir
from fcstnyctaxi.schemas.config.common import SliceName
from fcstnyctaxi.schemas.config.environment import EnvironmentConfig
from fcstnyctaxi.schemas.storage.common import LATEST_POINTER_FILENAME

CONFIG_DIR = get_project_root_dir() / "config"
ENV = "dev"
RUN_ID = "RUNID"


@pytest.fixture
def composed_bucket() -> str:
    """The bucket `EnvironmentConfig` declares, read from the field by name.

    Derived, not pinned: a bucket change in dev.yaml must not fail these, while a
    resolver reading a different field must.
    """
    environment = cast(
        EnvironmentConfig, compose_config(CONFIG_DIR, environment_bindings(ENV)).config
    )
    return environment.storage.bucket_name


# ================================================
# Path builder tests
# ================================================

# Below the bucket root on purpose: every real call site passes a run root.
BASE = "gs://b/dev/"
BUILDERS = [build_prefix_uri, build_object_uri]


def test_a_run_root_spells_the_convention() -> None:
    """Test that build_run_root writes gs://<bucket>/<env>/<slice>/<run_id>/."""
    assert build_run_root("b", "dev", "train", "R") == "gs://b/dev/train/R/"


def test_an_environment_root_is_the_bucket_and_env_alone() -> None:
    """Test that build_environment_root writes gs://<bucket>/<env>/."""
    assert build_environment_root("b", "dev") == "gs://b/dev/"


def test_a_prefix_ends_in_a_slash_and_an_object_does_not() -> None:
    """Test that the builder's name, not the segments, decides the trailing slash."""
    assert build_prefix_uri(BASE, "backtest", "m") == "gs://b/dev/backtest/m/"
    assert build_object_uri(BASE, "run_output.json") == "gs://b/dev/run_output.json"


def test_an_object_uri_cannot_be_extended() -> None:
    """Test that joining under a file raises rather than nesting a path inside it."""
    with pytest.raises(ValueError, match="base"):
        build_prefix_uri(build_object_uri(BASE, "run_output.json"), "x")


@pytest.mark.parametrize("builder", BUILDERS)
@pytest.mark.parametrize(
    "segment",
    ["", " ", "a/b", ".", ".."],
    ids=["empty", "whitespace", "slash", "dot", "dot_dot"],
)
def test_a_segment_that_would_move_the_layout_raises(
    builder: Callable[..., str], segment: str
) -> None:
    """Test that a segment changing the depth or escaping the prefix is refused."""
    with pytest.raises(ValueError, match="segment"):
        builder(BASE, "ok", segment)


@pytest.mark.parametrize("builder", BUILDERS)
@pytest.mark.parametrize(
    "base",
    ["b/dev/", "s3://b/dev/", "gs://", "gs:///", "gs://b"],
    ids=["no_scheme", "other_scheme", "no_authority", "empty_authority", "no_slash"],
)
def test_a_base_that_is_not_a_gcs_prefix_raises(
    builder: Callable[..., str], base: str
) -> None:
    """Test that a base without scheme, authority or trailing slash is refused."""
    with pytest.raises(ValueError, match="base"):
        builder(base, "x")


@pytest.mark.parametrize("builder", BUILDERS)
def test_joining_no_segments_raises(builder: Callable[..., str]) -> None:
    """Test that no segments raises rather than returning the base as a new path."""
    with pytest.raises(ValueError, match="segments"):
        builder(BASE)


@pytest.mark.parametrize("bucket", ["", "b/dev", ".."], ids=["empty", "slash", "dot"])
def test_a_bucket_that_is_not_one_segment_raises(bucket: str) -> None:
    """Test that the bucket gets the segment checks the joiner gives every segment."""
    with pytest.raises(ValueError, match="bucket"):
        build_environment_root(bucket, "dev")


@pytest.mark.parametrize(
    ("env", "run_id"), [("a/b", "R"), ("dev", "a/b")], ids=["env", "run_id"]
)
def test_a_run_root_argument_that_adds_a_level_raises(env: str, run_id: str) -> None:
    """Test that env and run_id reach the segment checks through build_run_root."""
    with pytest.raises(ValueError, match="segment"):
        build_run_root("b", env, "train", run_id)


def test_build_run_root_raises_on_unknown_slice() -> None:
    """Test that a slice token outside SliceName raises instead of building a path."""
    with pytest.raises(ValueError, match="slice_name"):
        build_run_root("b", "dev", "training", "R")  # type: ignore[arg-type]


@pytest.mark.parametrize("slice_name", ["feature", "train", "inference"])
def test_every_slice_builds_a_run_root(slice_name: SliceName) -> None:
    """Training is one slice of three, so a guard admitting it alone fails two."""
    assert build_run_root("b", "dev", slice_name, "R") == f"gs://b/dev/{slice_name}/R/"


# ================================================
# latest_pointer_path tests
# ================================================


def test_the_pointer_sits_at_the_environment_root(
    composed_bucket: str, tmp_path: Path
) -> None:
    """Both execution modes read the layout back to the same environment root."""
    # The gcsfuse mount and the local mirror differ only above the bucket, so a
    # resolver reaching for either spelling would serve one mode and fail one.
    for base in (Path("/gcs"), tmp_path):
        run_dir = base / composed_bucket / ENV / "train" / RUN_ID

        assert latest_pointer_path(run_dir) == (
            base / composed_bucket / ENV / LATEST_POINTER_FILENAME
        )


@pytest.mark.parametrize(
    "run_dir",
    [Path(f"/{RUN_ID}"), Path("/a/b"), Path(f"/{ENV}/training/{RUN_ID}")],
    ids=["flat", "two_segments", "misspelled_slice"],
)
def test_a_run_dir_that_is_not_env_slice_run_raises(run_dir: Path) -> None:
    """The inverse refuses the slice token the forward builder refuses going in."""
    with pytest.raises(ValueError, match="is not <env>/<slice_name>/<run_id>"):
        latest_pointer_path(run_dir)


@pytest.mark.parametrize(
    "run_dir",
    [Path(f"train/{RUN_ID}"), Path(f"/train/{RUN_ID}")],
    ids=["relative", "rooted"],
)
def test_a_run_dir_with_no_environment_segment_raises(run_dir: Path) -> None:
    """Both pass the slice check and would put the pointer at the filesystem root."""
    with pytest.raises(ValueError, match="no environment segment"):
        latest_pointer_path(run_dir)
