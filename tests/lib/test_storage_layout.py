from pathlib import Path
from typing import cast

import pytest

from fcstnyctaxi.lib.config.bindings import available_environments, environment_bindings
from fcstnyctaxi.lib.config.composition import compose_config
from fcstnyctaxi.lib.storage_layout import (
    _build_run_prefix,
    resolve_run_outputs_uri,
    resolve_run_prefix,
)
from fcstnyctaxi.lib.utils import get_project_root_dir
from fcstnyctaxi.schemas.config.common import SliceName
from fcstnyctaxi.schemas.config.environment import EnvironmentConfig

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
# _build_run_prefix tests
# ================================================


def test_build_run_prefix_constructs_expected_string() -> None:
    """Test that _build_run_prefix writes the convention and returns a run root.

    The trailing "/" is load-bearing: upload_to_gcs rejects a destination without
    one, so the prefix composes with a step name by concatenation.
    """
    prefix = _build_run_prefix(
        bucket="BUCKET", env="dev", slice_name="train", run_id="RUNID"
    )
    assert prefix == "gs://BUCKET/dev/train/RUNID/"


def test_build_run_prefix_raises_on_unknown_slice() -> None:
    """Test that a slice token outside SliceName raises instead of building a path."""
    with pytest.raises(ValueError, match="slice_name"):
        _build_run_prefix(
            bucket="BUCKET",
            env="dev",
            slice_name="training",  # type: ignore[arg-type]
            run_id="RUNID",
        )


# ================================================
# resolve_run_prefix tests
# ================================================


@pytest.mark.parametrize("slice_name", ["feature", "train", "inference"])
def test_resolve_run_prefix_places_each_slice_under_the_composed_bucket(
    slice_name: SliceName, composed_bucket: str
) -> None:
    """Test that every slice resolves against the real tree, not just Training.

    Training is the only slice with project-owned static configs, so a resolver
    reaching for them would serve one caller and fail two.
    """
    prefix = resolve_run_prefix(CONFIG_DIR, ENV, slice_name, RUN_ID)

    assert prefix == f"gs://{composed_bucket}/{ENV}/{slice_name}/{RUN_ID}/"


def test_resolve_run_prefix_rejects_an_env_with_no_file() -> None:
    """Test that an unknown env raises ValueError naming the alternatives.

    Without `require_known_environment` this degrades to a FileNotFoundError quoting
    a container-absolute path. The guard only improves an error, so nothing else
    watches it.
    """
    with pytest.raises(ValueError, match="Unknown env 'bogus'; available:") as err:
        resolve_run_prefix(CONFIG_DIR, "bogus", "train", RUN_ID)

    # The set itself is not pinned: a new environments/*.yaml must not fail this.
    assert all(name in str(err.value) for name in available_environments(CONFIG_DIR))


def test_resolve_run_prefix_takes_a_config_dir_no_caller_has_to_compose(
    tmp_path: Path,
) -> None:
    """Test that the tree it reads is the one passed, not a discovered default.

    A default would pass every test that supplies the parameter, then read the wrong
    tree for a caller that does not.
    """
    with pytest.raises(ValueError, match="No environments are defined"):
        resolve_run_prefix(tmp_path, ENV, "train", RUN_ID)


# ================================================
# resolve_run_outputs_uri tests
# ================================================


@pytest.mark.parametrize("slice_name", ["feature", "train", "inference"])
def test_resolve_run_outputs_uri_names_the_run_root_not_a_step(
    slice_name: SliceName, composed_bucket: str
) -> None:
    """A step segment here would need the step name this file exists to supply."""
    uri = resolve_run_outputs_uri(CONFIG_DIR, ENV, slice_name, RUN_ID)

    assert uri == f"gs://{composed_bucket}/{ENV}/{slice_name}/{RUN_ID}/run_outputs.json"
