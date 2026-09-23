"""One resolver for `gs://<bucket>/<env>/<slice_name>/<run_id>/<step>/` and the
`<env>/` root above it, so local vs Vertex execution modes cannot drift. Not
`lib/io.py`: composing a config is not IO.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import cast, get_args

from fcstnyctaxi.lib.config.bindings import (
    environment_bindings,
    require_known_environment,
)
from fcstnyctaxi.lib.config.composition import compose_config
from fcstnyctaxi.schemas.config.common import SliceName
from fcstnyctaxi.schemas.config.environment import EnvironmentConfig

# One place for the name a producer writes and a consumer reads.
RUN_OUTPUTS_FILENAME = "run_output.json"

LATEST_POINTER_FILENAME = "_latest.json"
"""The environment-root pointer every slice rewrites its own key in."""


def composed_config_filename(model_name: str) -> str:
    """The per-model config `compose_configs` emits and every later step reads.

    A function, not a format string per call site: a computed name can desynchronize.
    """
    return f"composed_config_{model_name}.yaml"


BUNDLE_MODEL_DIR_NAME = "model"
"""The bundle's subdirectory the save callable owns, apart from the impl's files."""


@dataclass(frozen=True)
class SourcedPath:
    """A filepath paired w/ the URI it represents. Can't check both are same object."""

    path: Path
    uri: str


def resolve_run_prefix(
    config_dir: Path, env: str, slice_name: SliceName, run_id: str
) -> str:
    """Where one run of one slice writes, ending in "/". The layout's one resolver.

    Takes `config_dir`, not a bucket: the bucket lives in `EnvironmentConfig`, so a
    bucket parameter pushes composition into every caller and those copies drift.
    `require_known_environment` turns an unknown env into a domain error rather
    than a `FileNotFoundError`.

    Args:
        config_dir: Root of the config tree.
        env: Deployment environment; needs an `environments/<env>.yaml`.
        slice_name: The pipeline this run belongs to.
        run_id: Per-run identifier, giving each run its own directory.

    Raises:
        ValueError: `env` has no config file, `slice_name` is not a SliceName, or
            the fragment fails to compose.
    """
    return _build_run_prefix(_composed_bucket(config_dir, env), env, slice_name, run_id)


def resolve_run_outputs_uri(
    config_dir: Path, env: str, slice_name: SliceName, run_id: str
) -> str:
    """Where one run's outputs manifest lives: its run root, plus one filename.

    No step segment: a consumer must not need the producer's internal step names.
    Slice-generic, so a later `read_train_run_outputs` reuses it.
    """
    prefix = resolve_run_prefix(config_dir, env, slice_name, run_id)
    return f"{prefix}{RUN_OUTPUTS_FILENAME}"


def resolve_environment_root(config_dir: Path, env: str) -> str:
    """Where one environment's shared files live, ending in "/".

    Above every slice, because what sits here is written by one pipeline and read
    by the others, so it cannot be run-scoped.

    Args:
        config_dir: Root of the config tree.
        env: Deployment environment; needs an `environments/<env>.yaml`.

    Raises:
        ValueError: `env` has no config file, or the fragment fails to compose.
    """
    return _build_environment_root(_composed_bucket(config_dir, env), env)


def resolve_latest_pointer_uri(config_dir: Path, env: str) -> str:
    """The run pointer's object URI: the environment root, plus one filename.

    No slice segment: one file carries a key per slice, so a slice token here
    would strand the other two.
    """
    return f"{resolve_environment_root(config_dir, env)}{LATEST_POINTER_FILENAME}"


def latest_pointer_path(run_dir: Path) -> Path:
    """The pointer beside one run's root, the layout read backwards.

    Two checks, because the slice token alone is not enough: `train/RUNID` passes
    it and would resolve to `_latest.json` in the working directory. The slice
    check runs first, which is what makes `parents[1]` safe to index.

    Args:
        run_dir: One run's root, `<env>/<slice_name>/<run_id>`, mounted or mirrored.

    Raises:
        ValueError: run_dir is not <env>/<slice_name>/<run_id>.
    """
    known_slices = get_args(SliceName)
    if run_dir.parent.name not in known_slices:
        raise ValueError(
            f"run_dir {run_dir} is not <env>/<slice_name>/<run_id>: its parent is "
            f"{run_dir.parent.name!r}, not one of {known_slices}."
        )
    if not run_dir.parents[1].name:
        raise ValueError(
            f"run_dir {run_dir} has no environment segment above "
            f"{run_dir.parent.name!r}, so it names no environment root."
        )
    return run_dir.parents[1] / LATEST_POINTER_FILENAME


def _build_run_prefix(bucket: str, env: str, slice_name: SliceName, run_id: str) -> str:
    """Construct the run root gs://<bucket>/<env>/<slice_name>/<run_id>/.

    The convention itself, plus the slice guard. Private, since every caller
    reaches `resolve_run_prefix`, which guards `env` before calling in. Ends in
    "/" because each step appends its own name. The typed slice puts the
    notebooks' dev/experiments/<run_id>/ out of reach: the convention's guarantor
    should not also mint scratch namespaces.

    Raises:
        ValueError: slice_name is not a defined SliceName.
    """
    known_slices = get_args(SliceName)
    if slice_name not in known_slices:
        raise ValueError(
            f"slice_name must be one of {known_slices}, got {slice_name!r}."
        )
    return f"{_build_environment_root(bucket, env)}{slice_name}/{run_id}/"


def _build_environment_root(bucket: str, env: str) -> str:
    """Construct gs://<bucket>/<env>/, the prefix every slice sits under."""
    return f"gs://{bucket}/{env}/"


def _composed_bucket(config_dir: Path, env: str) -> str:
    """The bucket `EnvironmentConfig` declares for `env`, guarding the env first."""
    require_known_environment(config_dir, env)
    environment = cast(
        EnvironmentConfig, compose_config(config_dir, environment_bindings(env)).config
    )
    return environment.storage.bucket_name
