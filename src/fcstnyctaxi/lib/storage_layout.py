"""One resolver for `gs://<bucket>/<env>/<slice_name>/<run_id>/<step>/`, so local vs
Vertex execution modes cannot drift. Not `lib/io.py`: composing a config is not IO.
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
RUN_OUTPUTS_FILENAME = "run_outputs.json"


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
    require_known_environment(config_dir, env)
    environment = cast(
        EnvironmentConfig, compose_config(config_dir, environment_bindings(env)).config
    )
    return _build_run_prefix(environment.storage.bucket_name, env, slice_name, run_id)


def resolve_run_outputs_uri(
    config_dir: Path, env: str, slice_name: SliceName, run_id: str
) -> str:
    """Where one run's outputs manifest lives: its run root, plus one filename.

    No step segment: a consumer must not need the producer's internal step names.
    Slice-generic, so a later `read_train_run_outputs` reuses it.
    """
    prefix = resolve_run_prefix(config_dir, env, slice_name, run_id)
    return f"{prefix}{RUN_OUTPUTS_FILENAME}"


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
    return f"gs://{bucket}/{env}/{slice_name}/{run_id}/"
