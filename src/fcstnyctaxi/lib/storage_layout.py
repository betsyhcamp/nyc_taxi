"""One resolver for `gs://<bucket>/<env>/<slice_name>/<run_id>/<step>/`, so local vs
Vertex execution modes cannot drift. Not `lib/io.py`: composing a config is not IO.
"""

from pathlib import Path
from typing import cast, get_args

from fcstnyctaxi.lib.config.bindings import (
    environment_bindings,
    require_known_environment,
)
from fcstnyctaxi.lib.config.composition import compose_config
from fcstnyctaxi.schemas.config.common import SliceName
from fcstnyctaxi.schemas.config.environment import EnvironmentConfig


def resolve_run_prefix(
    config_dir: Path, env: str, slice_name: SliceName, run_id: str
) -> str:
    """Where one run of one slice writes. The single place the layout is resolved.

    Takes `config_dir`, not a bucket: the bucket lives in `EnvironmentConfig`, so a
    bucket parameter pushes composition into every caller and those copies drift.
    `require_known_environment` turns an unknown env into a domain error rather than
    a `FileNotFoundError`; a test watches it, since it only improves an error.

    Args:
        config_dir: Root of the config tree.
        env: Deployment environment; needs an `environments/<env>.yaml`.
        slice_name: The pipeline this run belongs to.
        run_id: Per-run identifier, giving each run its own directory.

    Returns:
        Fully-qualified GCS prefix, ending in "/".

    Raises:
        ValueError: `env` has no config file, `slice_name` is not a SliceName, or
            the environment fragment fails to compose.
    """
    require_known_environment(config_dir, env)
    environment = cast(
        EnvironmentConfig, compose_config(config_dir, environment_bindings(env)).config
    )
    return _build_run_prefix(environment.storage.bucket_name, env, slice_name, run_id)


def _build_run_prefix(bucket: str, env: str, slice_name: SliceName, run_id: str) -> str:
    """Construct the run root gs://<bucket>/<env>/<slice_name>/<run_id>/.

    The convention itself, plus the slice guard. Pure, so its tests are a literal;
    private, since every caller reaches `resolve_run_prefix`. Ends in "/" because
    each step appends its own name and never accepts a full output path. The typed
    slice puts the notebooks' dev/experiments/<run_id>/ out of reach, a deliberate
    narrowing: the convention's guarantor should not also mint scratch namespaces.

    Args:
        bucket: GCS bucket name, without the gs:// scheme.
        env: Deployment environment, e.g. "dev". Not checked here;
            `resolve_run_prefix` guards it before calling in.
        slice_name: The pipeline that produced the artifacts.
        run_id: Per-run identifier, giving each run its own directory.

    Returns:
        Fully-qualified GCS prefix, ending in "/".

    Raises:
        ValueError: slice_name is not a SliceName. Raises rather than asserts,
            since python -O strips asserts and no type checker runs in CI;
            "training" for "train" would give a well-formed wrong path.
    """
    known_slices = get_args(SliceName)
    if slice_name not in known_slices:
        raise ValueError(
            f"slice_name must be one of {known_slices}, got {slice_name!r}."
        )
    return f"gs://{bucket}/{env}/{slice_name}/{run_id}/"
