"""One resolver for `gs://<bucket>/<env>/<slice_name>/<run_id>/<step>/` and the
`<env>/` root above it, so local vs Vertex execution modes cannot drift. Not
`lib/io.py`: composing a config is not IO.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import get_args

from fcstnyctaxi.lib.config.bindings import resolve_bucket
from fcstnyctaxi.schemas.config.common import SliceName
from fcstnyctaxi.schemas.storage.common import (
    LATEST_POINTER_FILENAME,
    RUN_OUTPUT_FILENAME,
)

GCS_SCHEME = "gs://"


def composed_config_filename(model_name: str) -> str:
    """The per-model config `compose_configs` emits and every later step reads.

    A function, not a format string per call site: a computed name can desynchronize.
    """
    return f"composed_config_{model_name}.yaml"


@dataclass(frozen=True)
class SourcedPath:
    """A filepath paired w/ the URI it represents. Can't check both are same object."""

    path: Path
    uri: str


def _validate_uri_segment(name: str, value: str) -> str:
    """Return `value`, refusing one that would change the layout's depth or escape."""
    if not value.strip():
        raise ValueError(f"{name} must be a non-empty segment, got {value!r}.")
    if "/" in value:
        raise ValueError(f"{name} must not contain '/', got {value!r}.")
    if value in (".", ".."):
        raise ValueError(f"{name} must not be a dot segment, got {value!r}.")
    return value


def _join_uri_segments(base: str, segments: tuple[str, ...]) -> str:
    """Join segments onto a gs:// prefix, adding no trailing "/" and stripping none:
    normalizing here would let a caller silently extend an object URI."""
    # A non-empty authority only: build_run_root passes a base below the bucket root.
    authority = base.removeprefix(GCS_SCHEME).split("/", 1)[0]
    if not base.startswith(GCS_SCHEME) or not authority or not base.endswith("/"):
        raise ValueError(
            f"base must start with {GCS_SCHEME!r}, have a non-empty authority and "
            f"end in '/', got {base!r}."
        )
    if not segments:
        raise ValueError(f"segments must not be empty, got none under {base!r}.")
    for segment in segments:
        _validate_uri_segment("segment", segment)
    return base + "/".join(segments)


def _bucket_root(bucket: str) -> str:
    """gs://<bucket>/, the one place the scheme is written into a URI."""
    return f"{GCS_SCHEME}{_validate_uri_segment('bucket', bucket)}/"


def _require_known_slice(slice_name: str) -> None:
    """Refuse a token outside SliceName at runtime: no type checker runs in CI."""
    known_slices = get_args(SliceName)
    if slice_name not in known_slices:
        raise ValueError(
            f"slice_name must be one of {known_slices}, got {slice_name!r}."
        )


def build_prefix_uri(base: str, *segments: str) -> str:
    """A folder URI under `base`, always ending in "/".

    Args:
        base: A gs:// prefix with a non-empty authority, ending in "/".
        *segments: At least one; none empty, containing "/", or a dot segment.

    Returns:
        str: `base` joined with `segments`, ending in "/".

    Raises:
        ValueError: `base` is not such a prefix, or a segment is missing or invalid.
    """
    return _join_uri_segments(base, segments) + "/"


def build_object_uri(base: str, *segments: str) -> str:
    """A file URI under `base`; the last segment is the object's name.

    Args:
        base: A gs:// prefix with a non-empty authority, ending in "/".
        *segments: At least one; none empty, containing "/", or a dot segment.

    Returns:
        str: `base` joined with `segments`, with no trailing "/".

    Raises:
        ValueError: `base` is not such a prefix, or a segment is missing or invalid.
    """
    return _join_uri_segments(base, segments)


def build_environment_root(bucket: str, env: str) -> str:
    """gs://<bucket>/<env>/, where one environment's shared files live.

    Above every slice, because what sits here is written by one pipeline and read
    by the others, so it cannot be run-scoped.

    Args:
        bucket: Bucket name, without the scheme.
        env: Deployment environment.

    Returns:
        str: The environment root, ending in "/".

    Raises:
        ValueError: `bucket` or `env` is not a single valid segment.
    """
    return build_prefix_uri(_bucket_root(bucket), env)


def build_run_root(bucket: str, env: str, slice_name: SliceName, run_id: str) -> str:
    """gs://<bucket>/<env>/<slice>/<run_id>/, where one run of one slice writes.

    Args:
        bucket: Bucket name, without the scheme.
        env: Deployment environment.
        slice_name: The pipeline this run belongs to.
        run_id: Per-run identifier; its format is checked where ids are minted.

    Returns:
        str: The run root, ending in "/".

    Raises:
        ValueError: `slice_name` is not a SliceName, or an argument is not a single
            valid segment.
    """
    _require_known_slice(slice_name)
    return build_prefix_uri(build_environment_root(bucket, env), slice_name, run_id)


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
    return f"{prefix}{RUN_OUTPUT_FILENAME}"


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
    """Delegates to `resolve_bucket` until the resolvers calling this are deleted."""
    return resolve_bucket(config_dir, env)
