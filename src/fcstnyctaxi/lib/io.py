"""IO for pipeline artifacts: SQL preparation, GCS URI construction, and transfer.

Prefixes end in "/", objects do not which is fsspec's own rule, and an upload's layout
depends on that slash. build_run_prefix returns a prefix, as upload_to_gcs
requires; build_run_scoped_uri returns an object URI, as download_from_gcs and
write_text_to_gcs take.

write_<thing>_to_gcs serialises an in-memory value. download_from_gcs and
upload_to_gcs move bytes that already exist, without interpreting them.

build_run_scoped_uri is transitional and serves an older layout temporarily; delete
when ingress pipeline conforms.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple, get_args

import fsspec
from tsbricks.blocks.dataio import read_sql, render_sql_template

from fcstnyctaxi.schemas.config.common import SliceName


class PreparedSql(NamedTuple):
    sql_text: str
    sha256: str


def prepare_sql(sql_path: Path, sql_params: Mapping[str, object]) -> PreparedSql:
    """Read SQL, render Jinja placeholders, and return text plus a SHA-256 fingerprint.

    Always invokes render_sql_template; mismatched params raise ValueError.

    Args:
        sql_path: Path to a .sql file, optionally containing Jinja placeholders.
        sql_params: Placeholder names to values. Must exactly match the file.

    Returns:
        PreparedSql with rendered text and its SHA-256 hex digest.

    Raises:
        FileNotFoundError: sql_path does not exist.
        ValueError: sql_params do not match the file's placeholders.
    """
    text = read_sql(sql_path=sql_path)
    text = render_sql_template(text, sql_params)
    sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return PreparedSql(sql_text=text, sha256=sha256)


def build_run_scoped_uri(bucket: str, prefix: str, run_id: str, filename: str) -> str:
    """Construct a run-scoped GCS URI for a pipeline artifact where URI has
    format: gs://<bucket>/<prefix>/<run_id>/<filename>.

    Args:
        bucket: GCS bucket name, without the gs:// scheme.
        prefix: Path prefix within the bucket, e.g. "snapshots/source".
        run_id: Per-run identifier placed between prefix and filename,
            giving each run its own directory.
        filename: Final path segment, e.g.
            "<your_filename>.parquet".

    Returns:
        Fully-qualified GCS URI
    """
    return f"gs://{bucket}/{prefix}/{run_id}/{filename}"


def build_run_prefix(bucket: str, env: str, slice_name: SliceName, run_id: str) -> str:
    """Construct the run root gs://<bucket>/<env>/<slice_name>/<run_id>/.

    The single place the storage convention is written. It returns the run
    *root*, ending in "/", because each component appends its own step name and
    never accepts a full output path since no step can write into another's
    directory.

    Now that the slice segment is typed, the notebooks' dev/experiments/<run_id>/
    root is out of reach. The function guaranteeing the production convention
    should not also mint disposable namespaces.

    Args:
        bucket: GCS bucket name, without the gs:// scheme.
        env: Deployment environment, e.g. "dev". Not checked here —
            require_known_environment validates it in the config layer.
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


def _require_gcs_uri(gcs_uri: str) -> None:
    """Reject any URI that is not gs://, at the call site rather than downstream.

    fsspec is protocol-agnostic, but this project's contracts are GCP-specific.
    Without this guard an s3:// URI  works here and fails later inside a
    Pydantic pattern check. Follows tsbricks' _check_storage_uri_str.

    Args:
        gcs_uri: The URI to validate. Both an object URI and a prefix ending
            in "/" are accepted; this checks the scheme only.

    Raises:
        ValueError: gcs_uri does not start with "gs://".
    """
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"gcs_uri must be a gs://... string, got {gcs_uri!r}.")


def write_text_to_gcs(text: str, gcs_uri: str) -> None:
    """Write text to GCS URI as utf-8

    This general purpose IO function is a candidate for promotion to
    tsbricks once the API stabilizes.

    Args:
      text: The string to be written to GCS as a text file.
      gcs_uri: The string GCS URI of the form "gs://.. .txt"
    """
    fs, path = fsspec.url_to_fs(gcs_uri)
    with fs.open(path, mode="w") as f:
        f.write(text)


def download_from_gcs(gcs_uri: str, destination_dir: Path) -> Path:
    """Download a single GCS object into destination_dir, returning its local path.

    Half of the local substitute for the gcsfuse mount: pathlib.Path is not an
    abstraction over remote storage, so a function can be handed a Path for a GCS
    object only by a driver that is misleading about where the bytes are (gcsfuse,
    on Vertex) or by putting the bytes on local disk. This exists so the local
    runner makes the identical call Vertex makes.

    Args:
        gcs_uri: Object URI, e.g. "gs://bucket/inputs/panel.parquet".
        destination_dir: Local directory to download into, created if missing.

    Returns:
        destination_dir / the URI's final segment.

    Raises:
        ValueError: gcs_uri is not a gs:// URI, or names a prefix rather than an
            object. Inbound directory support needs prefix listing and
            relative-path reconstruction.
    """
    _require_gcs_uri(gcs_uri)

    fs, path = fsspec.url_to_fs(gcs_uri)
    if fs.isdir(path):
        raise ValueError(
            f"gcs_uri must name a single object, got the prefix {gcs_uri!r}."
        )

    destination_dir.mkdir(parents=True, exist_ok=True)
    dest_file = destination_dir / gcs_uri.rsplit("/", 1)[-1]
    fs.get_file(path, str(dest_file))
    return dest_file


def upload_to_gcs(local_path: Path, gcs_uri: str) -> int:
    """Upload a file or directory to a GCS prefix, returning the objects written.

    Half of the local substitute for the gcsfuse mount; see download_from_gcs
    for why the pair exists. It would disappear if the bucket were mounted
    locally.

    A directory's *contents* land under gcs_uri; its basename is not repeated.
    The trailing "/" appended to the source is what guarantees that since fsspec
    appends the source basename unless the source ends in a separator. Existing
    objects are overwritten, and parent prefixes are implicit.

    Args:
        local_path: File or directory to upload. A directory is uploaded
            recursively.
        gcs_uri: Destination prefix, ending in "/", e.g.
            "gs://bucket/dev/train/RUNID/compose_configs/".

    Returns:
        Number of objects written: 1 for a file, the file count for a directory.

    Raises:
        ValueError: gcs_uri is not a gs:// URI, or does not end in "/". The
            missing slash is rejected rather than normalised because the
            mistake is otherwise invisible.
    """
    _require_gcs_uri(gcs_uri)
    if not gcs_uri.endswith("/"):
        raise ValueError(
            f"gcs_uri must be a destination prefix ending in '/', got {gcs_uri!r}."
        )

    fs, path = fsspec.url_to_fs(gcs_uri)
    if not local_path.is_dir():
        fs.put(str(local_path), path)
        return 1

    fs.put(f"{local_path}/", path, recursive=True)
    return sum(1 for child in local_path.rglob("*") if child.is_file())
