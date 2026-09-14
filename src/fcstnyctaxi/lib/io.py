"""IO for pipeline artifacts: SQL preparation, GCS URI construction, and transfer.

Prefixes end in "/" and objects do not, which is fsspec's rule and decides an
upload's layout. write_<thing>_to_gcs serialises a value; the rest move bytes.

build_run_scoped_uri is transitional and serves an older layout temporarily; delete
when ingress pipeline conforms.
"""

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

import fsspec
from tsbricks.blocks.dataio import read_sql, render_sql_template


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

    Transitional: this serves the ingress pipeline's older prefix-first layout.
    Delete it when that pipeline adopts storage_layout.resolve_run_prefix.

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


def _require_gcs_uri(gcs_uri: str) -> None:
    """Reject any URI that is not gs://, at the call site rather than downstream.

    fsspec is protocol-agnostic; an s3:// URI otherwise fails inside pydantic.
    """
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"gcs_uri must be a gs://... string, got {gcs_uri!r}.")


def _require_gcs_prefix(gcs_uri: str) -> None:
    """Reject a destination that is not a gs:// prefix ending in "/".

    The slash marks a prefix, not an object name; normalising hides the error.
    """
    _require_gcs_uri(gcs_uri)
    if not gcs_uri.endswith("/"):
        raise ValueError(
            f"gcs_uri must be a destination prefix ending in '/', got {gcs_uri!r}."
        )


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
    _require_gcs_prefix(gcs_uri)

    fs, path = fsspec.url_to_fs(gcs_uri)
    if not local_path.is_dir():
        fs.put(str(local_path), path)
        return 1

    fs.put(f"{local_path}/", path, recursive=True)
    return sum(1 for child in local_path.rglob("*") if child.is_file())


def sync_to_gcs(
    local_dir: Path, gcs_uri: str, completion_marker: str | None = None
) -> tuple[int, int]:
    """Publish a directory to a GCS prefix so the prefix holds exactly its contents.

    Unlike upload_to_gcs it deletes: uploads, then removes remote objects the
    directory lacks, by relative path so nested trees reconcile. The marker goes
    alone and last, so its presence means complete rather than merely written.

    Args:
        local_dir: Directory to publish. Its contents are placed under gcs_uri;
            its basename is not repeated.
        gcs_uri: Destination prefix, ending in "/", e.g.
            "gs://bucket/dev/train/RUNID/compose_configs/".
        completion_marker: Name of the file in local_dir whose presence means
            the prefix is complete. None publishes no marker.

    Returns:
        (objects uploaded, remote objects removed). The marker counts as an
        upload; deleting it in order to republish it is not a removal.

    Raises:
        ValueError: gcs_uri is not a gs:// prefix ending in "/", or
            completion_marker holds a path rather than a filename, which would
            key the marker differently from every body object.
        NotADirectoryError: local_dir is not a directory. An absent one lists
            empty, which would reconcile the prefix to nothing.
        FileNotFoundError: completion_marker names no file in local_dir, which
            would otherwise publish the real marker inside the batch.
    """
    # Ordered ahead of every remote call: this deletes before it writes, and a
    # bare local path resolves to LocalFileSystem.
    _require_gcs_prefix(gcs_uri)
    if not local_dir.is_dir():
        raise NotADirectoryError(
            f"local_dir must be an existing directory, got {str(local_dir)!r}."
        )
    if completion_marker and Path(completion_marker).name != completion_marker:
        raise ValueError(
            f"completion_marker {completion_marker!r} must be a filename in "
            "local_dir, not a path."
        )
    marker = None if completion_marker is None else local_dir / completion_marker
    if marker is not None and not marker.is_file():
        raise FileNotFoundError(
            f"completion_marker {completion_marker!r} names no file in {local_dir}."
        )

    fs, remote_root = fsspec.url_to_fs(gcs_uri)
    remote_root = remote_root.rstrip("/")
    prefix = f"{remote_root}/"
    if marker is not None and fs.exists(f"{prefix}{completion_marker}"):
        fs.rm_file(f"{prefix}{completion_marker}")

    local_files = sorted(p for p in local_dir.rglob("*") if p.is_file())
    body = [p for p in local_files if p != marker]
    # Explicit destinations rather than a prefix: fsspec takes matching lists
    # verbatim, so nesting needs no path arithmetic and no basename is appended.
    fs.put(
        [str(p) for p in body],
        [f"{prefix}{p.relative_to(local_dir).as_posix()}" for p in body],
    )

    published = {p.relative_to(local_dir).as_posix() for p in local_files}
    # startswith: fs.find can return an object named exactly like remote_root,
    # which is a sibling of the prefix rather than something under it.
    stale = [
        obj
        for obj in fs.find(remote_root)
        if obj.startswith(prefix) and obj.removeprefix(prefix) not in published
    ]
    if stale:
        fs.rm(stale)

    if marker is not None:
        # Alone, after the reconcile: a batch upload runs concurrently, so a
        # marker sent with the body can land while a sibling is still missing.
        fs.put_file(str(marker), f"{prefix}{completion_marker}")

    return len(local_files), len(stale)
