from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping, NamedTuple

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

    Args:
        bucket: GCS bucket name, without the gs:// scheme.
        prefix: Path prefix within the bucket, e.g. "snapshots/source".
        run_id: Per-run identifier placed between prefix and filename,
            giving each run its own directory.
        filename: Final path segment, e.g.
            "manhattan_daily_zone_pickups.parquet".

    Returns:
        Fully-qualified GCS URI
    """
    return f"gs://{bucket}/{prefix}/{run_id}/{filename}"
