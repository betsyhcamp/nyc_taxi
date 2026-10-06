import json
from pathlib import PurePosixPath

import gcsfs
import streamlit as st
from fsspec import AbstractFileSystem

from dashboard.shared import load, paths


def discover_run_ids(slice_root: str, fs: AbstractFileSystem) -> tuple[str, ...]:
    """Run ids whose evaluate step finished, newest first.

    The gate is the evaluate manifest, not `run_output.json`. A run that
    evaluated and then failed at registration is one this monitor exists to
    surface.
    """
    hits = fs.glob(paths.evaluate_manifest_pattern(slice_root))
    # Two fixed segments follow the wildcard, and gcsfs strips the gs:// scheme
    # off what it returns, so the run directory is the grandparent.
    run_ids = {PurePosixPath(hit).parents[1].name for hit in hits}
    # Lexical order is chronological for generated ids only.
    return tuple(sorted(run_ids, key=str.lower, reverse=True))


def read_pointer_run_id(pointer_uri: str, fs: AbstractFileSystem) -> str | None:
    """The train run `_latest.json` names, or None when it names none.

    A slice nothing has written reads the string `"unset"`, where a record is
    expected.
    """
    if not fs.exists(pointer_uri):
        return None
    record = json.loads(fs.cat(pointer_uri)).get(paths.TRAIN_SLICE)
    if not isinstance(record, dict):
        return None
    return record.get("train_run_id")


def default_run_id(run_ids: tuple[str, ...], pointer_run_id: str | None) -> str | None:
    """The pointer's run when the listing offers it, else the newest, else None.

    The pointer can name a run the listing lacks: its evaluate output gone, or
    written after the listing was cached.
    """
    if pointer_run_id in run_ids:
        return pointer_run_id
    return run_ids[0] if run_ids else None


@st.cache_data(show_spinner=False, ttl=load.MUTABLE_TTL_SECONDS)
def list_run_ids(slice_root: str) -> tuple[str, ...]:
    """`discover_run_ids` against GCS, cached on the prefix."""
    return discover_run_ids(slice_root, gcsfs.GCSFileSystem())


@st.cache_data(show_spinner=False, ttl=load.MUTABLE_TTL_SECONDS)
def pointer_run_id(pointer_uri: str) -> str | None:
    """`read_pointer_run_id` against GCS, cached on the URI."""
    return read_pointer_run_id(pointer_uri, gcsfs.GCSFileSystem())
