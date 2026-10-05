"""Every read the page makes, cached on plain strings.

Streamlit reruns the whole script on each widget change, so the cache does real work
"""

import json
import tomllib
from pathlib import Path
from typing import Any

import fsspec
import pandas as pd
import streamlit as st
from fsspec.core import url_to_fs

from dashboard.shared import paths

_REQUIRED_SETTINGS = ("env", "config_dir")

# For the run listing, the pointer and the registration marker, which change under
# a fixed URI. The cache belongs to the server process, so a browser refresh never
# clears it. A run's tables do not change under its id and stay at ttl=None.
MUTABLE_TTL_SECONDS = 60


@st.cache_data(show_spinner=False)
def dashboard_config(config_path: str) -> dict[str, str]:
    """`env` and `config_dir`, with a relative `config_dir` made absolute.

    Resolved against the file's own project root, not the working directory:
    Streamlit can be launched from anywhere, and a config tree that fails to
    resolve composes a different environment or none at all.
    """
    path = Path(config_path)
    settings = tomllib.loads(path.read_text())
    absent = [key for key in _REQUIRED_SETTINGS if key not in settings]
    if absent:
        raise ValueError(
            f"{path} supplies no {absent}, and every prefix the dashboard reads "
            "resolves from those two, so a default here would quietly open an "
            "environment the file does not name."
        )

    config_dir = Path(str(settings["config_dir"]))
    if not config_dir.is_absolute():
        config_dir = path.resolve().parents[1] / config_dir
    return {"env": str(settings["env"]), "config_dir": str(config_dir)}


@st.cache_data(show_spinner=False)
def slice_root_uri(config_dir: str, env: str) -> str:
    """Where the run lister looks."""
    return paths.train_slice_root(config_dir, env)


@st.cache_data(show_spinner=False)
def pointer_uri(config_dir: str, env: str) -> str:
    """`_latest.json` at the environment root."""
    return paths.pointer_uri(config_dir, env)


@st.cache_data(show_spinner=False)
def evaluate_uri(config_dir: str, env: str, run_id: str) -> str:
    """One run's evaluate directory, the prefix every table read below extends."""
    return paths.evaluate_prefix(config_dir, env, run_id)


@st.cache_data(show_spinner=False)
def run_outputs_uri(config_dir: str, env: str, run_id: str) -> str:
    """The pipeline's completion marker, read only for the selected run."""
    return paths.run_outputs_uri(config_dir, env, run_id)


@st.cache_data(show_spinner=False)
def load_evaluate_manifest(prefix: str) -> dict[str, Any]:
    """The whole manifest. Every field of the identity strip comes from it."""
    return _read_json(f"{prefix}{paths.EVALUATE_MANIFEST}")


@st.cache_data(show_spinner=False, ttl=MUTABLE_TTL_SECONDS)
def load_run_output(uri: str) -> dict[str, Any] | None:
    """The completion marker, or None when the run never registered.

    Absence is information, not an error. Returned unvalidated: the schema
    forbids extras, so validating would let a field the dashboard never reads
    blank the page.
    """
    fs, path = url_to_fs(uri)
    return _read_json(uri) if fs.exists(path) else None


@st.cache_data(show_spinner=False)
def load_summary_metrics(prefix: str) -> pd.DataFrame:
    """Scorecard grain."""
    return pd.read_parquet(f"{prefix}{paths.SUMMARY_METRICS}")


@st.cache_data(show_spinner=False)
def load_fold_metrics(prefix: str) -> pd.DataFrame:
    """Fold grain. The largest tableread."""
    return pd.read_parquet(f"{prefix}{paths.FOLD_METRICS}")


def check_lineage(
    run_id: str, manifest: dict[str, Any], frames: dict[str, pd.DataFrame]
) -> None:
    """Raise unless the manifest and every table name the selected run.

    `evaluate_impl` never files a table under another run's id, so this guards a
    manual copy or edit, which the loaders accept.

    Args:
        run_id: The run the selector chose.
        manifest: Its evaluate manifest.
        frames: Each loaded table, keyed by the name an error reports.

    Raises:
        ValueError: The manifest or any table names a run other than `run_id`.
    """
    declared = manifest.get("lineage", {}).get("train_run_id")
    if declared != run_id:
        raise ValueError(
            f"the evaluate manifest under run {run_id!r} names {declared!r}, so the "
            "identity strip would describe a run other than the one selected."
        )
    for name, frame in frames.items():
        stamped = set(frame["train_run_id"].unique())
        if stamped != {run_id}:
            raise ValueError(
                f"{name} under run {run_id!r} is stamped "
                f"{sorted(str(value) for value in stamped)}, so the "
                "page would chart another run's rows under this run's name."
            )


def _read_json(uri: str) -> dict[str, Any]:
    """One JSON object, over whichever filesystem the URI names."""
    with fsspec.open(uri) as handle:
        return json.load(handle)
