"""Every read the page makes, cached on plain strings.

Streamlit reruns the whole script on each widget change, so the cache does real work.
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
def sidecar_uri(config_dir: str, env: str, run_id: str, model_name: str) -> str:
    """One model's backtest sidecar, the prefix every sidecar read below extends."""
    return paths.sidecar_prefix(config_dir, env, run_id, model_name)


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
    """Fold grain."""
    return pd.read_parquet(f"{prefix}{paths.FOLD_METRICS}")


@st.cache_data(show_spinner=False)
def load_per_series_comparison(prefix: str) -> pd.DataFrame:
    """Series by fold, both forecasts on one row. The largest table read, and the
    only one growing with the series count."""
    return pd.read_parquet(f"{prefix}{paths.PER_SERIES_COMPARISON}")


@st.cache_data(show_spinner=False)
def load_backtest_manifest(prefix: str) -> dict[str, Any]:
    """A sidecar's manifest, the only file in it naming its run and model.

    Raises:
        FileNotFoundError: The sidecar holds none. Backtest writes it last, so
            that backtest never finished, or a rerun failed partway.
    """
    try:
        return _read_json(f"{prefix}{paths.BACKTEST_MANIFEST}")
    except FileNotFoundError as error:
        # gcsfs names the missing object percent-encoded: b/<bucket>/o/dev%2Ftrain...
        raise FileNotFoundError(
            f"the sidecar {prefix} holds no {paths.BACKTEST_MANIFEST}, which backtest "
            "writes last, so none of its files can be trusted."
        ) from error


@st.cache_data(show_spinner=False)
def load_fiscal_calendar(prefix: str) -> pd.DataFrame:
    """The calendar backtest copied into a sidecar, the same in every model's."""
    return pd.read_parquet(f"{prefix}{paths.FISCAL_CALENDAR}")


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


def check_sidecar_lineage(run_id: str, manifests: dict[str, dict[str, Any]]) -> None:
    """Raise unless each sidecar's manifest names the selected run and the model
    whose directory it was read from.

    The sidecar's tables carry no run id, so its manifest is their only witness.
    The pipeline cannot cross or swap sidecars, so this guards a manual copy or
    rename. Swapped directories are the worse case: a complete, plausible page
    labeling the benchmark's forecasts as the challenger's.

    Args:
        run_id: The run the selector chose.
        manifests: Each sidecar's manifest, keyed by the model its path names.

    Raises:
        ValueError: A manifest names another run or another model.
    """
    for model, manifest in manifests.items():
        declared_run = manifest.get("lineage", {}).get("train_run_id")
        if declared_run != run_id:
            raise ValueError(
                f"the {model!r} sidecar under run {run_id!r} names run "
                f"{declared_run!r}, so the page would chart another run's forecasts."
            )
        declared_model = manifest.get("model_name")
        if declared_model != model:
            raise ValueError(
                f"the {model!r} sidecar under run {run_id!r} names model "
                f"{declared_model!r}, so the page would label one model's forecasts "
                "as the other's."
            )


def _read_json(uri: str) -> dict[str, Any]:
    """One JSON object, over whichever filesystem the URI names."""
    with fsspec.open(uri) as handle:
        return json.load(handle)
