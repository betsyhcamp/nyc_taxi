from pathlib import Path

import streamlit as st

from fcstnyctaxi.lib.config.bindings import resolve_bucket
from fcstnyctaxi.lib.storage_layout import (
    build_environment_root,
    build_object_uri,
    build_prefix_uri,
    build_run_root,
)
from fcstnyctaxi.schemas.config.common import SliceName
from fcstnyctaxi.schemas.storage.common import (
    LATEST_POINTER_FILENAME,
    RUN_OUTPUT_FILENAME,
)
from fcstnyctaxi.schemas.storage.train import BACKTEST_DIR, EVALUATE_DIR

# Typed, so a token outside SliceName is a type error rather than a 404. Also
# the key `runs.py` reads `_latest.json` by, which holds one record per slice.
TRAIN_SLICE: SliceName = "train"

# Objects in the evaluate directory. Pinned to the producer's own names by test,
# since this is the fourth place they are written down.
EVALUATE_MANIFEST = "evaluate_manifest.json"
SUMMARY_METRICS = "summary_metrics.parquet"
FOLD_METRICS = "fold_metrics.parquet"


@st.cache_data(show_spinner=False)
def _bucket(config_dir: str, env: str) -> str:
    """The one config read, so every path function below is pure string work.
    `str`, not `Path`, throughout: every `Path` conversion lives in this module."""
    return resolve_bucket(Path(config_dir), env)


def train_slice_root(config_dir: str, env: str) -> str:
    """What the run lister enumerates, so above the run id rather than at it.

    Args:
        config_dir: Root of the config tree.
        env: Needs an `environments/<env>.yaml`.

    Raises:
        ValueError: Unknown `env`, or the fragment fails to compose.

    Returns:
        str: `gs://<bucket>/<env>/train/`.
    """
    environment_root = build_environment_root(_bucket(config_dir, env), env)
    return build_prefix_uri(environment_root, TRAIN_SLICE)


def evaluate_prefix(config_dir: str, env: str, run_id: str) -> str:
    """One run's evaluate step directory.

    Args:
        config_dir: Root of the config tree.
        env: Needs an `environments/<env>.yaml`.
        run_id: The train run to read.

    Raises:
        ValueError: Unknown `env`, or the fragment fails to compose.

    Returns:
        str: `gs://<bucket>/<env>/train/<run_id>/evaluate/`.
    """
    run_root = build_run_root(_bucket(config_dir, env), env, TRAIN_SLICE, run_id)
    return build_prefix_uri(run_root, EVALUATE_DIR)


def sidecar_prefix(config_dir: str, env: str, run_id: str, model_name: str) -> str:
    """One model's backtest sidecar directory. Unused until the diagnostic panels.

    Args:
        config_dir: Root of the config tree.
        env: Needs an `environments/<env>.yaml`.
        run_id: The train run to read.
        model_name: Read off the evaluate manifest's roles.

    Raises:
        ValueError: Unknown `env`, or the fragment fails to compose.

    Returns:
        str: `gs://<bucket>/<env>/train/<run_id>/backtest/<model_name>/`.
    """
    run_root = build_run_root(_bucket(config_dir, env), env, TRAIN_SLICE, run_id)
    return build_prefix_uri(run_root, BACKTEST_DIR, model_name)


def evaluate_manifest_pattern(slice_root: str) -> str:
    """The glob that gates the run listing, one call rather than one `exists` per
    run: on 27 runs that measured 0.38s against 6.36s.

    Args:
        slice_root: From `train_slice_root`, ending in "/".

    Returns:
        str: `<slice_root>*/evaluate/evaluate_manifest.json`.
    """
    # Not build_object_uri: a glob is not an object, and "*" passes the segment
    # checks only because they ignore wildcards.
    return f"{slice_root}*/{EVALUATE_DIR}/{EVALUATE_MANIFEST}"


def pointer_uri(config_dir: str, env: str) -> str:
    """`_latest.json` at the environment root, which carries a key per slice.

    Args:
        config_dir: Root of the config tree.
        env: Needs an `environments/<env>.yaml`.

    Returns:
        str: `gs://<bucket>/<env>/_latest.json`.
    """
    environment_root = build_environment_root(_bucket(config_dir, env), env)
    return build_object_uri(environment_root, LATEST_POINTER_FILENAME)


def run_outputs_uri(config_dir: str, env: str, run_id: str) -> str:
    """One run's completion marker, which the registration badge reads.

    Args:
        config_dir: Root of the config tree.
        env: Needs an `environments/<env>.yaml`.
        run_id: The train run to read.

    Returns:
        str: `gs://<bucket>/<env>/train/<run_id>/run_output.json`.
    """
    run_root = build_run_root(_bucket(config_dir, env), env, TRAIN_SLICE, run_id)
    return build_object_uri(run_root, RUN_OUTPUT_FILENAME)
