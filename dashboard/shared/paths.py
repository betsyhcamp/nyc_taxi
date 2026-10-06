from pathlib import Path

from fcstnyctaxi.lib.storage_layout import (
    resolve_environment_root,
    resolve_latest_pointer_uri,
    resolve_run_outputs_uri,
    resolve_run_prefix,
)
from fcstnyctaxi.schemas.config.common import SliceName

# Typed, so a token outside SliceName is a type error rather than a 404. Also
# the key `runs.py` reads `_latest.json` by, which holds one record per slice.
TRAIN_SLICE: SliceName = "train"
_EVALUATE_STEP = "evaluate"
_BACKTEST_STEP = "backtest"

# Objects in the evaluate directory. Pinned to the producer's own names by test,
# since this is the fourth place they are written down.
EVALUATE_MANIFEST = "evaluate_manifest.json"
SUMMARY_METRICS = "summary_metrics.parquet"
FOLD_METRICS = "fold_metrics.parquet"


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
    return f"{resolve_environment_root(Path(config_dir), env)}{TRAIN_SLICE}/"


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
    run_prefix = resolve_run_prefix(Path(config_dir), env, TRAIN_SLICE, run_id)
    return f"{run_prefix}{_EVALUATE_STEP}/"


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
    run_prefix = resolve_run_prefix(Path(config_dir), env, TRAIN_SLICE, run_id)
    return f"{run_prefix}{_BACKTEST_STEP}/{model_name}/"


def evaluate_manifest_pattern(slice_root: str) -> str:
    """The glob that gates the run listing, one call rather than one `exists` per
    run: on 27 runs that measured 0.38s against 6.36s.

    Args:
        slice_root: From `train_slice_root`, ending in "/".

    Returns:
        str: `<slice_root>*/evaluate/evaluate_manifest.json`.
    """
    return f"{slice_root}*/{_EVALUATE_STEP}/{EVALUATE_MANIFEST}"


def pointer_uri(config_dir: str, env: str) -> str:
    """`_latest.json` at the environment root, which carries a key per slice.

    A passthrough, so that every path shape and every `Path` conversion lives in
    this module and `load.py` only caches.

    Args:
        config_dir: Root of the config tree.
        env: Needs an `environments/<env>.yaml`.

    Returns:
        str: `gs://<bucket>/<env>/_latest.json`.
    """
    return resolve_latest_pointer_uri(Path(config_dir), env)


def run_outputs_uri(config_dir: str, env: str, run_id: str) -> str:
    """One run's completion marker, which the registration badge reads.

    Args:
        config_dir: Root of the config tree.
        env: Needs an `environments/<env>.yaml`.
        run_id: The train run to read.

    Returns:
        str: `gs://<bucket>/<env>/train/<run_id>/run_output.json`.
    """
    return resolve_run_outputs_uri(Path(config_dir), env, TRAIN_SLICE, run_id)
