from pathlib import Path

from fcstnyctaxi.lib.storage_layout import (
    resolve_environment_root,
    resolve_run_prefix,
)
from fcstnyctaxi.schemas.config.common import SliceName

# Typed, so a token outside SliceName is a type error rather than a 404.
_SLICE: SliceName = "train"
_EVALUATE_STEP = "evaluate"
_BACKTEST_STEP = "backtest"


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
    return f"{resolve_environment_root(Path(config_dir), env)}{_SLICE}/"


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
    run_prefix = resolve_run_prefix(Path(config_dir), env, _SLICE, run_id)
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
    run_prefix = resolve_run_prefix(Path(config_dir), env, _SLICE, run_id)
    return f"{run_prefix}{_BACKTEST_STEP}/{model_name}/"
