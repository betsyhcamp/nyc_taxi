"""Merge and save BacktestConfig YAML configs for the weekly backtest pipeline.

Mirrors the intended interface for tsbricks.backtesting.config_utils;
lives here until the API is validated through use in this project.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from tsbricks.backtesting.schema import BacktestConfig, parse_config

from fcstnyctaxi.lib.config.loading import _load_config_file
from fcstnyctaxi.lib.io import write_text_to_gcs


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = base.copy()

    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val

    return result


def merge_configs(*configs: dict | str | Path) -> BacktestConfig:
    merged: dict = {}
    for element in configs:
        if isinstance(element, dict):
            merged = _deep_merge(merged, element)
        elif isinstance(element, (str | Path)):
            merged = _deep_merge(merged, _load_config_file(Path(element)))
        else:
            raise ValueError(
                f"Passed {type(element).__name__}; only allow dict, str, or Path"
            )

    return parse_config(config=merged)


def save_config(config: BacktestConfig | dict, path: str | Path) -> None:
    if not isinstance(config, (BacktestConfig | dict)):
        raise ValueError(
            f"Passed config is type {type(config).__name__}; "
            "only dict or BacktestConfig allowed"
        )
    if not isinstance(path, (Path | str)):
        raise ValueError(
            f"Passed path is type {type(path).__name__}; only str or Path allowed"
        )

    if isinstance(config, BacktestConfig):
        raw_config = config.model_dump(by_alias=True, exclude_none=True)
    else:
        raw_config = config

    text = yaml.dump(raw_config, default_flow_style=False, sort_keys=False)

    if str(path).startswith("gs://"):
        write_text_to_gcs(text, gcs_uri=str(path))
    else:
        Path(path).write_text(text)
