from pathlib import Path
from typing import Any

import yaml

from fcstnyctaxi.schemas.config_schemas import PipelineConfig


def _load_config_file(path: Path) -> dict[str, Any]:
    """Load a .yaml / .yml configuration file and return the contents file as
    a dictionary. Private helper for load_pipeline_config()."""

    ext = path.suffix.lower()
    if ext not in (".yaml", ".yml"):
        raise ValueError(f"Unsupported config type {ext} at {path}; use .yaml/.yml")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_pipeline_config(path: Path) -> PipelineConfig:
    """Load and validate the pipeline config YAML into a PipelineConfig.

    Reads the YAML file at the given path and validates the result against
    the PipelineConfig Pydantic model. Raw dict access on the loaded YAML
    is intentionally not exposed; the only way to obtain config data is
    through this function, which guarantees a validated, frozen result.

    Args:
        path (Path): Path to the pipeline configuration YAML file.

    Raises:
        FileNotFoundError: If the file at `path` does not exist.
        ValueError: If the file extension is not .yaml / .yml.
        ValidationError: If the YAML contents do not conform to PipelineConfig
            (missing fields, extra fields, wrong types).

    Returns:
        PipelineConfig: Validated configuration object with all nested
            ProjectSettings and ExtractDbToBucketConfig models populated.
    """
    raw_config_dict = _load_config_file(path)
    return PipelineConfig(**raw_config_dict)
