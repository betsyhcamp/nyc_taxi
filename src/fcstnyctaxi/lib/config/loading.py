import collections.abc
from pathlib import Path
from typing import Any

import yaml

from fcstnyctaxi.schemas.config_schemas import PipelineConfig


class _StrictYamlLoader(yaml.SafeLoader):
    """SafeLoader that raises on a duplicate mapping key.

    PyYAML collapses `a: 1` / `a: 2` to {"a": 2} at parse time, before any
    validation stage can observe the loss so the composition stages check the
    merged document, and nothing is missing from it. This is a silent
    config loss path missed due to defaults so it is closed at the load/parse boundary.
    """

    def construct_mapping(
        self, node: yaml.MappingNode, deep: bool = False
    ) -> dict[Any, Any]:
        """Build a mapping, rejecting any key that appears more than once."""
        keys_seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if isinstance(key, collections.abc.Hashable):
                if key in keys_seen:
                    raise ValueError(
                        f"duplicate config key {key!r}{key_node.start_mark}"
                    )
                keys_seen.add(key)
        return super().construct_mapping(node, deep=deep)


def _load_config_file(path: Path) -> dict[str, Any]:
    """Read a YAML file into a non-empty mapping.

    Shared by load_pipeline_config, merge_configs' file branch, and the
    composition spine's fragment loading. All three want identical policy:
    no empty files, no non-mapping documents, no duplicate keys.

    Args:
        path (Path): Path to a .yaml / .yml file.

    Raises:
        FileNotFoundError: If the file at `path` does not exist.
        ValueError: If the extension is not .yaml / .yml, the document is
            empty, the document is not a mapping, or a mapping key repeats.

    Returns:
        dict[str, Any]: The parsed document.
    """
    ext = path.suffix.lower()
    if ext not in (".yaml", ".yml"):
        raise ValueError(f"Unsupported config type {ext} at {path}; use .yaml/.yml")

    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.load(f, Loader=_StrictYamlLoader)

    if loaded is None or loaded == {}:
        raise ValueError(f"Config is empty at path {path}")
    if not isinstance(loaded, dict):
        raise ValueError(
            f"Config must be a mapping, not {type(loaded).__name__}, at path {path}"
        )
    return loaded


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
        ValueError: If the file extension is not .yaml / .yml, the file is
            empty, the document is not a mapping, or a mapping key repeats.
        ValidationError: If the YAML contents do not conform to PipelineConfig
            (missing fields, extra fields, wrong types).

    Returns:
        PipelineConfig: Validated configuration object with all nested
            ProjectSettings and ExtractDbToBucketConfig models populated.
    """
    raw_config_dict = _load_config_file(path)
    return PipelineConfig(**raw_config_dict)
