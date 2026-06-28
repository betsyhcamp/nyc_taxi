from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from tsbricks.backtesting.schema import BacktestConfig

from fcstnyctaxi.lib.config_utils import _deep_merge, merge_configs, save_config

# Minimum valid BacktestConfig as a plain dict — reused across multiple tests
_MINIMAL_CONFIG: dict = {
    "data": {
        "freq": "W-SAT",
        "id_col": "unique_id",
        "date_col": "ds",
        "target_col": "y",
    },
    "cross_validation": {
        "mode": "explicit",
        "forecast_origins": [{"origin": "2025-04-20", "horizon": 3}],
    },
    "model": {"callable": "models.naive.naive_weekly"},
    "evaluation": {
        "native": {
            "metrics": {
                "definitions": [
                    {
                        "name": "mae",
                        "callable": "tsbricks.blocks.metrics.mae",
                        "type": "simple",
                    }
                ]
            }
        }
    },
}


@pytest.fixture
def minimal_config_yaml(tmp_path: Path) -> Path:
    """Write _MINIMAL_CONFIG to a temporary YAML file and return its path."""
    path = tmp_path / "base_config.yaml"
    path.write_text(yaml.dump(_MINIMAL_CONFIG))
    return path


# ================================================
# _deep_merge tests
# ================================================


def test_deep_merge_override_wins_for_scalar_key() -> None:
    """Override value replaces base value for a shared scalar key."""
    base = {"a": 1, "b": 2}
    override = {"b": 99}
    result = _deep_merge(base, override)
    assert result == {"a": 1, "b": 99}


def test_deep_merge_nested_dicts_merged_recursively() -> None:
    """Nested dicts are merged recursively — sibling keys in base are preserved."""
    base = {"data": {"freq": "W-SAT", "id_col": "unique_id"}}
    override = {"data": {"freq": "MS"}}
    result = _deep_merge(base, override)
    assert result == {"data": {"freq": "MS", "id_col": "unique_id"}}


def test_deep_merge_list_replaced_not_merged() -> None:
    """Lists are replaced wholesale — no element-level merging."""
    base = {"origins": [1, 2, 3]}
    override = {"origins": [99]}
    result = _deep_merge(base, override)
    assert result["origins"] == [99]


def test_deep_merge_empty_override_returns_base() -> None:
    """An empty override leaves base unchanged."""
    base = {"a": 1}
    result = _deep_merge(base, {})
    assert result == {"a": 1}


def test_deep_merge_does_not_mutate_inputs() -> None:
    """_deep_merge must not modify either input dict in place."""
    base = {"data": {"freq": "W-SAT"}}
    override = {
        "model": {"callable": "models.naive.naive_weekly"},
    }

    result = _deep_merge(base, override)

    assert result.keys() == {"data", "model"}
    assert result["data"].keys() == {"freq"}
    assert result["model"].keys() == {"callable"}


# ================================================
# merge_configs tests
# ================================================


def test_merge_configs_two_dicts_returns_backtest_config() -> None:
    """Two dicts are deep-merged and validated into a BacktestConfig."""
    base = {k: v for k, v in _MINIMAL_CONFIG.items() if k != "model"}
    model_spec = {"model": {"callable": "models.naive.naive_weekly"}}
    result = merge_configs(base, model_spec)
    assert isinstance(result, BacktestConfig)
    assert result.model.callable == "models.naive.naive_weekly"


def test_merge_configs_yaml_path_loads_and_merges(minimal_config_yaml: Path) -> None:
    """A YAML path is loaded and merged with a subsequent dict override."""
    hyperparams_override = {"model": {"hyperparameters": {"season_length": 52}}}
    result = merge_configs(str(minimal_config_yaml), hyperparams_override)
    assert result.model.hyperparameters == {"season_length": 52}


def test_merge_configs_three_way_merge() -> None:
    """Three-way merge: base + model_spec + runtime override — last dict wins."""
    # TODO: implement this test.
    # This is the core runtime pattern from the spec:
    #   merge_configs(base_dict, model_spec_dict, runtime_override_dict)
    # Split _MINIMAL_CONFIG across three layers (base without model, model spec,
    # and a runtime override that changes one field like data.freq).
    # Assert that the final BacktestConfig reflects values from all three layers,
    # and that the runtime override value wins over earlier layers.
    pass


def test_merge_configs_empty_yaml_raises(tmp_path: Path) -> None:
    """An empty YAML file raises ValueError."""
    empty = tmp_path / "empty.yaml"
    empty.write_text("")
    with pytest.raises(ValueError, match="Config is empty"):
        merge_configs(str(empty))


def test_merge_configs_invalid_type_raises() -> None:
    """A non-dict/str/Path element raises ValueError naming the bad type."""
    with pytest.raises(ValueError, match="only allow"):
        merge_configs(42)  # type: ignore[arg-type]


# ================================================
# save_config tests
# ================================================


def test_save_config_dict_writes_to_local_path(tmp_path: Path) -> None:
    """save_config writes a dict to a local path as valid YAML."""
    out_path = tmp_path / "composed.yaml"
    save_config(_MINIMAL_CONFIG, out_path)
    assert out_path.exists()
    loaded = yaml.safe_load(out_path.read_text())
    assert loaded["model"]["callable"] == "models.naive.naive_weekly"


def test_save_config_backtest_config_writes_to_local_path(tmp_path: Path) -> None:
    """save_config serializes a BacktestConfig to a local YAML path."""
    cfg = merge_configs(_MINIMAL_CONFIG)
    out_path = tmp_path / "composed.yaml"
    save_config(cfg, out_path)
    loaded = yaml.safe_load(out_path.read_text())
    assert loaded["model"]["callable"] == "models.naive.naive_weekly"


def test_save_config_routes_gcs_path_to_write_text_to_gcs() -> None:
    """save_config calls write_text_to_gcs for gs:// paths, not local write."""
    gcs_uri = "gs://bucket/sidecar/composed_config.yaml"
    with patch("fcstnyctaxi.lib.config_utils.write_text_to_gcs") as mock_write:
        save_config(_MINIMAL_CONFIG, gcs_uri)
    mock_write.assert_called_once()
    assert mock_write.call_args.kwargs["gcs_uri"] == gcs_uri


def test_save_config_key_order_preserved(tmp_path: Path) -> None:
    """Keys in saved YAML follow dict insertion order, not alphabetical order."""
    # TODO implement this test.
    # Construct a small dict with keys in a deliberately non-alphabetical order
    # (e.g. "z", "a", "m"). Call save_config to a local tmp_path.
    # Read the raw YAML text back and check that the top-level keys appear
    # in the same order you inserted them — not sorted alphabetically.
    # Hint: parse the raw lines rather than yaml.safe_load (which returns a dict
    # and doesn't tell you the file's key order).
    pass
