from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pytest_mock import MockerFixture
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
    "model": {"fit_predict_callable": "models.naive.naive_weekly"},
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
        "model": {"fit_predict_callable": "models.naive.naive_weekly"},
    }

    result = _deep_merge(base, override)

    assert result.keys() == {"data", "model"}
    assert result["data"].keys() == {"freq"}
    assert result["model"].keys() == {"fit_predict_callable"}


# ================================================
# merge_configs tests
# ================================================


def test_merge_configs_two_dicts_returns_backtest_config() -> None:
    """Two dicts are deep-merged and validated into a BacktestConfig."""
    base = {k: v for k, v in _MINIMAL_CONFIG.items() if k != "model"}
    model_spec = {"model": {"fit_predict_callable": "models.naive.naive_weekly"}}
    result = merge_configs(base, model_spec)
    assert isinstance(result, BacktestConfig)
    assert result.model.fit_predict_callable == "models.naive.naive_weekly"


def test_merge_configs_yaml_path_loads_and_merges(minimal_config_yaml: Path) -> None:
    """A YAML path is loaded and merged with a subsequent dict override."""
    hyperparams_override = {"model": {"hyperparameters": {"season_length": 52}}}
    result = merge_configs(str(minimal_config_yaml), hyperparams_override)
    assert result.model.hyperparameters == {"season_length": 52}


def test_merge_configs_three_way_merge() -> None:
    """Three-way merge: base + model_spec + runtime override — last dict wins."""
    base = {k: v for k, v in _MINIMAL_CONFIG.items() if k != "model"}
    model_spec = {k: v for k, v in _MINIMAL_CONFIG.items() if k == "model"}
    override = {"data": {"target_col": "y_new"}}

    result = merge_configs(base, model_spec, override)
    assert result.data.id_col == "unique_id"
    assert result.model.fit_predict_callable == "models.naive.naive_weekly"
    assert result.data.target_col == "y_new"


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
    assert loaded["model"]["fit_predict_callable"] == "models.naive.naive_weekly"


def test_save_config_backtest_config_writes_to_local_path(tmp_path: Path) -> None:
    """save_config serializes a BacktestConfig to a local YAML path."""
    cfg = merge_configs(_MINIMAL_CONFIG)
    out_path = tmp_path / "composed.yaml"
    save_config(cfg, out_path)
    loaded = yaml.safe_load(out_path.read_text())
    assert loaded["model"]["fit_predict_callable"] == "models.naive.naive_weekly"


def test_save_config_routes_gcs_path_to_write_text_to_gcs(
    mocker: MockerFixture,
) -> None:
    """save_config calls write_text_to_gcs for gs:// paths, not local write."""
    gcs_uri = "gs://bucket/sidecar/composed_config.yaml"
    mock_write = mocker.patch("fcstnyctaxi.lib.config_utils.write_text_to_gcs")
    save_config(_MINIMAL_CONFIG, gcs_uri)
    mock_write.assert_called_once()
    assert mock_write.call_args.kwargs["gcs_uri"] == gcs_uri


def test_save_config_key_order_preserved(tmp_path: Path) -> None:
    """Keys in saved YAML follow dict insertion order, not alphabetical order."""
    config = {"z": "z_value", "a": "a_value", "m": "m_value"}
    out_path = tmp_path / "config_preserved.yaml"
    save_config(config, out_path)
    loaded = out_path.read_text().splitlines()
    loaded_keys = [item.split(":")[0].strip() for item in loaded]
    assert loaded_keys == ["z", "a", "m"]
