"""Validates the committed `config/` tree against its destination schemas.

This file checks **content**, not schema behavior. Anything about how a schema
validates — which constraints fire, what an error says — belongs in
`tests/schemas/config/`. Only "do the shipped files still satisfy it" belongs
here. Without that line the two converge and pydantic gets tested twice.

It exists because nothing else in the suite reads the real tree: commit 7a's
composition spine is tested against a synthetic fixture slice under `tmp_path`,
deliberately, so that the spine is demonstrably promotable. That leaves these
files unread by any test until an impl loads them — and the tree is baked into
every image as its final layer, so a bad value costs a rebuild rather than an
edit. See `config/README.md`.

The gate is split, because only some fragments are complete on their own:

  environments/dev.yaml, train/infra.yaml, train/modeling.yaml
      validate against their destination — single-fragment, no runtime overrides

  base/data.yaml, train/backtest.yaml, train/models/*.yaml
      fragment-level only — parses, non-empty mapping, top-level keys subset of
      the fragment's allowed set

Asserting destination validation for the second group would contradict the rule
that fragments are partial by construction. Composed BacktestConfig validation
arrives in commit 7a (unit tests), commit 9 (emitted under tmp_path), and
commit 11 (landed in GCS).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tsbricks.backtesting.schema import BacktestConfig

from fcstnyctaxi.lib.config.loading import _load_config_file
from fcstnyctaxi.lib.utils import get_project_root_dir
from fcstnyctaxi.schemas.config.environment import EnvironmentConfig
from fcstnyctaxi.schemas.config.train import TrainInfraConfig, TrainModelingConfig

CONFIG_DIR = get_project_root_dir() / "config"

# A generated mirror rather than a written list, so new tsbricks fields are
# picked up automatically and this cannot go stale.
_BACKTEST_KEYS = frozenset(BacktestConfig.model_fields)


def _assert_fragment_keys(path: Path, allowed: frozenset[str]) -> None:
    """Fragment-level check: loads, and top-level keys are within `allowed`.

    _load_config_file already guarantees the file parses into a non-empty
    mapping and rejects duplicate keys, so this adds only the key check.
    """
    fragment = _load_config_file(path)
    unexpected = set(fragment) - allowed
    assert not unexpected, (
        f"{path.name} declares keys outside its destination: {unexpected}"
    )


# ================================================
# Single-fragment destinations — full validation
# ================================================


def test_dev_environment_validates() -> None:
    """config/environments/dev.yaml is a complete EnvironmentConfig."""
    config = EnvironmentConfig(
        **_load_config_file(CONFIG_DIR / "environments/dev.yaml")
    )

    assert config.storage.bucket_name == "nyc-taxi-ehc--modeling"
    assert config.vertex.pipeline_root.startswith("gs://")
    assert config.compute.location == config.artifact_registry.location


def test_train_infra_validates() -> None:
    """config/train/infra.yaml is a complete TrainInfraConfig."""
    config = TrainInfraConfig(**_load_config_file(CONFIG_DIR / "train/infra.yaml"))

    assert config.feature_source.panel_filename.endswith(".parquet")
    assert config.feature_source.calendar_filename.endswith(".parquet")


def test_train_modeling_validates() -> None:
    """config/train/modeling.yaml is a complete TrainModelingConfig."""
    config = TrainModelingConfig(
        **_load_config_file(CONFIG_DIR / "train/modeling.yaml")
    )

    assert config.evaluation_periods.start_months is None
    assert len(config.tiering.tier_labels) == 5


def test_every_model_role_has_a_config_file() -> None:
    """Each name in model_roles must have a config/train/models/<name>.yaml.

    A missing file is a composition failure in commit 7b; catching it here means
    editing model_roles without adding the file fails immediately.
    """
    modeling = TrainModelingConfig(
        **_load_config_file(CONFIG_DIR / "train/modeling.yaml")
    )
    roles = modeling.model_roles

    for name in (roles.benchmark, roles.challenger):
        assert (CONFIG_DIR / "train" / "models" / f"{name}.yaml").is_file(), (
            f"model_roles names {name!r} but config/train/models/{name}.yaml is missing"
        )


# ================================================
# Layered fragments — fragment-level checks only
# ================================================


def test_base_data_fragment_keys() -> None:
    """base/data.yaml declares only `data`."""
    _assert_fragment_keys(CONFIG_DIR / "base/data.yaml", frozenset({"data"}))


def test_backtest_fragment_keys() -> None:
    """train/backtest.yaml declares only real BacktestConfig fields.

    It is partial — no `data`, no `model`, no forecast_origins — so it cannot
    validate against BacktestConfig alone.
    """
    _assert_fragment_keys(CONFIG_DIR / "train/backtest.yaml", _BACKTEST_KEYS)


@pytest.mark.parametrize("name", ["naive", "xgboost"])
def test_model_fragment_keys(name: str) -> None:
    """train/models/<name>.yaml declares only `model`.

    A narrower allowed set than the schema's, expressed as data rather than as a
    branch inside a check, so relaxing it later is an edit.
    """
    _assert_fragment_keys(
        CONFIG_DIR / "train" / "models" / f"{name}.yaml", frozenset({"model"})
    )


def test_evaluation_periods_is_not_in_the_backtest_fragment() -> None:
    """The orphan key that motivated the two-axis split must not come back.

    In notebooks/backtest_configs/backtest_config.yaml this key sits at top
    level, is silently dropped by parse_config on every run, and the notebook
    re-reads the raw YAML to recover it. Its home is TrainModelingConfig.
    """
    fragment = _load_config_file(CONFIG_DIR / "train/backtest.yaml")

    assert "evaluation_periods" not in fragment


def test_calendar_source_is_declared_and_null() -> None:
    """calendar_source stays declared null and is never injected.

    It is the last rung of a tsbricks fallback for callers who do not hold the
    calendar; this project stages both artifacts and passes calendar_df
    directly. Kept so the fragment says so at the point of use.
    """
    fragment = _load_config_file(CONFIG_DIR / "train/backtest.yaml")

    assert "calendar_source" in fragment["aggregation"]
    assert fragment["aggregation"]["calendar_source"] is None


def test_naive_declares_freq_and_xgboost_does_not() -> None:
    """The asymmetry is deliberate, and gives the cross-field check both branches.

    naive_weekly's freq is load-bearing; xgboost's hyperparameters land in PR 1.
    A model may legitimately declare no freq and let its callable infer one, so
    a universal requirement would be wrong.
    """
    naive = _load_config_file(CONFIG_DIR / "train/models/naive.yaml")
    xgboost = _load_config_file(CONFIG_DIR / "train/models/xgboost.yaml")

    assert naive["model"]["hyperparameters"]["freq"] == "W-SUN"
    assert "hyperparameters" not in xgboost["model"]


def test_model_freq_matches_base_data_freq() -> None:
    """A model's declared freq must agree with the data contract.

    This is the shipped-config half of stage 5's cross-field check, which lands
    in commit 7a. A tsbricks default of freq=1 once silently cast
    datetime64[ns] to integers, which is why the value is checked rather than
    trusted.
    """
    data_freq = _load_config_file(CONFIG_DIR / "base/data.yaml")["data"]["freq"]
    naive = _load_config_file(CONFIG_DIR / "train/models/naive.yaml")

    assert naive["model"]["hyperparameters"]["freq"] == data_freq
