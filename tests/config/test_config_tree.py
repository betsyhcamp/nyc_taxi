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

import ast
import importlib.util
from pathlib import Path

from tsbricks.backtesting.schema import BacktestConfig

from fcstnyctaxi.lib.config.bindings import model_names_from_roles
from fcstnyctaxi.lib.config.loading import _load_config_file
from fcstnyctaxi.lib.registry_ids import compose_display_name, compose_model_id
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
    """config/train/infra.yaml is a complete TrainInfraConfig.

    display_name_prefix is also asserted in tests/lib/config/test_bindings.py,
    and that is not duplication to be tidied away: this file asks whether the
    shipped file validates and carries the intended value, that one asks whether
    the binding composes it into the intended destination. An edit to infra.yaml
    should fail both.
    """
    config = TrainInfraConfig(**_load_config_file(CONFIG_DIR / "train/infra.yaml"))

    assert config.display_name_prefix == "fcst-train-pipeline"
    assert config.model_registry.display_name_prefix == "fcst-monthly-revenue"
    # Pinned because an edit forks the registry: the next run creates a new Model
    # resource rather than a version, and nothing raises.
    assert config.model_registry.model_id_prefix == "fcst-monthly-revenue"


def test_train_modeling_validates() -> None:
    """config/train/modeling.yaml is a complete TrainModelingConfig."""
    config = TrainModelingConfig(
        **_load_config_file(CONFIG_DIR / "train/modeling.yaml")
    )

    assert config.evaluation_periods.start_months is None
    assert len(config.tiering.tier_labels) == 5


def test_every_registrable_model_composes_legal_registry_names() -> None:
    """The gate compose_configs runs on the image's tree, run here on the repo's.
    Registrable means a role model declaring callables, the only kind final_fit fits."""
    infra = TrainInfraConfig(**_load_config_file(CONFIG_DIR / "train/infra.yaml"))
    modeling = TrainModelingConfig(
        **_load_config_file(CONFIG_DIR / "train/modeling.yaml")
    )
    registrable = [
        name
        for name in model_names_from_roles(modeling.model_roles)
        if modeling.model_settings[name].fit_callable is not None
    ]

    # Self-check: an empty set would pass vacuously.
    assert registrable
    for model_name in registrable:
        compose_model_id(infra.model_registry.model_id_prefix, model_name)
        compose_display_name(infra.model_registry.display_name_prefix, model_name)


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


def _model_files() -> list[Path]:
    """Every train/models/*.yaml, whether or not its model holds a role."""
    files = sorted((CONFIG_DIR / "train" / "models").glob("*.yaml"))
    # Self-check: an empty glob would pass every loop over it.
    assert files
    return files


def _named_callables() -> tuple[list[str], list[str]]:
    """Every dotted path in train/models/*.yaml, then every one in modeling.yaml."""
    model_paths = [
        path
        for file in _model_files()
        for key, path in _load_config_file(file)["model"].items()
        if key.endswith("_callable")
    ]
    settings = _load_config_file(CONFIG_DIR / "train/modeling.yaml")["model_settings"]
    settings_paths = [
        path
        for entry in settings.values()
        for key, path in entry.items()
        if key.endswith("_callable")
    ]
    return model_paths, settings_paths


def test_every_callable_the_config_names_is_defined_where_it_points() -> None:
    """Composition keeps a dotted path as a string, so a typo would surface on Vertex.
    Located without importing it, so no modeling package is needed."""
    model_paths, settings_paths = _named_callables()
    # Self-check: both sources contribute, so neither half passes vacuously.
    assert model_paths and settings_paths

    for path in model_paths + settings_paths:
        module_name, function_name = path.rsplit(".", 1)
        spec = importlib.util.find_spec(module_name)
        assert spec is not None and spec.origin is not None, path
        tree = ast.parse(Path(spec.origin).read_text())
        defined = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        assert function_name in defined, path


def test_base_data_fragment_keys() -> None:
    """base/data.yaml declares only `data`."""
    _assert_fragment_keys(CONFIG_DIR / "base/data.yaml", frozenset({"data"}))


def test_backtest_fragment_keys() -> None:
    """train/backtest.yaml declares only real BacktestConfig fields.

    It is partial — no `data`, no `model`, no forecast_origins — so it cannot
    validate against BacktestConfig alone.
    """
    _assert_fragment_keys(CONFIG_DIR / "train/backtest.yaml", _BACKTEST_KEYS)


def test_model_fragment_keys() -> None:
    """Every train/models/*.yaml declares only `model`, including a model holding no
    role, whose file no composition reads. A loop: an empty parametrize would skip."""
    for path in _model_files():
        _assert_fragment_keys(path, frozenset({"model"}))


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

    naive_weekly's freq is load-bearing; xgboost's land with its module.
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
    declared = {}
    for path in _model_files():
        hyperparameters = _load_config_file(path)["model"].get("hyperparameters") or {}
        declared[path.stem] = hyperparameters.get("freq")

    # Required, not only checked: naive_weekly has no default for freq.
    assert declared["naive"] == data_freq
    assert set(declared.values()) <= {data_freq, None}, declared
