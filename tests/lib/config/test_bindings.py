from pathlib import Path

import pytest
import yaml
from tsbricks.backtesting.schema import BacktestConfig

from fcstnyctaxi.lib.config.bindings import (
    available_environments,
    environment_bindings,
    feature_bindings,
    inference_bindings,
    model_names_from_roles,
    require_known_environment,
    resolve_model_names,
    train_backtest_bindings,
    train_infra_bindings,
    train_modeling_bindings,
)
from fcstnyctaxi.lib.config.composition import (
    RUNTIME_SOURCE,
    _preflight_bindings,
    compose_config,
)
from fcstnyctaxi.lib.utils import get_project_root_dir
from fcstnyctaxi.schemas.config.environment import EnvironmentConfig
from fcstnyctaxi.schemas.config.train import (
    ModelRoles,
    TrainInfraConfig,
    TrainModelingConfig,
)

CONFIG_DIR = get_project_root_dir() / "config"

# The one runtime override. forecast_origins is derived from the panel, so no
# file in the tree can hold it and BacktestConfig cannot validate without it.
# Only forecast_origins is supplied: it deep-merges onto the `mode` that
# train/backtest.yaml already declares.
_ORIGINS_OVERRIDE = {
    "cross_validation": {"forecast_origins": [{"origin": "2025-04-20", "horizon": 3}]}
}


@pytest.fixture
def environments_dir(tmp_path: Path) -> Path:
    """A tree whose `environments/` holds two `.yaml` files and one `.yml`.

    The contents are irrelevant — `available_environments` reads stems only —
    so this fixture is about the glob, not about any destination.
    """
    config_dir = tmp_path / "config"
    (config_dir / "environments").mkdir(parents=True)
    for filename in ("prod.yaml", "dev.yaml", "staging.yml"):
        (config_dir / "environments" / filename).write_text("compute: {}\n")

    return config_dir


@pytest.fixture
def duplicated_roles_tree(tmp_path: Path) -> Path:
    """A tree whose `model_roles` name one model in both slots."""
    document = yaml.safe_load((CONFIG_DIR / "train" / "modeling.yaml").read_text())
    document["model_roles"] = {"benchmark": "model_a", "challenger": "model_a"}

    config_dir = tmp_path / "config"
    (config_dir / "train").mkdir(parents=True)
    (config_dir / "train" / "modeling.yaml").write_text(yaml.dump(document))

    TrainModelingConfig.model_validate(document)
    return config_dir


# ================================================
# The environment selector
# ================================================


def test_available_environments_lists_the_shipped_tree() -> None:
    """The committed tree defines dev and nothing else."""
    assert available_environments(CONFIG_DIR) == ["dev"]


def test_available_environments_is_sorted_and_ignores_yml(
    environments_dir: Path,
) -> None:
    """`.yml` is excluded deliberately, and the result is sorted.

    A `prod.yml` would load when named explicitly yet never appear here, so it
    would exist and be unreachable. The narrower rule turns that into a named
    error instead.
    """
    assert available_environments(environments_dir) == ["dev", "prod"]


def test_available_environments_is_empty_when_the_directory_is_absent(
    tmp_path: Path,
) -> None:
    """A missing `environments/` returns empty rather than raising.

    Discovery reports what is there; deciding whether that is acceptable is
    `require_known_environment`'s job, which is what lets it distinguish an
    unknown env from a wrong `config_dir`.
    """
    assert available_environments(tmp_path) == []


def test_require_known_environment_accepts_a_defined_environment() -> None:
    """dev has a file, so the guard returns without raising."""
    require_known_environment(CONFIG_DIR, "dev")


def test_require_known_environment_names_the_env_and_the_alternatives() -> None:
    """An unknown selector names itself and what is available."""
    with pytest.raises(ValueError, match="Unknown env 'prod'; available: dev"):
        require_known_environment(CONFIG_DIR, "prod")


def test_require_known_environment_names_the_directory_when_none_are_defined(
    tmp_path: Path,
) -> None:
    """An empty set almost always means `config_dir` is wrong, so it says so.

    `available: ` followed by nothing would point at the environment, which is
    the one thing that is not the problem.
    """
    with pytest.raises(ValueError, match="No environments are defined under"):
        require_known_environment(tmp_path, "dev")


# ================================================
# Single-fragment declarations
# ================================================


def test_environment_bindings_names_the_selected_file() -> None:
    """The selector becomes the path; nothing else records the environment."""
    (binding,) = environment_bindings("dev")

    assert binding.path == Path("environments/dev.yaml")
    assert binding.destination is EnvironmentConfig
    assert binding.destination_key == "EnvironmentConfig"
    assert binding.allowed_keys is None


def test_train_infra_bindings_is_one_unnarrowed_fragment() -> None:
    """One destination, one file — and the file is the whole destination.

    No `allowed_keys`, because this fragment is not defined by one block of its
    schema the way `base/data.yaml` and the model files are.
    """
    (binding,) = train_infra_bindings()

    assert binding.path == Path("train/infra.yaml")
    assert binding.destination is TrainInfraConfig
    assert binding.destination_key == "TrainInfraConfig"
    assert binding.allowed_keys is None


def test_train_modeling_bindings_is_one_unnarrowed_fragment() -> None:
    """Phase 1's third destination, and the one phase 2 reads its model set from."""
    (binding,) = train_modeling_bindings()

    assert binding.path == Path("train/modeling.yaml")
    assert binding.destination is TrainModelingConfig
    assert binding.destination_key == "TrainModelingConfig"
    assert binding.allowed_keys is None


@pytest.mark.parametrize(
    "slice_bindings",
    [feature_bindings, inference_bindings],
    ids=["feature", "inference"],
)
def test_a_slice_with_no_project_owned_destinations_composes_the_environment_alone(
    slice_bindings,
) -> None:
    """Absence is a fact in code, not an inference from a missing directory.

    It is also why absence never takes the form of an empty binding sequence,
    which the spine's preflight rejects: a slice with no project-owned
    destinations still has one.
    """
    assert slice_bindings("dev") == environment_bindings("dev")


# ================================================
# The layered destination
# ================================================


def test_train_backtest_bindings_layers_three_fragments_in_precedence_order() -> None:
    """Data contract, then backtest settings, then the model. Later wins."""
    bindings = train_backtest_bindings("naive")

    assert [binding.path for binding in bindings] == [
        Path("base/data.yaml"),
        Path("train/backtest.yaml"),
        Path("train/models/naive.yaml"),
    ]


def test_train_backtest_bindings_narrow_only_the_single_block_fragments() -> None:
    """A fragment narrows when it is defined by the block it carries.

    Narrowing `base/data.yaml` is load-bearing rather than tidy: BacktestConfig
    does not forbid extra keys, so unnarrowed this fragment could legally declare
    any field of that schema — and a stray `model:` block there would silently
    apply to every model. `train/backtest.yaml` stays open because its allowed
    set would be a definition by subtraction.
    """
    data, backtest, model = train_backtest_bindings("naive")

    assert data.allowed_keys == frozenset({"data"})
    assert backtest.allowed_keys is None
    assert model.allowed_keys == frozenset({"model"})


def test_train_backtest_bindings_key_every_fragment_to_the_same_model() -> None:
    """One key across all three, so the preflight can never see a mixed sequence."""
    bindings = train_backtest_bindings("xgboost")

    assert {binding.destination_key for binding in bindings} == {
        "BacktestConfig:xgboost"
    }
    assert all(binding.destination is BacktestConfig for binding in bindings)


def test_two_models_share_a_destination_and_differ_by_key() -> None:
    """Exactly what `destination_key` exists for, and why both preflight checks stay.

    A destination-only check would let two models' fragments into one call.
    """
    naive = train_backtest_bindings("naive")
    xgboost = train_backtest_bindings("xgboost")

    assert naive[0].destination is xgboost[0].destination
    assert naive[0].destination_key != xgboost[0].destination_key


# ================================================
# The model set
# ================================================


def test_model_names_from_roles_keeps_declaration_order() -> None:
    """Benchmark first, challenger second — field declaration order."""
    roles = ModelRoles(benchmark="naive", challenger="xgboost")

    assert model_names_from_roles(roles) == ("naive", "xgboost")


def test_model_names_from_roles_does_not_sort() -> None:
    """Order is role order, not alphabetical.

    The shipped roles happen to be alphabetical, so `sorted(set(...))` would pass
    every other test in this section. These roles are reversed, which is the only
    arrangement that tells the two apart.
    """
    roles = ModelRoles(benchmark="xgboost", challenger="naive")

    assert model_names_from_roles(roles) == ("xgboost", "naive")


def test_model_names_from_roles_deduplicates_a_repeated_name() -> None:
    """One name used in both roles composes once and emits one config.

    Permitted rather than rejected: benchmark == challenger yields WRMAE = 1.0,
    a legitimate smoke test of the whole evaluation path.
    """
    roles = ModelRoles(benchmark="naive", challenger="naive")

    assert model_names_from_roles(roles) == ("naive",)


def test_resolve_model_names_composes_the_shipped_tree() -> None:
    """Every name the real tree resolves to has a model file behind it."""
    model_names = resolve_model_names(CONFIG_DIR)

    # Non-vacuity: an empty tuple would satisfy the loop below.
    assert model_names
    for model_name in model_names:
        assert (CONFIG_DIR / "train" / "models" / f"{model_name}.yaml").is_file()


def test_resolve_model_names_reads_the_tree_it_is_given(
    duplicated_roles_tree: Path,
) -> None:
    """A name in both roles resolves once, from the passed tree rather than config/."""
    assert resolve_model_names(duplicated_roles_tree) == ("model_a",)


# ================================================
# Every tuple is a legal compose_config argument
# ================================================


@pytest.mark.parametrize(
    "bindings",
    [
        environment_bindings("dev"),
        train_infra_bindings(),
        train_modeling_bindings(),
        train_backtest_bindings("naive"),
        feature_bindings("dev"),
        inference_bindings("dev"),
    ],
    ids=[
        "environment",
        "train_infra",
        "train_modeling",
        "train_backtest",
        "feature",
        "inference",
    ],
)
def test_every_binding_tuple_passes_the_spine_preflight(bindings) -> None:
    """The module's central claim, checked without opening a file.

    `compose_config` is per-destination and its preflight rejects an empty
    sequence, a mixed destination or key, an absolute path, or a `..`. Nothing
    this module returns may trip it — which is what lets a caller pass a result
    straight through instead of grouping or filtering it first.
    """
    _preflight_bindings(bindings)


def test_the_shipped_tree_composes_every_phase_one_destination() -> None:
    """Phase 1 end to end against the real tree — no data, no credentials.

    This is what `compose_train_static_configs` will do in commit 9, and it is
    the only place bindings.py and the committed files are checked against each
    other.
    """
    environment = compose_config(CONFIG_DIR, environment_bindings("dev"))
    infra = compose_config(CONFIG_DIR, train_infra_bindings())
    modeling = compose_config(CONFIG_DIR, train_modeling_bindings())

    assert environment.config.storage.bucket_name == "nyc-taxi-ehc--modeling"
    assert infra.config.display_name_prefix == "fcst-train-pipeline"
    # Not the model names themselves: those change whenever a role is reassigned,
    # and deriving the expectation from the same call would make this unfailable.
    # The invariant is that every configured role names a model file that exists.
    for model_name in model_names_from_roles(modeling.config.model_roles):
        assert (CONFIG_DIR / "train" / "models" / f"{model_name}.yaml").is_file()


@pytest.mark.parametrize("model_name", ["naive", "xgboost"])
def test_the_shipped_tree_composes_a_backtest_config_for_each_model(
    model_name: str,
) -> None:
    """Phase 2 for both shipped models, with the one runtime override.

    `config_files` is asserted in full because it is what a manifest records:
    relative, POSIX, in precedence order, with the runtime source last. Absolute
    paths here would make two runs of identical config non-comparable.
    """
    composed = compose_config(
        CONFIG_DIR, train_backtest_bindings(model_name), _ORIGINS_OVERRIDE
    )

    assert composed.config.data.freq == "W-SUN"
    assert composed.config_files == [
        "base/data.yaml",
        "train/backtest.yaml",
        f"train/models/{model_name}.yaml",
        RUNTIME_SOURCE,
    ]


# ================================================
# The two checks this module deliberately omits
# ================================================


def test_a_model_with_no_file_is_declared_here_and_raises_at_composition() -> None:
    """Stage 1 owns the missing-file failure, and its message names the path.

    The binding builds without complaint. A second existence check here would
    have to take `config_dir` purely to re-raise what stage 1 already raises.
    """
    bindings = train_backtest_bindings("nonexistent_model")

    with pytest.raises(FileNotFoundError, match="nonexistent_model.yaml"):
        compose_config(CONFIG_DIR, bindings, _ORIGINS_OVERRIDE)


def test_a_model_name_escaping_the_tree_is_rejected_by_the_spine() -> None:
    """The other omitted check: `..` belongs to the bindings preflight.

    `ModelRoles` pins names to `[a-z0-9_]+`, so this cannot arrive through the
    production path — but the guard is what makes that a second line of defense
    rather than the only one.
    """
    with pytest.raises(ValueError, match="must not contain"):
        _preflight_bindings(train_backtest_bindings("../../etc/passwd"))
