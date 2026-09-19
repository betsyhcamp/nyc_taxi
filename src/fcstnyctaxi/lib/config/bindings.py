# TODO: Once conventions are understood by the team, trim down comments
"""Per-slice binding declarations — which of this project's files feed which schema.

`composition.py` is the generic spine and names no tree; this module is its
opposite and names `config/base/data.yaml` and `config/train/models/<name>.yaml`
outright. That is why they are separate modules: the spine can be promoted to
tsbricks and this can never be. The rules the tree follows are in
``config/README.md``.

**Functions, not constants.** Two paths are parameterized — ``environments/<env>.yaml``
by the selector and ``train/models/<name>.yaml`` by the model set — and the model
set is read from ``TrainModelingConfig``, a destination that must be composed
first. Composition is therefore two-phase, and the phases are expressed here as
which function takes what:

    phase 1, no data     environment_bindings · train_infra_bindings ·
                         train_modeling_bindings
    phase 2, needs data  train_backtest_bindings(model_name)

There is deliberately no single ``train_bindings()`` spanning both: it would need
``model_names`` as a parameter while declaring the destination that produces
them.

**One function, one `compose_config` call.** Every tuple returned here carries a
single ``destination_key``, so it is always a legal argument to
``compose_config``, whose bindings preflight rejects a mixed sequence. Callers
never group, filter, or index a result.

**When a fragment narrows its allowed keys.** A fragment narrows when it is
*defined by the block it carries* — ``base/data.yaml`` is the data contract and
nothing else, ``train/models/<name>.yaml`` is one model and nothing else. A
schema-wide fragment does not: ``train/backtest.yaml``'s allowed set would be
everything ``BacktestConfig`` declares minus what the other two layers own, a
definition by subtraction that goes stale the moment tsbricks adds a field.

**No existence check and no name check here.** Binding functions are
declarations; they open nothing. A missing ``train/models/<name>.yaml`` raises
``FileNotFoundError`` naming the resolved path in the spine's stage 1; a ``..``
in a model name is rejected by the spine's bindings preflight; and ``ModelRoles``
already pins names to ``[a-z0-9_]+``. Each fault is already owned by the earliest
layer that can see it, so a copy here would move ownership without adding cover.

**Absence is a fact in code, not an inference from a missing directory.**
``feature_bindings`` and ``inference_bindings`` exist and return the environment
binding alone, which is what ``config/README.md``'s parity rule asks for — and it
is why a slice with no project-owned destinations never calls ``compose_config``
with an empty sequence.

``available_environments``, ``require_known_environment`` and
``resolve_model_names`` are the only functions here that touch the filesystem,
and the only ones that take ``config_dir``. The first two discover the
environment set; the third composes the destination that holds the model set.
Every other function is a pure declaration.
"""

from pathlib import Path
from typing import cast

from tsbricks.backtesting.schema import BacktestConfig

from fcstnyctaxi.lib.config.composition import ConfigBinding, compose_config
from fcstnyctaxi.schemas.config.environment import EnvironmentConfig
from fcstnyctaxi.schemas.config.train import (
    ModelRoles,
    TrainInfraConfig,
    TrainModelingConfig,
)

# The tree's layout, in one block. Every path is RELATIVE to config_dir, which
# is what lets the same binding resolve under tmp_path, from the working tree,
# and at /app/config — and what keeps a manifest comparable across machines.
_ENVIRONMENTS_DIR = Path("environments")
_DATA_FRAGMENT = Path("base/data.yaml")
_TRAIN_INFRA_FRAGMENT = Path("train/infra.yaml")
_TRAIN_MODELING_FRAGMENT = Path("train/modeling.yaml")
_TRAIN_BACKTEST_FRAGMENT = Path("train/backtest.yaml")
_TRAIN_MODELS_DIR = Path("train/models")


# ================================================
# The environment selector
# ================================================


def available_environments(config_dir: Path) -> list[str]:
    """Environment names, discovered from the files that define them.

    The allowed set is deployment data, not code structure, so it is read from
    the filesystem — the actual authority — and cannot drift from the tree.
    Adding an environment is adding a file, with no code edit and no new branch.

    ``.yaml`` only, deliberately narrower than ``_load_config_file``'s
    ``.yaml``/``.yml``: a ``prod.yml`` that loads when named explicitly yet never
    appears here would exist and be unreachable. The narrower rule fails as
    ``Unknown env 'prod'; available: dev``, which names the problem at once.

    Args:
        config_dir (Path): Root of the config tree.

    Returns:
        list[str]: Environment names, sorted. Empty if the directory has no
            ``.yaml`` files or does not exist.
    """
    return sorted(path.stem for path in (config_dir / _ENVIRONMENTS_DIR).glob("*.yaml"))


def require_known_environment(config_dir: Path, env: str) -> None:
    """Reject an ``env`` selector with no file behind it, naming the alternatives.

    A guard, not a resolver: it returns nothing and selects nothing, so a caller
    still reads the environment through ``environment_bindings`` and the
    precedence stays visible at the call site. It lives here so the message is
    written once rather than in each caller that needs it — 0a's runner, the
    impl, and 0b's wrapper.

    The empty case gets its own message because the two failures are different
    and the wrong one is misleading: ``available:`` followed by nothing almost
    always means ``config_dir`` is wrong, not that the environment is.

    Args:
        config_dir (Path): Root of the config tree.
        env (str): The selector to check.

    Raises:
        ValueError: If no ``environments/<env>.yaml`` exists.
    """
    available = available_environments(config_dir)

    if not available:
        raise ValueError(
            f"No environments are defined under {config_dir / _ENVIRONMENTS_DIR}; "
            f"cannot select {env!r}"
        )
    if env not in available:
        raise ValueError(f"Unknown env {env!r}; available: {', '.join(available)}")


# ================================================
# Binding declarations, pure: every function in this section opens nothing
# ================================================


def environment_bindings(env: str) -> tuple[ConfigBinding, ...]:
    """EnvironmentConfig from ``environments/<env>.yaml`` — every slice composes it.

    Plural, returning a one-element tuple rather than a bare binding, so that
    every public function here returns something directly passable to
    ``compose_config`` and no call site wraps.

    ``env`` is a selector and appears in no config file: the path is the binding.
    That is why ``EnvironmentConfig`` declares no ``env`` field and why the
    selected environment is recorded in the emitted manifest instead.
    """
    return (
        ConfigBinding(
            path=_ENVIRONMENTS_DIR / f"{env}.yaml",
            destination=EnvironmentConfig,
            destination_key=EnvironmentConfig.__name__,
        ),
    )


def train_infra_bindings() -> tuple[ConfigBinding, ...]:
    """TrainInfraConfig from ``train/infra.yaml`` — one environment-independent file.

    One fragment, per the parity rule: a category that exists has exactly one
    destination and exactly one fragment. No ``allowed_keys``, because the file
    is the whole destination rather than one block of it.
    """
    return (
        ConfigBinding(
            path=_TRAIN_INFRA_FRAGMENT,
            destination=TrainInfraConfig,
            destination_key=TrainInfraConfig.__name__,
        ),
    )


def train_modeling_bindings() -> tuple[ConfigBinding, ...]:
    """TrainModelingConfig from ``train/modeling.yaml`` — one file.

    Phase 1, and the destination phase 2 depends on: its ``model_roles`` becomes
    the model set through ``model_names_from_roles``, and each name there is a
    ``train/models/<name>.yaml`` stem.
    """
    return (
        ConfigBinding(
            path=_TRAIN_MODELING_FRAGMENT,
            destination=TrainModelingConfig,
            destination_key=TrainModelingConfig.__name__,
        ),
    )


def train_backtest_bindings(model_name: str) -> tuple[ConfigBinding, ...]:
    """BacktestConfig for one model which is the only destination that layers.

    Three fragments in precedence order, later wins: the shared data contract,
    the backtest settings, then the model. ``base/data.yaml`` and the model file
    each narrow their allowed keys, because each is defined by the one block it
    carries; ``train/backtest.yaml`` does not.

    Narrowing ``base/data.yaml`` is load-bearing rather than tidy.
    ``BacktestConfig`` does not forbid extra keys, so without it this fragment
    could legally declare any field of that schema — a stray ``model:`` block
    there would silently apply to **every** model.

    ``destination_key`` is computed once and shared by all three, so two models
    can never reach one ``compose_config`` call. That is the cross-contamination
    the spine's preflight exists to reject, made unreachable from this module.

    ``cross_validation.forecast_origins`` is the one runtime override and arrives
    as ``compose_config``'s third argument, not from here: it is derived from the
    panel, so no file in the tree can hold it.

    Args:
        model_name (str): A ``train/models/<name>.yaml`` stem, from
            ``model_names_from_roles``.
    """
    destination_key = f"{BacktestConfig.__name__}:{model_name}"

    return (
        ConfigBinding(
            path=_DATA_FRAGMENT,
            destination=BacktestConfig,
            destination_key=destination_key,
            allowed_keys=frozenset({"data"}),
        ),
        ConfigBinding(
            path=_TRAIN_BACKTEST_FRAGMENT,
            destination=BacktestConfig,
            destination_key=destination_key,
        ),
        ConfigBinding(
            path=_TRAIN_MODELS_DIR / f"{model_name}.yaml",
            destination=BacktestConfig,
            destination_key=destination_key,
            allowed_keys=frozenset({"model"}),
        ),
    )


def feature_bindings(env: str) -> tuple[ConfigBinding, ...]:
    """Feature composes EnvironmentConfig and nothing else.

    Not a placeholder. ``schemas/config/feature.py`` names the two destinations
    that belong here which are ``FeatureInfraConfig`` and ``FeatureModelingConfig`` and
    records why they are not written: the split they encode is assigned to
    Feature's own build. Stating the absence here is what the parity rule asks
    for, and it is why absence never takes the form of an empty binding sequence.
    """
    return environment_bindings(env)


def inference_bindings(env: str) -> tuple[ConfigBinding, ...]:
    """Inference composes EnvironmentConfig and nothing else.

    ``schemas/config/inference.py`` records the reason, and it is not Feature's:
    Inference has a fourth configuration layer with no counterpart in the other
    slices — the registered model's own composed config, read from the Model
    Registry. Writing its destinations now would fix a precedence order that
    layer may well change.
    """
    return environment_bindings(env)


# ================================================
# The model set
# ================================================


def model_names_from_roles(model_roles: ModelRoles) -> tuple[str, ...]:
    """The model set: ``model_roles`` values, deduplicated, in declaration order.

    One name per ``train/models/<name>.yaml`` to compose. A name in both roles
    composes once — ``benchmark == challenger`` is a legitimate smoke test — and
    deduplicating here rather than downstream is what stops the compile-time
    loop emitting two KFP tasks with one name, a DAG error several PRs from the
    line of ``train/modeling.yaml`` that caused it.
    """
    # model_dump() yields fields in declaration order; dict.fromkeys dedups
    # without losing it. sorted(set(...)) passes today only because the shipped
    # roles are alphabetical. Order is contract: benchmark first.
    return tuple(dict.fromkeys(model_roles.model_dump().values()))


def resolve_model_names(config_dir: Path) -> tuple[str, ...]:
    """The model set, composed from ``train/modeling.yaml`` under ``config_dir``.

    Takes no ``env``: the compile script that reads it has none.

    Args:
        config_dir (Path): Root of the config tree.

    Raises:
        FileNotFoundError: If ``train/modeling.yaml`` does not exist.
        ValueError: On any composition failure.
        ValidationError: If the file does not satisfy ``TrainModelingConfig``.

    Returns:
        tuple[str, ...]: Model names, deduplicated, in declaration order.
    """
    modeling = cast(
        TrainModelingConfig,
        compose_config(config_dir, train_modeling_bindings()).config,
    )
    return model_names_from_roles(modeling.model_roles)
