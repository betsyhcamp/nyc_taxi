"""The composition spine where config fragments go to validated destination models.

`compose_config` is the entry point: it resolves one destination's ConfigBinding
list against `config_dir`, merges the fragments later-wins, validates once, and
returns the model alongside the metadata a run manifest records.

Five validation stages: load per fragment, merge recording value sources, validate
per destination, round-trip drop check, cross-field. Two preflights check
`compose_config`'s own arguments before stage 1.

This module is generic and carries no knowledge of this project's own config
files; `bindings.py` holds those.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel
from tsbricks.backtesting.schema import BacktestConfig, parse_config

from fcstnyctaxi.lib.config.loading import _load_config_file
from fcstnyctaxi.lib.io import write_text_to_gcs

RUNTIME_SOURCE = "<runtime>"
"""Label for values injected at runtime rather than read from a file.

Appears last in `ComposedConfig.config_files` and in any `value_sources` entry the
override supplied, so a computed value is distinguishable from a configured one.
"""


def _merge_with_value_sources(
    base: dict[str, Any], override: dict[str, Any], prefix: str = ""
) -> tuple[dict[str, Any], list[str]]:
    """Deep-merge `override` onto `base`, reporting the leaf paths it supplied.

    One recursion produces both results, which is what makes the reported paths a
    record of the merge rather than an independent traversal: they recurse
    wherever the merge recursed and stop wherever it replaced. Lists are replaced
    atomically, so a list is a leaf here which is unlike `_flatten_leaf_paths`.
    An empty mapping is a leaf in both, so the two agree on what a document's
    leaves are and differ only about lists.

    Args:
        base (dict): The accumulated document. Not mutated.
        override (dict): The layer to merge on top. Not mutated.
        prefix (str): Dotted path of `base` within the whole document.

    Returns:
        tuple[dict, list[str]]: The merged document, and the dotted leaf paths
            `override` supplied, in traversal order.
    """
    result = base.copy()
    supplied: list[str] = []

    for key, val in override.items():
        path = f"{prefix}{key}"
        if isinstance(val, dict):
            existing = result[key] if isinstance(result.get(key), dict) else {}
            result[key], nested = _merge_with_value_sources(existing, val, f"{path}.")
            supplied.extend(nested)
            if result[key] == {}:
                # An explicitly declared empty block is a leaf of the merged
                # document, and the recursion yields no paths for it. Without
                # this the manifest cannot tell `hyperparameters: {}` from a
                # fragment that never mentioned it. Guarded on the merged
                # result rather than on `val`, because `{}` landing on a
                # populated block is a no-op that supplied nothing.
                supplied.append(path)
        else:
            result[key] = val
            supplied.append(path)

    return result, supplied


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge `override` onto `base`, later-wins, lists replaced atomically.

    A thin wrapper over `_merge_with_value_sources`, which performs the single
    recursion both callers share. Keeping one recursion is what stops the merge
    and its record of itself from drifting apart. The observable contract here is
    unchanged and is pinned by regression tests.
    """
    return _merge_with_value_sources(base, override)[0]


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


# ================================================
# Dataclass declarations
# ================================================


@dataclass(frozen=True)
class ConfigBinding:
    """An association between a config file and the schema it feeds.

    Not config content, which is why it names a binding rather than the file it
    points at. `path` is **relative to `config_dir`** so a run manifest records
    the same string wherever the tree is mounted; `compose_config` resolves it.

    Attributes:
        path (Path): Relative to `config_dir`. No leading `/`, no `..`.
        destination (type[BaseModel]): The schema the merged document validates
            against.
        destination_key (str): The manifest's own `"<Schema>[:<model>]"` form, so
            a binding and the manifest agree by construction. Two bindings may
            share a `destination` and differ here, as one model's config and
            another's do.
        allowed_keys (frozenset[str] | None): A set narrower than the schema's,
            expressed as data so relaxing it later is an edit rather than a code
            change. None allows every top-level field the schema declares.
    """

    path: Path
    destination: type[BaseModel]
    destination_key: str
    allowed_keys: frozenset[str] | None = None


@dataclass(frozen=True)
class ComposedConfig:
    """One destination's validated model plus what a manifest records about it.

    The field is `config`, not `model`: `model` is this repo's most loaded domain
    word and `BacktestConfig` declares a `model` field of its own, so
    `composed.config.model.hyperparameters` reads where `composed.model.model`
    would not.

    Attributes:
        config (BaseModel): The validated instance.
        value_sources (dict[str, list[str]]): Dotted leaf path to the sources that
            supplied it, winner last — the full shadowing history, since
            winner-only cannot distinguish an overridden value from an unread file.
        config_files (list[str]): Sources in precedence order, `RUNTIME_SOURCE`
            last when overrides were applied.
    """

    config: BaseModel
    value_sources: dict[str, list[str]]
    config_files: list[str]


# ================================================
# Path helpers
# ================================================


def _flatten_leaf_paths(value: Any, path: str = "") -> list[str]:
    """Leaf paths of a nested document, descending both dicts and lists.

    Deliberately unlike `_merge_with_value_sources`, which stops at lists because
    the merge replaces them atomically. The drop check descends because it
    compares *serialized* leaves: an unaliased dump renames `transforms[0].class`
    to `transforms[0].class_path`, which a list-as-leaf walk could never see, and
    a key dropped inside a list element would go unreported.

    Empty dicts and empty lists are leaves such that an emptied block must stay visible.
    """
    if isinstance(value, dict) and value:
        return [
            leaf
            for key, val in value.items()
            for leaf in _flatten_leaf_paths(val, f"{path}.{key}" if path else str(key))
        ]
    if isinstance(value, list) and value:
        return [
            leaf
            for index, item in enumerate(value)
            for leaf in _flatten_leaf_paths(item, f"{path}[{index}]")
        ]
    return [path]


def _path_prefixes(path: str) -> list[str]:
    """Every ancestor of a leaf path, shallowest first, including the path itself.

    Splits at both separators, so `transforms[0].class` yields `transforms`,
    `transforms[0]`, `transforms[0].class`.
    """
    prefixes = [path[:index] for index, char in enumerate(path) if char in ".["]
    prefixes.append(path)
    return prefixes


def _collapse_to_shallowest_missing(
    missing_paths: list[str], surviving_prefixes: frozenset[str]
) -> list[str]:
    """Collapse dropped leaf paths to their shallowest missing ancestor.

    A dropped block reports once rather than once per leaf beneath it. For example,
    an unknown top-level `evaluation_periods` reports as `evaluation_periods`, not as
    its four leaves. A dropped key beside a surviving sibling still reports in full
    `data.bogus`, never `data` which is why `surviving_prefixes` is needed and why
    taking each path's first segment would be wrong. A path whose every ancestor
    survived is **skipped**, not reported.

    Args:
        missing_paths (list[str]): Leaf paths of the merged input that are not
            leaf paths of the dump, in input order.
        surviving_prefixes (frozenset[str]): Every leaf path the dump produced
            and every ancestor of one — `_path_prefixes` over the dump's leaves.

    Returns:
        list[str]: Reported paths, deduplicated, in first-seen order.
    """
    reported: list[str] = []
    for path in missing_paths:
        for prefix in _path_prefixes(path):
            if prefix not in surviving_prefixes:
                reported.append(prefix)
                break

    return list(dict.fromkeys(reported))


# ================================================
# The five validation stages
# ================================================


def _load_fragment(config_dir: Path, binding: ConfigBinding) -> dict[str, Any]:
    """Stage 1 — load one fragment and check its top-level keys.

    `_load_config_file` covers parsing, the non-empty mapping, and duplicate keys.
    This the key check which is worth having separately from the drop check since it
    attributes a bad key to a file, where the drop check sees only the merged whole.

    The allowed set is read from the schema's own fields rather than written out,
    so it picks up new fields automatically and cannot go stale. Top-level keys
    are all a partial document can be judged on.

    Args:
        config_dir (Path): Root the binding's relative path resolves against.
        binding (ConfigBinding): The fragment to load.

    Raises:
        FileNotFoundError: If the fragment does not exist.
        ValueError: On any load failure, or a top-level key outside the
            fragment's allowed set.

    Returns:
        dict[str, Any]: The parsed fragment, still unvalidated against the schema.
    """
    fragment = _load_config_file(config_dir / binding.path)

    if binding.allowed_keys is not None:
        unexpected = sorted(set(fragment) - binding.allowed_keys)
        if unexpected:
            raise ValueError(
                f"{binding.path} declares {unexpected}, outside the keys this "
                f"fragment may hold for {binding.destination_key}: "
                f"{sorted(binding.allowed_keys)}"
            )
        return fragment

    unexpected = sorted(set(fragment) - set(binding.destination.model_fields))
    if unexpected:
        raise ValueError(
            f"{binding.path} declares {unexpected}, which "
            f"{binding.destination_key} does not declare as a field"
        )

    return fragment


def _check_no_dropped_keys(
    merged: dict[str, Any], config: BaseModel, destination_key: str
) -> None:
    """Stage 4 — report keys the schema accepted and then silently discarded.

    Dumps the validated model and subtracts its leaf paths from the merged
    input's; whatever remains was accepted and thrown away. Two dump flags are
    load-bearing in opposite directions: `by_alias=True`, because an unaliased
    dump renames aliased fields and reports correct keys as dropped; and NOT
    `exclude_none`, because a fragment may declare `null` deliberately and an
    `exclude_none` dump omits those leaves.

    Exists only because `BacktestConfig` does not forbid extra keys. A destination
    that forbids them raises during validation instead, and this stage finds
    nothing.
    """
    dumped_paths = frozenset(_flatten_leaf_paths(config.model_dump(by_alias=True)))
    surviving = frozenset(
        prefix for path in dumped_paths for prefix in _path_prefixes(path)
    )
    missing = [path for path in _flatten_leaf_paths(merged) if path not in dumped_paths]

    dropped = _collapse_to_shallowest_missing(missing, surviving)
    if dropped:
        raise ValueError(
            f"{destination_key} accepted and silently discarded {dropped}; "
            "these paths are not declared by its schema"
        )


def _check_cross_field(config: BaseModel) -> None:
    """Stage 5 — invariants spanning fragments, which no single schema can see.

    `data.freq` and `model.hyperparameters.freq` arrive from different fragments
    and are each individually valid, so only a post-validation check can compare
    them. Conditional on the model *declaring* a freq: a callable may legitimately
    infer the frequency instead, so a universal requirement would be wrong.

    Dispatches on type rather than duck typing which a `getattr` chain no-ops silently
    when an attribute is renamed, and a check that can quietly stop running is
    worse than no check.
    """
    if not isinstance(config, BacktestConfig):
        return

    declared = (config.model.hyperparameters or {}).get("freq")
    if declared is not None and declared != config.data.freq:
        raise ValueError(
            f"model.hyperparameters.freq is {declared!r} but data.freq is "
            f"{config.data.freq!r}; the model would be fit at a different "
            "frequency than the panel is declared to carry"
        )


# ================================================
# Argument preflights — not a sixth stage
# ================================================


def _preflight_bindings(bindings: Sequence[ConfigBinding]) -> None:
    """Reject a bindings sequence `compose_config` cannot honor.

    Four rejections, each guarding a stated contract. `compose_config` is
    per-destination, so a mixed sequence has no single schema to validate against.
    Binding paths are relative to `config_dir`, which `pathlib` cannot enforce on
    its own — `config_dir / <absolute>` silently discards `config_dir`, so an
    absolute path composes a valid run and writes a machine-specific path into the
    manifest, which is what makes two runs of identical config non-comparable.
    `..` is rejected for the same portability reason and because two spellings of
    one path defeat the manifest's deduplication.

    Raises:
        ValueError: On an empty sequence, a mixed destination or destination_key,
            an absolute path, or a path containing `..`.
    """
    if not bindings:
        raise ValueError("compose_config requires at least one binding; got none")

    keys = {binding.destination_key for binding in bindings}
    if len(keys) > 1:
        raise ValueError(
            f"compose_config is per-destination; got destination_keys {sorted(keys)}"
        )

    destinations = {binding.destination for binding in bindings}
    if len(destinations) > 1:
        names = sorted(destination.__name__ for destination in destinations)
        raise ValueError(f"compose_config is per-destination; got destinations {names}")

    for binding in bindings:
        if binding.path.is_absolute():
            raise ValueError(
                "ConfigBinding path must be relative to config_dir, so the "
                f"manifest records the same string on every machine; got "
                f"{binding.path}"
            )
        if ".." in binding.path.parts:
            raise ValueError(
                f"ConfigBinding path must not contain '..'; got {binding.path}"
            )


def _preflight_overrides(
    overrides: dict[str, Any] | None, destination: type[BaseModel]
) -> None:
    """Check a runtime override's top-level keys against the destination.

    Top-level keys only, against the destination's own fields — never a binding's
    narrower `allowed_keys`, since runtime may fill any field of the destination
    while a narrowed set belongs to one file. An empty override is valid and means
    *merge nothing*.

    It earns its place because validation preempts the drop check: a mistyped
    top-level key beside a required sibling raises "field required" and points at
    the wrong thing, and the drop report that would have named the typo never runs.

    Raises:
        ValueError: If a top-level key is not a field of `destination`.
    """
    if not overrides:
        return

    unexpected = sorted(set(overrides) - set(destination.model_fields))
    if unexpected:
        raise ValueError(
            f"Runtime override declares {unexpected}, which "
            f"{destination.__name__} does not declare as a field"
        )


def compose_config(
    config_dir: Path,
    bindings: Sequence[ConfigBinding],
    overrides: dict[str, Any] | None = None,
) -> ComposedConfig:
    """Compose one destination from its fragments and any runtime override.

    Runs both preflights, then the five stages in order. Per-destination by
    contract: `ComposedConfig` holds one validated instance, and a manifest is
    assembled from several.

    Args:
        config_dir (Path): Root the bindings' relative paths resolve against.
        bindings (Sequence[ConfigBinding]): Fragments in precedence order, later
            wins. All must name the same destination.
        overrides (dict | None): Values injected at runtime, merged last and
            labeled `RUNTIME_SOURCE`.

    Raises:
        FileNotFoundError: If a fragment does not exist.
        ValueError: On a preflight rejection, a fragment-level failure, or a
            dropped or contradictory key.
        ValidationError: If the merged document does not satisfy the destination.

    Returns:
        ComposedConfig: The validated instance, its value sources, and its sources
            in precedence order.
    """
    _preflight_bindings(bindings)
    destination = bindings[0].destination
    destination_key = bindings[0].destination_key
    _preflight_overrides(overrides, destination)

    layers: list[tuple[str, dict[str, Any]]] = [
        (binding.path.as_posix(), _load_fragment(config_dir, binding))
        for binding in bindings
    ]
    if overrides:
        layers.append((RUNTIME_SOURCE, overrides))

    merged: dict[str, Any] = {}
    value_sources: dict[str, list[str]] = {}
    for source, layer in layers:
        merged, supplied = _merge_with_value_sources(merged, layer)
        for leaf_path in supplied:
            value_sources.setdefault(leaf_path, []).append(source)

    config = destination.model_validate(merged)
    _check_no_dropped_keys(merged, config, destination_key)
    _check_cross_field(config)

    return ComposedConfig(
        config=config,
        value_sources=value_sources,
        config_files=[source for source, _ in layers],
    )
