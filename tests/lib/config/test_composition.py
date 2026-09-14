from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pytest_mock import MockerFixture
from tsbricks.backtesting.schema import BacktestConfig

from fcstnyctaxi.lib.config.composition import (
    RUNTIME_SOURCE,
    ComposedConfig,
    ConfigBinding,
    _check_cross_field,
    _check_no_dropped_keys,
    _collapse_to_shallowest_missing,
    _deep_merge,
    _flatten_leaf_paths,
    _merge_with_value_sources,
    _path_prefixes,
    compose_config,
    merge_configs,
    save_config,
)

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
    mock_write = mocker.patch("fcstnyctaxi.lib.config.composition.write_text_to_gcs")
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


# ================================================
# The synthetic fixture slice
#
# A pretend project's config tree and destination schemas, so the spine is
# exercised without a single binding naming a real file in config/. That is what
# demonstrates the module is generic rather than merely asserting it.
# ================================================


class _WidgetSettings(BaseModel):
    """A nested block, so merges and drops have somewhere to happen below root."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    size: int = 1


class _SyntheticInfra(BaseModel):
    """A strict destination — the shape a project-owned schema takes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    widget: _WidgetSettings


class _SyntheticStep(BaseModel):
    """An aliased field inside a list, mirroring TransformConfig's `class`."""

    model_config = ConfigDict(populate_by_name=True)

    name: str
    class_path: str = Field(alias="class")


class _SyntheticNested(BaseModel):
    """A permissive nested block — where the drop check actually earns its keep.

    Stage 1 checks top-level keys, so an unknown key at root never reaches the
    drop check. What reaches it is an unknown key *below* root, in a block whose
    schema does not forbid extras.
    """

    kept: str


class _SyntheticModeling(BaseModel):
    """A permissive destination — no `extra="forbid"`, as BacktestConfig has none.

    This is the shape the drop check exists for: unknown keys are accepted and
    silently discarded rather than raising during validation.
    """

    label: str
    widget: _WidgetSettings
    nested: _SyntheticNested | None = None
    steps: list[_SyntheticStep] | None = None
    note: str | None = None
    origins: list[str] | None = None


_SHARED_FRAGMENT = Path("base/widget.yaml")
_INFRA_FRAGMENT = Path("slice/infra.yaml")
_MODELING_FRAGMENT = Path("slice/modeling.yaml")
_MODEL_FRAGMENT = Path("slice/models/one.yaml")


@pytest.fixture
def synthetic_config_dir(tmp_path: Path) -> Path:
    """Write a synthetic config tree: two destinations sharing one fragment.

    `base/widget.yaml` feeds both, which is what makes the shared-fragment and
    shadowing cases testable; `slice/models/one.yaml` is the narrowed fragment.
    """
    config_dir = tmp_path / "config"
    (config_dir / "base").mkdir(parents=True)
    (config_dir / "slice" / "models").mkdir(parents=True)

    fragments = {
        _SHARED_FRAGMENT: {"widget": {"name": "alpha", "size": 3}},
        _INFRA_FRAGMENT: {"label": "infra-label"},
        _MODELING_FRAGMENT: {
            "label": "modeling-label",
            "note": None,
            "steps": [{"name": "step-one", "class": "pkg.Alpha"}],
        },
        _MODEL_FRAGMENT: {"widget": {"size": 9}},
    }
    for relative_path, content in fragments.items():
        (config_dir / relative_path).write_text(yaml.dump(content))

    return config_dir


def _infra_bindings() -> list[ConfigBinding]:
    """Bindings for the single-layer strict destination."""
    return [
        ConfigBinding(_SHARED_FRAGMENT, _SyntheticInfra, "_SyntheticInfra"),
        ConfigBinding(_INFRA_FRAGMENT, _SyntheticInfra, "_SyntheticInfra"),
    ]


def _modeling_bindings() -> list[ConfigBinding]:
    """Bindings for the three-layer permissive destination, narrowest last."""
    return [
        ConfigBinding(_SHARED_FRAGMENT, _SyntheticModeling, "_SyntheticModeling:one"),
        ConfigBinding(_MODELING_FRAGMENT, _SyntheticModeling, "_SyntheticModeling:one"),
        ConfigBinding(
            _MODEL_FRAGMENT,
            _SyntheticModeling,
            "_SyntheticModeling:one",
            allowed_keys=frozenset({"widget"}),
        ),
    ]


# ================================================
# _merge_with_value_sources — the recording merge
# ================================================


def test_merge_with_value_sources_reports_scalar_leaf() -> None:
    """A scalar the override supplied is reported at its dotted path."""
    merged, supplied = _merge_with_value_sources({}, {"data": {"freq": "W-SUN"}})
    assert merged == {"data": {"freq": "W-SUN"}}
    assert supplied == ["data.freq"]


def test_merge_with_value_sources_enumerates_a_replaced_block() -> None:
    """A block written where base had nothing still reports its leaves.

    Stopping at `data` would make shadowing invisible: a later fragment
    overriding `data.freq` could not be seen to have overridden anything.
    """
    _, supplied = _merge_with_value_sources({}, {"data": {"freq": "W-SUN", "id": "u"}})
    assert sorted(supplied) == ["data.freq", "data.id"]


def test_merge_with_value_sources_treats_a_list_as_one_leaf() -> None:
    """Lists are replaced atomically, so the merge stops at the list itself."""
    _, supplied = _merge_with_value_sources({}, {"origins": ["a", "b"]})
    assert supplied == ["origins"]


def test_merge_with_value_sources_reports_only_what_the_override_supplied() -> None:
    """A sibling already in base is not reported — this layer did not supply it."""
    base = {"widget": {"name": "alpha", "size": 3}}
    _, supplied = _merge_with_value_sources(base, {"widget": {"size": 9}})
    assert supplied == ["widget.size"]


def test_merge_with_value_sources_reports_an_explicitly_empty_block() -> None:
    """An empty mapping is a leaf, so declaring one is distinguishable from silence.

    Without this the manifest cannot tell `hyperparameters: {}` from a fragment
    that never mentioned it — the same failure winner-only value sources have.
    """
    merged, supplied = _merge_with_value_sources({}, {"model": {"hyperparameters": {}}})
    assert merged == {"model": {"hyperparameters": {}}}
    assert supplied == ["model.hyperparameters"]


def test_merge_with_value_sources_ignores_an_empty_block_over_a_populated_one() -> None:
    """`{}` onto a populated block is a no-op, so this layer supplied nothing.

    The discriminating case: guarding on `val` rather than on the merged result
    passes the empty-block test above and puts a false entry in the manifest here.
    """
    base = {"model": {"hyperparameters": {"freq": "W-SUN"}}}
    merged, supplied = _merge_with_value_sources(
        base, {"model": {"hyperparameters": {}}}
    )
    assert merged == base
    assert supplied == []


def test_merge_with_value_sources_reports_a_deeply_nested_empty_block() -> None:
    """The empty-block rule applies at the recursion level that emptied out.

    `a` is not reported — it holds `b` — while `a.b` is, which is what proves the
    guard is not merely correct one layer down from the root.
    """
    _, supplied = _merge_with_value_sources({}, {"a": {"b": {}}})
    assert supplied == ["a.b"]


# ================================================
# Path helpers
# ================================================


def test_flatten_leaf_paths_descends_lists_with_indexes() -> None:
    """List elements are indexed, so two elements stay distinguishable."""
    document = {"steps": [{"name": "a"}, {"name": "b"}]}
    assert _flatten_leaf_paths(document) == ["steps[0].name", "steps[1].name"]


def test_flatten_leaf_paths_treats_empty_containers_as_leaves() -> None:
    """An emptied block stays visible rather than vanishing from the walk."""
    assert sorted(_flatten_leaf_paths({"a": {}, "b": [], "c": 1})) == ["a", "b", "c"]


def test_path_prefixes_splits_at_both_separators() -> None:
    """An indexed path yields its list, its element, and itself."""
    assert _path_prefixes("transforms[0].class") == [
        "transforms",
        "transforms[0]",
        "transforms[0].class",
    ]


def test_path_prefixes_of_a_root_key_is_itself() -> None:
    """A single-segment path has no ancestors but itself."""
    assert _path_prefixes("evaluation_periods") == ["evaluation_periods"]


def test_collapse_reports_a_dropped_block_once() -> None:
    """A whole dropped block collapses to its shallowest missing ancestor."""
    missing = ["evaluation_periods.n_start_months", "evaluation_periods.step"]
    surviving = frozenset({"data", "data.freq"})
    assert _collapse_to_shallowest_missing(missing, surviving) == ["evaluation_periods"]


def test_collapse_keeps_a_dropped_key_beside_a_surviving_sibling() -> None:
    """`data.bogus` must not collapse to `data`, which survived."""
    surviving = frozenset({"data", "data.freq"})
    assert _collapse_to_shallowest_missing(["data.bogus"], surviving) == ["data.bogus"]


def test_collapse_deduplicates_in_first_seen_order() -> None:
    """Two leaves under one dropped block report that block once."""
    missing = ["block.one", "block.two", "other.x"]
    surviving = frozenset({"kept"})
    assert _collapse_to_shallowest_missing(missing, surviving) == ["block", "other"]


# ================================================
# Stage 1 — per-fragment load and key check
# ================================================


def test_compose_config_rejects_a_key_outside_a_narrowed_allowed_set(
    synthetic_config_dir: Path,
) -> None:
    """A narrowed fragment declaring an extra key is named with its own path."""
    (synthetic_config_dir / _MODEL_FRAGMENT).write_text(
        yaml.dump({"widget": {"size": 9}, "label": "sneaky"})
    )
    with pytest.raises(ValueError, match="slice/models/one.yaml"):
        compose_config(synthetic_config_dir, _modeling_bindings())


def test_compose_config_rejects_a_key_the_destination_does_not_declare(
    synthetic_config_dir: Path,
) -> None:
    """An un-narrowed fragment is checked against the generated schema mirror."""
    (synthetic_config_dir / _INFRA_FRAGMENT).write_text(
        yaml.dump({"label": "infra-label", "not_a_field": 1})
    )
    with pytest.raises(ValueError, match="not_a_field"):
        compose_config(synthetic_config_dir, _infra_bindings())


def test_compose_config_propagates_a_missing_fragment(
    synthetic_config_dir: Path,
) -> None:
    """A binding naming a file that does not exist raises from the loader."""
    (synthetic_config_dir / _INFRA_FRAGMENT).unlink()
    with pytest.raises(FileNotFoundError):
        compose_config(synthetic_config_dir, _infra_bindings())


# ================================================
# compose_config — the composed result
# ================================================


def test_compose_config_validates_a_two_fragment_destination(
    synthetic_config_dir: Path,
) -> None:
    """Two fragments compose into one validated instance of the destination."""
    composed = compose_config(synthetic_config_dir, _infra_bindings())
    assert isinstance(composed, ComposedConfig)
    assert isinstance(composed.config, _SyntheticInfra)
    assert composed.config.label == "infra-label"
    assert composed.config.widget.size == 3


def test_compose_config_layers_three_fragments_later_wins(
    synthetic_config_dir: Path,
) -> None:
    """The narrowed model fragment overrides a value the shared fragment set."""
    composed = compose_config(synthetic_config_dir, _modeling_bindings())
    assert composed.config.widget.size == 9
    assert composed.config.widget.name == "alpha"


def test_compose_config_records_config_files_in_precedence_order(
    synthetic_config_dir: Path,
) -> None:
    """Sources are recorded as relative POSIX paths, in merge order."""
    composed = compose_config(synthetic_config_dir, _modeling_bindings())
    assert composed.config_files == [
        "base/widget.yaml",
        "slice/modeling.yaml",
        "slice/models/one.yaml",
    ]


def test_compose_config_records_the_full_shadowing_history(
    synthetic_config_dir: Path,
) -> None:
    """A shadowed leaf lists every fragment that supplied it, winner last."""
    composed = compose_config(synthetic_config_dir, _modeling_bindings())
    assert composed.value_sources["widget.size"] == [
        "base/widget.yaml",
        "slice/models/one.yaml",
    ]
    assert composed.value_sources["widget.name"] == ["base/widget.yaml"]


def test_compose_config_merges_a_runtime_override_last(
    synthetic_config_dir: Path,
) -> None:
    """An override wins over every file and is labelled, not given a path."""
    composed = compose_config(
        synthetic_config_dir, _modeling_bindings(), overrides={"label": "injected"}
    )
    assert composed.config.label == "injected"
    assert composed.config_files[-1] == RUNTIME_SOURCE
    assert composed.value_sources["label"] == [
        "slice/modeling.yaml",
        RUNTIME_SOURCE,
    ]


def test_compose_config_accepts_an_empty_override(
    synthetic_config_dir: Path,
) -> None:
    """`overrides={}` means merge nothing; rejecting it would burden callers."""
    composed = compose_config(synthetic_config_dir, _infra_bindings(), overrides={})
    assert composed.config_files == ["base/widget.yaml", "slice/infra.yaml"]


def test_compose_config_raises_when_the_merged_document_is_invalid(
    synthetic_config_dir: Path,
) -> None:
    """Stage 3 is the real schema: a missing required field raises."""
    (synthetic_config_dir / _INFRA_FRAGMENT).write_text(yaml.dump({"widget": {}}))
    with pytest.raises(ValidationError):
        compose_config(synthetic_config_dir, _infra_bindings())


# ================================================
# Preflights — arguments, checked before stage 1
# ================================================


def test_preflight_rejects_an_empty_bindings_sequence(tmp_path: Path) -> None:
    """With no bindings there is no destination and nothing to compose."""
    with pytest.raises(ValueError, match="at least one binding"):
        compose_config(tmp_path, [])


def test_preflight_rejects_a_mixed_destination_key(synthetic_config_dir: Path) -> None:
    """Two models' configs in one call would silently cross-contaminate."""
    bindings = [
        ConfigBinding(_SHARED_FRAGMENT, _SyntheticModeling, "_SyntheticModeling:one"),
        ConfigBinding(_MODELING_FRAGMENT, _SyntheticModeling, "_SyntheticModeling:two"),
    ]
    with pytest.raises(ValueError, match="per-destination"):
        compose_config(synthetic_config_dir, bindings)


def test_preflight_rejects_a_mixed_destination(synthetic_config_dir: Path) -> None:
    """Stage 3 has one schema to validate against, so the list must agree."""
    bindings = [
        ConfigBinding(_SHARED_FRAGMENT, _SyntheticInfra, "shared"),
        ConfigBinding(_INFRA_FRAGMENT, _SyntheticModeling, "shared"),
    ]
    with pytest.raises(ValueError, match="per-destination"):
        compose_config(synthetic_config_dir, bindings)


def test_preflight_rejects_an_absolute_binding_path(
    synthetic_config_dir: Path,
) -> None:
    """pathlib discards config_dir for an absolute path, so the guard is here.

    Without it the run succeeds and writes a machine-specific path into the
    manifest, which is a wrong record rather than an error.
    """
    absolute = synthetic_config_dir / _INFRA_FRAGMENT
    bindings = [ConfigBinding(absolute, _SyntheticInfra, "_SyntheticInfra")]
    with pytest.raises(ValueError, match="must be relative"):
        compose_config(synthetic_config_dir, bindings)


def test_preflight_rejects_a_dot_dot_binding_path(synthetic_config_dir: Path) -> None:
    """`..` escapes the tree and gives one file two manifest spellings."""
    bindings = [
        ConfigBinding(
            Path("slice/../base/widget.yaml"), _SyntheticInfra, "_SyntheticInfra"
        )
    ]
    with pytest.raises(ValueError, match=r"must not contain"):
        compose_config(synthetic_config_dir, bindings)


def test_preflight_rejects_an_unknown_override_key(
    synthetic_config_dir: Path,
) -> None:
    """A mistyped top-level override is named here, not diagnosed downstream.

    Validation would otherwise raise about a missing required sibling and point
    at the wrong thing, and the drop check would never run.
    """
    with pytest.raises(ValueError, match="not_a_field"):
        compose_config(
            synthetic_config_dir, _infra_bindings(), overrides={"not_a_field": 1}
        )


# ================================================
# Stage 4 — the round-trip drop check
# ================================================


def test_drop_check_reports_a_nested_key_the_schema_discarded(
    synthetic_config_dir: Path,
) -> None:
    """A key below root is accepted, discarded, and reported by name.

    Below root is the drop check's whole domain: stage 1 already rejects an
    unknown *top-level* key, and it names the file when it does.
    """
    (synthetic_config_dir / _MODELING_FRAGMENT).write_text(
        yaml.dump({"label": "m", "nested": {"kept": "yes", "discarded": "gone"}})
    )
    bindings = [
        ConfigBinding(_SHARED_FRAGMENT, _SyntheticModeling, "_SyntheticModeling"),
        ConfigBinding(_MODELING_FRAGMENT, _SyntheticModeling, "_SyntheticModeling"),
    ]
    with pytest.raises(ValueError, match=r"nested\.discarded"):
        compose_config(synthetic_config_dir, bindings)


def test_drop_check_keeps_a_surviving_sibling_out_of_the_report(
    synthetic_config_dir: Path,
) -> None:
    """`nested.kept` survives, so the report must not collapse to `nested`."""
    (synthetic_config_dir / _MODELING_FRAGMENT).write_text(
        yaml.dump({"label": "m", "nested": {"kept": "yes", "discarded": "gone"}})
    )
    bindings = [
        ConfigBinding(_SHARED_FRAGMENT, _SyntheticModeling, "_SyntheticModeling"),
        ConfigBinding(_MODELING_FRAGMENT, _SyntheticModeling, "_SyntheticModeling"),
    ]
    with pytest.raises(ValueError) as caught:
        compose_config(synthetic_config_dir, bindings)
    assert "nested.kept" not in str(caught.value)


def test_drop_check_requires_by_alias_on_the_dump() -> None:
    """An unaliased dump renames `steps[0].class` and would report it dropped.

    The assertions on both dumps are the point: with a list-as-leaf walk the two
    would be indistinguishable and this test could not fail.
    """
    merged = {
        "label": "m",
        "widget": {"name": "alpha"},
        "steps": [{"name": "step-one", "class": "pkg.Alpha"}],
    }
    config = _SyntheticModeling.model_validate(merged)

    aliased = set(_flatten_leaf_paths(config.model_dump(by_alias=True)))
    unaliased = set(_flatten_leaf_paths(config.model_dump()))
    assert "steps[0].class" in aliased
    assert "steps[0].class" not in unaliased
    assert "steps[0].class_path" in unaliased

    _check_no_dropped_keys(merged, config, "_SyntheticModeling")


def test_drop_check_must_not_exclude_none_from_the_dump() -> None:
    """`note: null` is a deliberate declaration an exclude_none dump would lose."""
    merged = {"label": "m", "widget": {"name": "alpha"}, "note": None}
    config = _SyntheticModeling.model_validate(merged)

    kept = set(_flatten_leaf_paths(config.model_dump(by_alias=True)))
    excluded = set(
        _flatten_leaf_paths(config.model_dump(by_alias=True, exclude_none=True))
    )
    assert "note" in kept
    assert "note" not in excluded

    _check_no_dropped_keys(merged, config, "_SyntheticModeling")


def test_drop_check_passes_on_a_faithfully_composed_config(
    synthetic_config_dir: Path,
) -> None:
    """Nothing is reported when every input key survives validation."""
    composed = compose_config(synthetic_config_dir, _modeling_bindings())
    assert composed.config.steps is not None
    assert composed.config.steps[0].class_path == "pkg.Alpha"


# ================================================
# Stage 5 — cross-field invariants
# ================================================


def test_cross_field_passes_when_the_model_declares_no_freq() -> None:
    """A callable may infer the frequency, so declaring none is legitimate."""
    _check_cross_field(merge_configs(_MINIMAL_CONFIG))


def test_cross_field_passes_when_the_two_freqs_agree() -> None:
    """A declared freq matching data.freq is the intended configuration."""
    agreeing = {"model": {"hyperparameters": {"freq": "W-SAT"}}}
    _check_cross_field(merge_configs(_MINIMAL_CONFIG, agreeing))


def test_cross_field_raises_when_the_two_freqs_disagree() -> None:
    """Both fragments are individually valid, so only a cross-check sees this."""
    disagreeing = {"model": {"hyperparameters": {"freq": "MS"}}}
    with pytest.raises(ValueError, match="data.freq"):
        _check_cross_field(merge_configs(_MINIMAL_CONFIG, disagreeing))


def test_cross_field_ignores_destinations_it_does_not_govern() -> None:
    """The freq invariant is BacktestConfig's; other destinations pass through."""
    _check_cross_field(_SyntheticInfra(label="x", widget=_WidgetSettings(name="alpha")))
