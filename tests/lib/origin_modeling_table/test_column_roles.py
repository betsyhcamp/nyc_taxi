from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from fcstnyctaxi.lib.origin_modeling_table.column_roles import (
    ModelingTableSchema,
    _require_unique_columns,
)

# ================================================
# Fixtures
#
# A miniature of the direct-month declaration: three keys, two features, one
# target, one passthrough, one progress column. Small enough that .columns can be
# written out by hand — k1, k2, k3, f1, f2, t, p, g in exactly that order.
# ================================================


@pytest.fixture
def schema() -> ModelingTableSchema:
    return ModelingTableSchema(
        key_cols=("k1", "k2", "k3"),
        feature_cols=("f1", "f2"),
        target_col="t",
        passthrough_cols=("p",),
        progress_cols=("g",),
    )


@pytest.fixture
def frame(schema: ModelingTableSchema) -> pd.DataFrame:
    """A conforming frame: the declared columns, in declared order, two rows."""
    return pd.DataFrame(
        [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]],
        columns=list(schema.columns),
    )


def _kwargs(**overrides) -> dict:
    """Valid constructor kwargs with the named fields replaced."""
    base = {
        "key_cols": ("k1",),
        "feature_cols": ("f1",),
        "target_col": "t",
        "passthrough_cols": (),
        "progress_cols": ("g",),
    }
    return {**base, **overrides}


# ================================================
# Construction — declaration invariants (§5)
#
# These run in __post_init__, so they cannot be skipped by a framing that never
# calls validate(). The tests assert on the CONSTRUCTOR, not on validate().
# ================================================


def test_construction_accepts_lists_and_coerces_them_to_tuples() -> None:
    """A list is accepted and stored as a tuple, so the role cannot be mutated."""
    s = ModelingTableSchema(**_kwargs(feature_cols=["f1", "f2"]))
    assert s.feature_cols == ("f1", "f2")
    assert isinstance(s.feature_cols, tuple)


def test_construction_accepts_an_empty_role() -> None:
    """A framing with no passthroughs is legitimate, not a declaration error."""
    assert ModelingTableSchema(**_kwargs(passthrough_cols=())).passthrough_cols == ()


def test_construction_raises_when_the_target_is_also_a_feature() -> None:
    """Target leakage: the model would be handed the answer as an input column."""
    with pytest.raises(ValueError, match=r"declared more than once"):
        ModelingTableSchema(**_kwargs(feature_cols=("f1", "t")))


def test_construction_raises_when_two_roles_share_a_column() -> None:
    """Any overlap, not just the target: here a key is also a progress column."""
    with pytest.raises(ValueError, match=r"declared more than once"):
        ModelingTableSchema(**_kwargs(key_cols=("k1", "g")))


def test_construction_raises_on_a_name_repeated_within_one_role() -> None:
    """A name repeated inside one role fails, not only a name shared across two."""
    with pytest.raises(ValueError, match=r"declared more than once"):
        ModelingTableSchema(**_kwargs(feature_cols=("f1", "f1")))


def test_collision_message_names_the_column_and_both_roles() -> None:
    """The message names the offending column and every role it was declared in."""
    with pytest.raises(ValueError) as excinfo:
        ModelingTableSchema(**_kwargs(feature_cols=("f1", "t")))
    message = str(excinfo.value)
    assert "'t'" in message
    assert "feature_cols" in message and "target_col" in message


def test_construction_raises_on_a_bare_string_role() -> None:
    """tuple("f1") would silently become ("f","1") and pass every later check."""
    with pytest.raises(ValueError, match=r"must be a list or tuple of str"):
        ModelingTableSchema(**_kwargs(feature_cols="f1"))


def test_construction_raises_on_a_set_role() -> None:
    """Set iteration order varies per process, and .columns order is contractual."""
    with pytest.raises(ValueError, match=r"must be a list or tuple of str"):
        ModelingTableSchema(**_kwargs(feature_cols={"f1", "f2"}))


def test_construction_raises_on_a_non_str_member() -> None:
    """A non-string would reach df.loc[:, [...]] as a column label."""
    with pytest.raises(ValueError, match=r"contains non-str entries"):
        ModelingTableSchema(**_kwargs(feature_cols=("f1", 42)))


def test_construction_raises_on_a_none_role() -> None:
    """tuple(None) would raise TypeError far from any mention of a declaration."""
    with pytest.raises(ValueError, match=r"must be a list or tuple of str"):
        ModelingTableSchema(**_kwargs(feature_cols=None))


def test_construction_raises_when_the_target_is_not_a_str() -> None:
    """A list target would put an unhashable element into .columns."""
    with pytest.raises(ValueError, match=r"target_col must be a str"):
        ModelingTableSchema(**_kwargs(target_col=["t"]))


def test_the_schema_is_frozen_after_construction() -> None:
    """Reassigning a role after construction raises rather than silently succeeding."""
    s = ModelingTableSchema(**_kwargs())
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.feature_cols = ("other",)  # type: ignore[misc]


# ================================================
# .columns
# ================================================


def test_columns_concatenates_the_roles_in_field_order(
    schema: ModelingTableSchema,
) -> None:
    """The order is contractual: validate() compares list(df.columns) against it."""
    assert schema.columns == ("k1", "k2", "k3", "f1", "f2", "t", "p", "g")


def test_columns_skips_an_empty_role() -> None:
    """An empty passthrough contributes nothing rather than a gap."""
    assert ModelingTableSchema(**_kwargs()).columns == ("k1", "f1", "t", "g")


# ================================================
# .select and .select_features
# ================================================


def test_select_projects_to_the_declared_columns_in_order(
    schema: ModelingTableSchema, frame: pd.DataFrame
) -> None:
    """Column order comes from the declaration, not from the source frame."""
    shuffled = frame[list(reversed(schema.columns))]
    assert list(schema.select(shuffled).columns) == list(schema.columns)


def test_select_drops_columns_the_schema_does_not_declare(
    schema: ModelingTableSchema, frame: pd.DataFrame
) -> None:
    """An undeclared column is removed, bounding the superset from the join."""
    selected = schema.select(frame.assign(undeclared=0))
    assert "undeclared" not in selected.columns


def test_select_features_returns_only_the_model_matrix(
    schema: ModelingTableSchema, frame: pd.DataFrame
) -> None:
    """X carries only feature_cols - no keys, target, passthrough or progress."""
    assert list(schema.select_features(frame).columns) == ["f1", "f2"]


def test_select_does_not_mutate_its_input(
    schema: ModelingTableSchema, frame: pd.DataFrame
) -> None:
    """Projection returns a new frame; the caller's is untouched."""
    before = frame.copy()
    schema.select(frame)
    pd.testing.assert_frame_equal(frame, before)


def test_select_raises_on_a_duplicated_source_column(
    schema: ModelingTableSchema, frame: pd.DataFrame
) -> None:
    """pandas EXPANDS rather than projects: 2 names asked for, 3 columns returned."""
    duplicated = pd.concat([frame, frame[["f1"]]], axis=1)
    with pytest.raises(ValueError, match=r"duplicate columns"):
        schema.select(duplicated)


def test_select_features_raises_on_a_duplicated_source_column(
    schema: ModelingTableSchema, frame: pd.DataFrame
) -> None:
    """Same hazard on the model matrix, where it would hand LightGBM an extra column."""
    duplicated = pd.concat([frame, frame[["f1"]]], axis=1)
    with pytest.raises(ValueError, match=r"duplicate columns"):
        schema.select_features(duplicated)


def test_select_raises_when_a_declared_column_is_absent(
    schema: ModelingTableSchema, frame: pd.DataFrame
) -> None:
    """A declared column absent from the frame raises a KeyError naming that column."""
    with pytest.raises(KeyError, match=r"f2"):
        schema.select(frame.drop(columns=["f2"]))


# ================================================
# .validate — frame invariants only (§5)
# ================================================


def test_validate_returns_none_on_a_conforming_frame(
    schema: ModelingTableSchema, frame: pd.DataFrame
) -> None:
    """A conforming frame passes and returns None rather than a boolean to test."""
    assert schema.validate(frame) is None


def test_validate_raises_on_a_column_added_after_selection(
    schema: ModelingTableSchema, frame: pd.DataFrame
) -> None:
    """The notebook-drift case: a stray cell adds a column after select()."""
    with pytest.raises(ValueError, match=r"unexpected: \['stray'\]"):
        schema.validate(frame.assign(stray=0))


def test_validate_raises_on_a_missing_column(
    schema: ModelingTableSchema, frame: pd.DataFrame
) -> None:
    """Reports which declaration is absent rather than two column lists to diff."""
    with pytest.raises(ValueError, match=r"missing: \['f2'\]"):
        schema.validate(frame.drop(columns=["f2"]))


def test_validate_raises_on_the_right_columns_in_the_wrong_order(
    schema: ModelingTableSchema, frame: pd.DataFrame
) -> None:
    """Same membership, different order gets its own message."""
    with pytest.raises(ValueError, match=r"different order"):
        schema.validate(frame[list(reversed(schema.columns))])


def test_validate_raises_on_a_frame_with_duplicate_columns(
    schema: ModelingTableSchema, frame: pd.DataFrame
) -> None:
    """A duplicated column fails validation, not only projection."""
    duplicated = pd.concat([frame, frame[["f1"]]], axis=1)
    with pytest.raises(ValueError, match=r"duplicate columns"):
        schema.validate(duplicated)


def test_validate_accepts_the_output_of_select(
    schema: ModelingTableSchema, frame: pd.DataFrame
) -> None:
    """The notebook idiom: select then validate, in one cell."""
    assert schema.validate(schema.select(frame.assign(undeclared=0))) is None


# ================================================
# _require_unique_columns
# ================================================


def test_require_unique_columns_passes_on_unique_names() -> None:
    """Returns None rather than raising when nothing is repeated."""
    assert _require_unique_columns(pd.DataFrame({"a": [1], "b": [2]})) is None


def test_require_unique_columns_names_every_repeated_column() -> None:
    """One call surfaces all duplicates rather than one fix-and-retry each."""
    df = pd.DataFrame([[1, 2, 3, 4]], columns=["a", "a", "b", "b"])
    with pytest.raises(ValueError) as excinfo:
        _require_unique_columns(df)
    assert "'a'" in str(excinfo.value) and "'b'" in str(excinfo.value)
