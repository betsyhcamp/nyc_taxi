from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ModelingTableSchema:
    """Declare which columns of a modeling table play which role.

    Roles are declared locally in a notebook but applied by shared library code. A
    column cannot hold two roles which protects against issues like target leakage at
    the earliest point in the code. Check runs at construction rather than in
    validate(), so it cannot be skipped. Frame conformance is a separate concern and
    lives in validate().

    Args:
        key_cols: Columns that identify a row. Must be a superkey — no two rows
            may share the same values — and should express the intended grain
            rather than the minimal set that happens to be unique today.
        feature_cols: The model matrix, and nothing else.
        target_col: The column being predicted.
        passthrough_cols: Carried through the table but not modeled on, and not
            identifying: values needed downstream, and join addresses.
        progress_cols: Intramonth position, carried through the table for gate
            arithmetic and reporting rather than modeled on.
    """

    key_cols: tuple[str, ...]
    feature_cols: tuple[str, ...]
    target_col: str
    passthrough_cols: tuple[str, ...]
    progress_cols: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "key_cols",
            "feature_cols",
            "passthrough_cols",
            "progress_cols",
        ):
            value = getattr(self, field_name)
            # list/tuple rather than "any iterable": tuple("lag1") silently becomes
            # ("l","a","g","1") and passes every later check, and a set would make
            # column order vary between processes, which .columns cannot tolerate.
            if not isinstance(value, list | tuple):
                raise ValueError(
                    f"{field_name} must be a list or tuple of str, got "
                    f"{type(value).__name__}"
                )
            non_str = [v for v in value if not isinstance(v, str)]
            if non_str:
                raise ValueError(f"{field_name} contains non-str entries: {non_str}")
            object.__setattr__(self, field_name, tuple(value))

        if not isinstance(self.target_col, str):
            raise ValueError(
                f"target_col must be a str, got {type(self.target_col).__name__}"
            )

        # Setup for disjoint column role check
        roles_by_column: dict[str, list[str]] = {}
        for role, names in (
            ("key_cols", self.key_cols),
            ("feature_cols", self.feature_cols),
            ("target_col", (self.target_col,)),
            ("passthrough_cols", self.passthrough_cols),
            ("progress_cols", self.progress_cols),
        ):
            for name in names:
                existing_roles = roles_by_column.setdefault(name, [])
                existing_roles.append(role)
        # check that a single column isn't listed in multiple role groups
        offenders = {c: roles for c, roles in roles_by_column.items() if len(roles) > 1}
        if offenders:
            raise ValueError(f"columns declared more than once: {offenders}")

    @property
    def columns(self) -> tuple[str, ...]:
        """Every declared column, in field order. The order is contractual."""
        return (
            self.key_cols
            + self.feature_cols
            + (self.target_col,)
            + self.passthrough_cols
            + self.progress_cols
        )

    def select(self, df: pd.DataFrame) -> pd.DataFrame:
        """Project df to the declared columns, in declared order."""
        _require_unique_columns(df)
        return df.loc[:, list(self.columns)]

    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Project df to the model matrix; X's columns are correct by construction."""
        _require_unique_columns(df)
        return df.loc[:, list(self.feature_cols)]

    def validate(self, df: pd.DataFrame) -> None:
        """Assert df carries exactly the declared columns, in declared order.

        Catches drift between the declaration and a frame that has already been
        selected through it — a stray cell adding a column in a notebook whose
        cells re-run in arbitrary order. It cannot live inside select(), which
        would be comparing its own output to its own input.

        Raises:
            ValueError: If df has duplicate columns, or its columns differ from
                the declaration in membership or in order.
        """
        _require_unique_columns(df)

        if list(df.columns) == list(self.columns):
            return

        declared, actual = set(self.columns), set(df.columns)
        missing = [c for c in self.columns if c not in actual]
        unexpected = [c for c in df.columns if c not in declared]
        if missing or unexpected:
            raise ValueError(
                f"frame does not match the schema — missing: {missing}, "
                f"unexpected: {unexpected}"
            )
        raise ValueError(
            f"frame has the declared columns in a different order: "
            f"{list(df.columns)} vs {list(self.columns)}"
        )


def _require_unique_columns(df: pd.DataFrame) -> None:
    """Reject a frame with repeated column names before projecting from it.

    pandas expands rather than projects: selecting ["lag1", "other"] from a frame
    carrying two lag1 columns returns three columns. Checking the input names the
    cause; checking the output would only report a puzzling column list.
    """
    if not df.columns.is_unique:
        duplicated = df.columns[df.columns.duplicated()].unique().tolist()
        raise ValueError(f"frame has duplicate columns: {duplicated}")
