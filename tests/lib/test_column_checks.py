import pandas as pd
import pytest

from fcstnyctaxi.lib.column_checks import (
    require_columns,
    require_matching_feature_run_id,
    require_single_feature_run_id,
    trim_to_allowlist,
)
from fcstnyctaxi.schemas.run_identity import LINEAGE_COLUMN

FEATURE_RUN_ID = "f-2026-09-08"


def _frame(feature_run_id: str | None = FEATURE_RUN_ID, rows: int = 3) -> pd.DataFrame:
    """A frame carrying only the lineage column, "string" dtype like the real one."""
    return pd.DataFrame(
        {LINEAGE_COLUMN: pd.array([feature_run_id] * rows, dtype="string")}
    )


def test_require_columns_passes_when_all_present() -> None:
    """Returns None rather than raising when every required column is there."""
    df = pd.DataFrame({"a": [1], "b": [2]})
    assert require_columns(df, ["a", "b"], "df") is None


def test_require_columns_ignores_extra_columns() -> None:
    """Requires a subset, not an exact column set, so extras pass."""
    df = pd.DataFrame({"a": [1], "b": [2], "extra": [3]})
    assert require_columns(df, ["a"], "df") is None


def test_require_columns_passes_vacuously_on_empty_required() -> None:
    """An empty requirement list is a silent pass, and that is intentional.

    Pinned rather than left implied: a caller that builds `required` dynamically
    and produces [] gets no check at all, so the vacuous pass should be a stated
    contract instead of something a reader has to infer from the comprehension.
    """
    df = pd.DataFrame({"a": [1]})
    assert require_columns(df, [], "df") is None


def test_require_columns_raises_naming_frame_and_missing_columns() -> None:
    """The message names which frame is at fault, which a bare KeyError cannot."""
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match=r"panel_df is missing required columns"):
        require_columns(df, ["a", "b"], "panel_df")


def test_require_columns_reports_every_missing_column_not_just_the_first() -> None:
    """One call surfaces all missing columns instead of one fix-and-retry each."""
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError) as excinfo:
        require_columns(df, ["a", "b", "c"], "df")
    assert "'b'" in str(excinfo.value)
    assert "'c'" in str(excinfo.value)


# ================================================
# require_single_feature_run_id
# ================================================


def test_single_feature_run_id_returns_the_one_id_the_frame_carries() -> None:
    """The happy path hands back what was validated, rather than returning None."""
    assert require_single_feature_run_id(_frame(), "panel") == FEATURE_RUN_ID


def test_single_feature_run_id_names_the_frame_when_the_column_is_absent() -> None:
    """Delegates to require_columns, so an unstamped artifact names itself."""
    df = pd.DataFrame({"y": [1, 2, 3]})
    with pytest.raises(ValueError, match=r"panel is missing required columns"):
        require_single_feature_run_id(df, "panel")


@pytest.mark.parametrize("null_rows", [slice(None), slice(0, 1)], ids=["all", "some"])
def test_single_feature_run_id_rejects_nulls(null_rows: slice) -> None:
    """nunique() skips nulls, so a partly-null column would otherwise pass."""
    df = _frame()
    df.loc[df.index[null_rows], LINEAGE_COLUMN] = None

    with pytest.raises(ValueError, match=r"null feature_run_id"):
        require_single_feature_run_id(df, "panel")


def test_single_feature_run_id_rejects_a_frame_mixing_two_runs() -> None:
    """Two ids in one frame means every table stamped from it claims a false run."""
    df = _frame()
    df.loc[df.index[0], LINEAGE_COLUMN] = "another-run"

    with pytest.raises(ValueError, match=r"panel mixes Feature runs"):
        require_single_feature_run_id(df, "panel")


# ================================================
# require_matching_feature_run_id
# ================================================


def test_matching_feature_run_id_returns_the_id_both_frames_carry() -> None:
    """The observed value is what a caller stamps, so it must come back."""
    observed = require_matching_feature_run_id(_frame(), _frame(), FEATURE_RUN_ID)
    assert observed == FEATURE_RUN_ID


def test_matching_feature_run_id_rejects_frames_from_different_runs() -> None:
    """Origins from one run's calendar and actuals from another's is wrong numbers."""
    with pytest.raises(ValueError, match=r"!= calendar"):
        require_matching_feature_run_id(
            _frame(), _frame(feature_run_id="another-run"), FEATURE_RUN_ID
        )


def test_matching_feature_run_id_rejects_a_consistent_pair_from_the_wrong_run() -> None:
    """The check that makes pasting explicit URIs safe: right shape, wrong run."""
    with pytest.raises(ValueError, match=r"not the declared"):
        require_matching_feature_run_id(_frame(), _frame(), "a-different-run")


def test_matching_feature_run_id_names_which_of_the_two_frames_is_unstamped() -> None:
    """Two frames are read here, so a bare KeyError would not say which one failed."""
    calendar = pd.DataFrame(
        {"ds": pd.date_range("2025-01-05", periods=3, freq="W-SUN")}
    )

    with pytest.raises(ValueError, match=r"calendar is missing required columns"):
        require_matching_feature_run_id(_frame(), calendar, FEATURE_RUN_ID)


def test_matching_feature_run_id_checks_frame_agreement_before_the_declaration() -> (
    None
):
    """Both faults at once: a shared value must exist before it can be compared."""
    with pytest.raises(ValueError, match=r"!= calendar"):
        require_matching_feature_run_id(
            _frame(), _frame(feature_run_id="another-run"), "a-third-run"
        )


# ================================================
# trim_to_allowlist
# ================================================

_ALLOWED = ("a", "b", "optional")
_REQUIRED = ("a", "b")


def _wide() -> pd.DataFrame:
    """Everything allowed, plus the metadata columns Feature keeps adding."""
    return pd.DataFrame(
        {
            "a": [1],
            "b": [2],
            "optional": [3],
            LINEAGE_COLUMN: pd.array(["f-1"], dtype="string"),
            "executed_at": [pd.Timestamp("2026-09-16")],
        }
    )


def test_trim_drops_columns_outside_the_allowlist() -> None:
    """The hazard: an unlisted column reaching a model crashes it four layers down."""
    trimmed = trim_to_allowlist(
        _wide(), required=_REQUIRED, allowed=_ALLOWED, frame_name="panel"
    )
    assert list(trimmed.columns) == list(_ALLOWED)


def test_trim_names_the_frame_and_every_missing_required_column() -> None:
    """A bare pandas KeyError would name neither the frame nor the second gap."""
    frame = _wide().drop(columns=["a", "b"])
    with pytest.raises(ValueError, match=r"calendar is missing required columns") as e:
        trim_to_allowlist(
            frame, required=_REQUIRED, allowed=_ALLOWED, frame_name="calendar"
        )
    assert "'a'" in str(e.value) and "'b'" in str(e.value)


def test_trim_tolerates_an_allowed_column_that_is_not_required() -> None:
    """The contract says the unconsumed two can change without a coordinated release."""
    frame = _wide().drop(columns=["optional"])
    trimmed = trim_to_allowlist(
        frame, required=_REQUIRED, allowed=_ALLOWED, frame_name="calendar"
    )
    assert list(trimmed.columns) == list(_REQUIRED)


def test_trim_allows_exactly_what_it_requires_by_default() -> None:
    """The panel's case: every column it carries into a model is consumed."""
    trimmed = trim_to_allowlist(_wide(), required=_REQUIRED, frame_name="panel")
    assert list(trimmed.columns) == list(_REQUIRED)


def test_trim_leaves_the_callers_frame_alone() -> None:
    """Callers rebind in place, so an in-place drop would hide until it mattered."""
    frame = _wide()
    before = list(frame.columns)
    trim_to_allowlist(frame, required=_REQUIRED, allowed=_ALLOWED, frame_name="panel")
    assert list(frame.columns) == before
