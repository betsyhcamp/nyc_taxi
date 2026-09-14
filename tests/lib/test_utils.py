import re
from datetime import datetime
from pathlib import Path

import pytest

from fcstnyctaxi.lib.utils import (
    find_root_project_dir,
    generate_run_id,
    get_project_root_dir,
    require_path_safe_run_id,
)

# ================================================
# generate_run_id tests
# ================================================


def test_generate_run_id_is_chronological_and_increasing():
    "Test that run_ids are monotonic, chronologically ordered and sortable."
    ids = [generate_run_id() for _ in range(20)]
    assert ids == sorted(ids)


def test_generate_run_id_makes_correct_format():
    """Test that generate_run_id makes desired microsecond format"""
    run_id = generate_run_id()

    assert re.fullmatch(r"\d{8}T\d{12}Z", run_id)

    # strptime raises ValueError if run_id doesn't match format;
    # then pytest catches ValueError & test will fail
    datetime.strptime(run_id, "%Y%m%dT%H%M%S%fZ")


# ================================================
# find_root_project_dir — happy path
# ================================================


def test_find_root_project_dir_finds_root_from_subdirectory() -> None:
    """Starting from this test file, the ancestor containing pyproject.toml is found."""
    result = find_root_project_dir(start_path=Path(__file__).resolve())
    assert (result / "pyproject.toml").exists()


def test_find_root_project_dir_accepts_custom_markers_list() -> None:
    """A custom markers_list overrides the defaults and still resolves the root."""
    result = find_root_project_dir(
        markers_list=["pyproject.toml"],
        start_path=Path(__file__).resolve(),
    )
    assert (result / "pyproject.toml").exists()


# ================================================
# find_root_project_dir — error path
# ================================================


def test_find_root_project_dir_raises_when_no_markers_found(tmp_path: Path) -> None:
    """RuntimeError raised when no marker files exist anywhere in the directory tree."""

    with pytest.raises(RuntimeError, match="not locate project"):
        find_root_project_dir(
            markers_list=["nonexistent_marker_file"],
            start_path=tmp_path,
        )


# ================================================
# get_project_root_dir — happy path
# ================================================


def test_get_project_root_dir_uses_env_var_when_set(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """PROJECT_ROOT env var is returned directly as a resolved Path."""
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    result = get_project_root_dir()
    assert result == tmp_path.resolve()


def test_get_project_root_dir_falls_back_when_env_not_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without PROJECT_ROOT, find_root_project_dir is used and returns a valid root."""
    monkeypatch.delenv("PROJECT_ROOT", raising=False)
    result = get_project_root_dir(start_path=Path(__file__).resolve())
    assert (result / "pyproject.toml").exists()


# ================================================
# require_path_safe_run_id tests
# ================================================


@pytest.mark.parametrize("run_id", ["f1\n", "..", ".hidden", "-x", ""])
def test_require_path_safe_run_id_rejects_unsafe_ids(run_id: str) -> None:
    """Test that an id which would build a malformed path or line raises.

    Each case is a distinct hazard rather than a variation. ".." escapes the
    scratch container the runner recursively deletes; ".hidden" and "-x" give a
    segment that hides from a listing or reads as a flag; "" leaves a doubled
    slash in the prefix; and the trailing newline is what $(cat run_id.txt)
    supplies, which is the case re.match with "$" accepts.
    """
    with pytest.raises(ValueError, match="--run-id"):
        require_path_safe_run_id(run_id, "--run-id")


def test_require_path_safe_run_id_accepts_a_generated_id() -> None:
    """Test that the shared minter's output satisfies its inverse.

    generate_run_id supplies the id whenever --run-id is absent, so a guard that
    rejected it would fail every run that did not name one.
    """
    require_path_safe_run_id(generate_run_id(), "--run-id")
