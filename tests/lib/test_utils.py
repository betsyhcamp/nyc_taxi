from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pytest

from fcstnyctaxi.lib.utils import (
    find_root_project_dir,
    generate_run_id,
    get_project_root_dir,
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


# TODO(human): test that RuntimeError is raised when no marker files are found.
# Use the tmp_path fixture (pytest's built-in: just add `tmp_path: Path` to the
# function signature — pytest injects a fresh empty directory automatically).
# Pass tmp_path as start_path and a markers_list with a filename that doesn't
# exist anywhere in the tmp directory tree. Assert pytest.raises(RuntimeError).
# Tip: the error message in utils.py contains "marker files" — use that as
# the match= string to keep the test resilient to minor wording changes.
def test_find_root_project_dir_raises_when_no_markers_found(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Could not locate"):
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
