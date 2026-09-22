import re
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

from fcstnyctaxi.lib.utils import (
    find_root_project_dir,
    generate_run_id,
    get_project_root_dir,
    require_git_hash,
    require_label_safe_run_id,
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

    assert re.fullmatch(r"\d{8}t\d{12}z", run_id)

    # strptime raises ValueError if run_id doesn't match format;
    # then pytest catches ValueError & test will fail
    datetime.strptime(run_id, "%Y%m%dt%H%M%S%fz")


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

PATH_UNSAFE_RUN_IDS = ["f1\n", "..", ".hidden", "-x", ""]


@pytest.mark.parametrize("run_id", PATH_UNSAFE_RUN_IDS)
def test_require_path_safe_run_id_rejects_unsafe_ids(run_id: str) -> None:
    """Each case is a distinct hazard, not a variation on one."""
    with pytest.raises(ValueError, match="--run-id"):
        require_path_safe_run_id(run_id, "--run-id")


def test_require_path_safe_run_id_accepts_a_generated_id() -> None:
    """A guard rejecting generate_run_id's output would fail every unnamed run."""
    require_path_safe_run_id(generate_run_id(), "--run-id")


# ================================================
# require_label_safe_run_id tests
# ================================================


@pytest.mark.parametrize("run_id", PATH_UNSAFE_RUN_IDS)
def test_require_label_safe_run_id_rejects_every_path_unsafe_id(run_id: str) -> None:
    """It replaces the path guard on --run-id, so it must refuse what that one does."""
    with pytest.raises(ValueError, match="--run-id"):
        require_label_safe_run_id(run_id, "--run-id")


@pytest.mark.parametrize("run_id", ["t-20260913T000000000000Z", "t.1", "a" * 65])
def test_require_label_safe_run_id_rejects_a_path_safe_illegal_label(
    run_id: str,
) -> None:
    """Uppercase, a dot, a 65th character: each is illegal only as a label."""
    # Self-check: the case would otherwise prove only what the path guard does.
    require_path_safe_run_id(run_id, "--run-id")

    with pytest.raises(ValueError, match="--run-id"):
        require_label_safe_run_id(run_id, "--run-id")


@pytest.mark.parametrize("run_id", ["a" * 64, "backfill_2026-09"])
def test_require_label_safe_run_id_accepts_a_legal_label(run_id: str) -> None:
    """The cap's own boundary, and the underscore the label charset allows."""
    require_label_safe_run_id(run_id, "--run-id")


def test_require_label_safe_run_id_accepts_a_generated_id() -> None:
    """A guard rejecting generate_run_id's output would fail every unnamed run."""
    require_label_safe_run_id(generate_run_id(), "--run-id")


# ================================================
# require_git_hash tests
# ================================================


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A real repo with one commit: git's own exit-code contract is what is tested."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "tracked.txt").write_text("original\n")
    for command in (
        ["git", "init", "-q"],
        ["git", "add", "tracked.txt"],
        # Identity per-command, so the suite needs no global git config.
        ["git", "-c", "user.email=t@e.co", "-c", "user.name=T", "commit", "-qm", "i"],
    ):
        subprocess.run(command, cwd=repo, check=True)
    return repo


def test_require_git_hash_marks_a_modified_tree_but_not_a_clean_one(
    git_repo: Path,
) -> None:
    """Test both halves: either alone is satisfiable by a constant."""
    assert re.fullmatch(r"[0-9a-f]{40}", require_git_hash(git_repo))

    (git_repo / "tracked.txt").write_text("modified\n")

    assert re.fullmatch(r"[0-9a-f]{40}-dirty", require_git_hash(git_repo))


def test_require_git_hash_ignores_an_untracked_file(git_repo: Path) -> None:
    """Why the check is `git diff`: untracked files must not read as modified."""
    (git_repo / "untracked.txt").write_text("scratch\n")

    assert not require_git_hash(git_repo).endswith("-dirty")


def test_require_git_hash_raises_rather_than_returning_null(tmp_path: Path) -> None:
    """A run whose commit is unknown cannot be reproduced from its own record."""
    with pytest.raises(RuntimeError, match="cannot record the commit"):
        require_git_hash(tmp_path)
