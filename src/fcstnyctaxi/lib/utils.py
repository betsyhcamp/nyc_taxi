import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def find_root_project_dir(
    markers_list: list[str] | None = None, start_path: Path | None = None
) -> Path:
    """Finds the path of the root project directory of a python project where files in
        markers_list are located with start of the search at start_path.

    Args:
        markers_list (list[str] | None, optional): filename strings of files that are in
            the root project directory. Defaults to None.
        start_path (Path | None, optional): Intial starting location to start
            path search. Defaults to None.

    Raises:
        RuntimeError: Could not find marker files so could not find project directory.

    Returns:
        Path: Path of project root directory.
    """
    ROOT_MARKERS_LIST = ["pyproject.toml", "uv.lock", ".project-root"]
    root_markers_list = (
        ROOT_MARKERS_LIST if markers_list is None else list(markers_list)
    )

    path_start = Path(start_path or globals().get("__file__", Path.cwd())).resolve()

    for path_candidate in [path_start, *path_start.parents]:
        check = [
            (path_candidate / root_marker).exists() for root_marker in root_markers_list
        ]
        if any(check) and path_candidate.is_dir():
            return path_candidate
    raise RuntimeError(
        "Could not locate project root directory using marker files:"
        f"{root_markers_list}"
    )


def get_project_root_dir(start_path: Path | None = None) -> Path:
    """Geth the Path of the project root either by loading an environment
        variable PROJECT_ROOT or by walking directories to find marker file
        location.

    Args:
        start_path (Path | None, optional): Intial starting location to start
            path search. Defaults to None.

    Returns:
        Path: Path of project root directory.
    """
    try:
        env_root = os.environ["PROJECT_ROOT"]
    except KeyError as e:
        env_root = None
        print(f"Environment variable {e} not set. Finding project root.")
    if env_root:
        return Path(env_root).resolve()
    return find_root_project_dir(start_path=start_path)


def generate_run_id() -> str:
    """Return a UTC microsecond-precision timestamp suitable as a per-run identifier.

    Format: YYYYMMDDtHHMMSSffffffz (e.g. 20260502t143022123456z).
    Lexicographic sort matches chronological order; microsecond precision
    avoiding collisions. Lowercase, so it passes `require_label_safe_run_id`.

    Interim implementation pending the architecture lineage/run_id decision.
    """
    return datetime.now(UTC).strftime("%Y%m%dt%H%M%S%fz")


def require_path_safe_run_id(run_id: str, flag_name: str) -> None:
    """Reject a run id that would build a malformed path or an unpasteable line.

    Every slice's run id becomes a GCS path segment and a local directory name.
    Call it as soon as the id is resolved, before anything builds or prints it.

    Args:
        run_id (str): The identifier to check.
        flag_name (str): The CLI flag it arrived on, named in the error so a caller
            holding two ids says which one to fix.

    Raises:
        ValueError: run_id does not match `[A-Za-z0-9][A-Za-z0-9._-]*`.
    """
    # The first character is anchored separately because [A-Za-z0-9._-]+ accepts
    # "..", ".hidden" and "-x"; fullmatch rather than match, whose "$" also
    # matches before the trailing newline that $(cat run_id.txt) supplies.
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise ValueError(
            f"{flag_name} {run_id!r} must start with a letter or digit and "
            "contain only letters, digits, '.', '_' or '-'."
        )


def require_label_safe_run_id(run_id: str, flag_name: str) -> None:
    """Reject a Training run id that is not a legal GCP label value for its model.

    Stricter than `require_path_safe_run_id`, which it replaces. Not for a Feature
    run id, whose format is not this repo's.

    Args:
        run_id (str): The identifier to check.
        flag_name (str): The CLI flag it arrived on, named in the error.

    Raises:
        ValueError: run_id does not match `[a-z0-9][a-z0-9_-]*` or exceeds 64
            characters.
    """
    # The SDK does not enforce this; the service does, at the pipeline's last task.
    # Lowercasing there instead would collide ids differing only in case.
    if len(run_id) > 64 or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", run_id):
        raise ValueError(
            f"{flag_name} {run_id!r} must match [a-z0-9][a-z0-9_-]*, at most 64 "
            "characters."
        )


def require_git_hash(repo_dir: Path) -> str:
    """The commit this run reproduces from, refused rather than stamped as null.

    Both commands run at repo_dir, since the process CWD can be a sibling repo.

    Args:
        repo_dir (Path): Repository the hash is required from.

    Returns:
        str: The HEAD sha, suffixed "-dirty" when tracked files are modified.

    Raises:
        RuntimeError: git could not answer for repo_dir, so a run whose commit is
            unknown cannot be reproduced from its own record.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            stdout=subprocess.PIPE,
            text=True,
            check=True,
        ).stdout.strip()
    # git's own stderr reaches the terminal, so this adds only what git cannot
    # know: which directory the run required a hash from.
    except (OSError, subprocess.CalledProcessError) as err:
        raise RuntimeError(
            f"git rev-parse HEAD failed in {repo_dir}, so this run cannot record "
            "the commit that produced it."
        ) from err

    # git status --porcelain would read dirty on every run due to untracked files
    completed = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--"], cwd=repo_dir, check=False
    )

    if completed.returncode not in (0, 1):
        raise RuntimeError(
            f"git diff --quiet HEAD -- exited {completed.returncode}, so a clean "
            "tree cannot be told from a modified one."
        )
    return f"{sha}-dirty" if completed.returncode else sha
