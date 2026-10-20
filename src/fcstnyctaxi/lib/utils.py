import os
import subprocess
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
