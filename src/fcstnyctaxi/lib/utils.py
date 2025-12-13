from pathlib import Path
import os
from typing import Any
import yaml


def find_root_project_dir(
    markers_list: list[str] | None = None, start_path: Path | None = None
) -> Path:
    """Finds the path of the root project directory of a python project where files in
        markers_list are located with start of the search at start_path.

    Args:
        markers_list (list[str] | None, optional): filename strings of files that are in
            the root project directory. Defaults to None.
        start_path (Path | None, optional): Intial starting location to start path search.
            Defaults to None.

    Raises:
        RuntimeError: Could not find marker files so could not find project directory.

    Returns:
        Path: Path of project root directory.
    """
    ROOT_MARKERS_LIST = ["pyproject.toml", "uv.lock", ".project-root"]
    root_markers_list = ROOT_MARKERS_LIST if markers_list is None else list(markers_list)

    path_start = Path(start_path or globals().get("__file__", Path.cwd())).resolve()

    for path_candidate in [path_start, *path_start.parents]:
        check = [(path_candidate / root_marker).exists() for root_marker in root_markers_list]
        if any(check) and path_candidate.is_dir():
            return path_candidate
    raise RuntimeError(
        f"Count not locate project root directory using marker files:{root_markers_list}"
    )


def get_project_root_dir(start_path: Path | None = None) -> Path:
    """Geth the Path of the project root either by loading an environment
        variable PROJECT_ROOT or by walking directories to find marker file
        location.

    Args:
        start_path (Path | None, optional): Intial starting location to start path search.
            Defaults to None.

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


def load_config_file(path: Path) -> dict[str, Any]:
    """Load a .yaml / .yml configuration file and return the contents of the
        configuration file as a dictionary

    Args:
        path (Path): Path to the configuration .yaml / .yml file

    Raises:
        ValueError: If the file extension is not .yaml / .yml, ValueError
            raised.

    Returns:
        dict[str, Any]: Data in the configuration file
    """
    ext = path.suffix.lower()
    if ext in [".yaml", ".yml"]:
        with path.open("r", encoding="utf-8") as file_object:
            data = yaml.safe_load(file_object) or {}
    else:
        raise ValueError(f"Unsupported config type {ext}, use .yaml / .yml")
    return data
