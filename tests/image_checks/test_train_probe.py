import importlib
import io
import re
import sys

import pytest

from fcstnyctaxi.image_checks import train
from fcstnyctaxi.lib.utils import get_project_root_dir

_CONFIG_DIR = get_project_root_dir() / "config"


def _listed_tree() -> list[str]:
    """The repo's config files relative to config/, as the build step lists them."""
    files = sorted(
        str(p.relative_to(_CONFIG_DIR)) for p in _CONFIG_DIR.rglob("*") if p.is_file()
    )
    # Self-check: an empty tree would make every comparison vacuous.
    assert files
    return files


def _run_main(monkeypatch: pytest.MonkeyPatch, listed: list[str]) -> None:
    """Run the probe with the repo's config/ as the baked tree and `listed` on stdin."""
    monkeypatch.setattr(train, "CONFIG_DIR", _CONFIG_DIR)
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(listed) + "\n"))
    train.main()


def test_train_probe_imports_resolve():
    """A renamed impl fails here rather than minutes into a Cloud Build."""
    importlib.import_module("fcstnyctaxi.image_checks.train")


def test_train_probe_passes_a_baked_tree_equal_to_the_listed_one(monkeypatch):
    """A baked config tree matching the build's config/ passes."""
    _run_main(monkeypatch, _listed_tree())


def test_train_probe_rejects_a_listed_file_the_image_lacks(monkeypatch):
    """A listed file left unbaked, as a .dockerignore drop leaves it, fails."""
    listed = [*_listed_tree(), "train/models/extra_file_a.yaml"]

    with pytest.raises(
        SystemExit, match=re.escape("missing ['train/models/extra_file_a.yaml']")
    ):
        _run_main(monkeypatch, listed)


def test_train_probe_rejects_a_baked_file_the_build_did_not_list(monkeypatch):
    """A baked file absent from the build's config/ fails."""
    dropped, *listed = _listed_tree()

    with pytest.raises(SystemExit, match=re.escape(f"extra [{dropped!r}]")):
        _run_main(monkeypatch, listed)
