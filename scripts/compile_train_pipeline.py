import os
from pathlib import Path
from typing import Any

import yaml
from kfp import compiler

from fcstnyctaxi.lib.config.bindings import model_names_from_roles, resolve_model_roles
from fcstnyctaxi.lib.container_images import artifact_registry_prefix
from fcstnyctaxi.lib.utils import get_project_root_dir
from fcstnyctaxi.pipelines.train_pipeline import build_train_pipeline

_TEMPLATE = "fcst-train-pipeline.yaml"


def _executor_images(spec: dict[str, Any]) -> set[str]:
    """Every Docker image the compiled pipeline will actually run. Importer steps run
    no image and are skipped.
    """
    executors = spec["deploymentSpec"]["executors"]
    return {ex["container"]["image"] for ex in executors.values() if "container" in ex}


def _require_pinned_project_images(spec: dict[str, Any], expected: str) -> None:
    """Raise unless the spec's project-owned images are exactly `expected`. Classified
    by prefix, not an allowlist."""
    prefix = artifact_registry_prefix(expected)
    project_images = {i for i in _executor_images(spec) if i.startswith(f"{prefix}/")}
    if project_images != {expected}:
        raise RuntimeError(
            f"Project-owned images under {prefix}/ must be exactly [{expected!r}], "
            f"found {sorted(project_images) or 'none'}."
        )


def _compile(project_root: Path, template_path: Path) -> None:
    """Compile the DAG for one tree's model set, refusing a wrongly pinned spec.

    Two roots, not one: the model set is read from `project_root/config` while the
    template goes wherever the caller asks, and the test that redirects the output
    writes to a temp directory with no config tree in it.

    Args:
        project_root (Path): Root whose `config/` tree names the model set.
        template_path (Path): Where the compiled template is written.

    Raises:
        RuntimeError: If the spec's project-owned images are not exactly
            FCST_TRAIN_IMAGE.
        ValueError: If the model set is empty or repeats a name.
    """
    # The import above already refused an unset or malformed value.
    expected = os.environ["FCST_TRAIN_IMAGE"]
    template_path.parent.mkdir(parents=True, exist_ok=True)

    # One composition, two derived values, so the template's model set and its
    # role map cannot disagree with each other.
    model_roles = resolve_model_roles(project_root / "config")
    compiler.Compiler().compile(
        pipeline_func=build_train_pipeline(
            model_names=model_names_from_roles(model_roles), model_roles=model_roles
        ),
        package_path=str(template_path),
    )
    try:
        spec = yaml.safe_load(template_path.read_text())
        _require_pinned_project_images(spec, expected)
    except Exception:
        # compile() overwrote any previous one; a rejected spec here is submittable.
        template_path.unlink(missing_ok=True)
        raise

    print(f"\n  compiled : {template_path}\n  pinned   : {expected}\n")


def main() -> None:
    """Compile to build/ for the working tree's own model set."""
    project_root = get_project_root_dir()
    _compile(project_root, project_root / "build" / _TEMPLATE)


if __name__ == "__main__":
    main()
