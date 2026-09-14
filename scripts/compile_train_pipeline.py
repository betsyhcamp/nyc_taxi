import os
from typing import Any

import yaml
from kfp import compiler

from fcstnyctaxi.lib.container_images import artifact_registry_prefix
from fcstnyctaxi.lib.utils import get_project_root_dir
from fcstnyctaxi.pipelines.train_pipeline import train_pipeline

_TEMPLATE = "fcst-train-pipeline.yaml"


def _executor_images(spec: dict[str, Any]) -> set[str]:
    """Every Docker image the compiled pipeline will actually run.Importer steps run no
    image and are skipped.
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


def main() -> None:
    """Compile to build/ and refuse to leave a spec that pins the wrong image."""
    # The import above already refused an unset or malformed value.
    expected = os.environ["FCST_TRAIN_IMAGE"]
    template = get_project_root_dir() / "build" / _TEMPLATE
    template.parent.mkdir(parents=True, exist_ok=True)

    compiler.Compiler().compile(
        pipeline_func=train_pipeline,  # type: ignore[arg-type]
        package_path=str(template),
    )
    try:
        _require_pinned_project_images(yaml.safe_load(template.read_text()), expected)
    except Exception:
        # compile() overwrote any previous one; a rejected spec here is submittable.
        template.unlink(missing_ok=True)
        raise

    print(f"\n  compiled : {template}\n  pinned   : {expected}\n")


if __name__ == "__main__":
    main()
