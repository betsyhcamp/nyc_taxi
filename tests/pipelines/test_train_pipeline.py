import os

# Must precede the imports below: @dsl.component binds base_image at import time,
# with no fallback, so an unset value aborts collection for the whole suite.
os.environ.setdefault(
    "FCST_TRAIN_IMAGE",
    "us-central1-docker.pkg.dev/nyc-taxi-ehc/fcst-ml-containers/train@sha256:"
    + "a" * 64,
)

from pathlib import Path
from typing import Any

import yaml
from kfp import compiler

from fcstnyctaxi.pipelines.train_pipeline import train_pipeline


def _compiled_ir(tmp_path: Path) -> dict[str, Any]:
    """Compile the pipeline to tmp_path and return the parsed IR."""
    out = tmp_path / "pipeline.yaml"
    compiler.Compiler().compile(
        pipeline_func=train_pipeline,  # type: ignore[arg-type]
        package_path=str(out),
    )
    return yaml.safe_load(out.read_text())


def test_train_pipeline_compiles_with_the_compose_configs_task(tmp_path: Path) -> None:
    """Test that the DAG compiles without GCP and holds its one task.

    Compilation is the assertion: a mis-typed artifact or an unwired parameter
    raises here rather than at submission against a billed cluster.
    """
    tasks = _compiled_ir(tmp_path)["root"]["dag"]["tasks"]

    assert "compose-configs" in tasks


def test_each_importer_takes_its_uri_as_a_runtime_parameter(tmp_path: Path) -> None:
    """Test that both artifact URIs stay runtime parameters, not compile-time values.

    A constant here would bake one Feature run into the compiled spec, so every
    submission would consume that panel whatever --panel-uri said.
    """
    executors = _compiled_ir(tmp_path)["deploymentSpec"]["executors"]
    importers = [spec["importer"] for spec in executors.values() if "importer" in spec]

    # Non-vacuity: an empty list would satisfy the loop below without testing anything.
    assert importers
    for importer in importers:
        assert "runtimeParameter" in importer["artifactUri"]
