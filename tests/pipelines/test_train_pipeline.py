import os

# Precedes the imports: @dsl.component binds base_image at import time, and an unset
# value aborts collection for the whole suite.
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
    """Test that compiling raises nothing without GCP and the one task is present."""
    tasks = _compiled_ir(tmp_path)["root"]["dag"]["tasks"]

    assert "compose-configs" in tasks


def _executor_of(ir: dict[str, Any], task_name: str) -> dict[str, Any]:
    """The deployment spec that runs one task; its name alone cannot tell an importer
    from an ordinary component."""
    component_name = ir["root"]["dag"]["tasks"][task_name]["componentRef"]["name"]
    executor_label = ir["components"][component_name]["executorLabel"]
    return ir["deploymentSpec"]["executors"][executor_label]


def test_each_artifact_input_comes_from_its_own_importer(tmp_path: Path) -> None:
    """Test that panel and calendar each arrive from an importer of their own URI.

    Both miswirings are silent until runtime: a shared importer reaches the impl's
    same-filepath guard, and a transposed pair reaches nothing until pandas.
    """
    ir = _compiled_ir(tmp_path)
    tasks = ir["root"]["dag"]["tasks"]

    for artifact_name, expected_parameter in (
        ("panel", "panel_uri"),
        ("calendar", "calendar_uri"),
    ):
        producer = tasks["compose-configs"]["inputs"]["artifacts"][artifact_name][
            "taskOutputArtifact"
        ]["producerTask"]

        assert "importer" in _executor_of(ir, producer)
        uri = tasks[producer]["inputs"]["parameters"]["uri"]
        assert uri["componentInputParameter"] == expected_parameter


def test_each_importer_takes_its_uri_as_a_runtime_parameter(tmp_path: Path) -> None:
    """Test that both artifact URIs stay runtime parameters, not compile-time values.

    A constant would bake one Feature run into the spec, so every submission would
    consume that panel whatever --panel-uri said.
    """
    executors = _compiled_ir(tmp_path)["deploymentSpec"]["executors"]
    importers = [spec["importer"] for spec in executors.values() if "importer" in spec]

    # Non-vacuity: an empty list would satisfy the loop below.
    assert importers
    for importer in importers:
        assert "runtimeParameter" in importer["artifactUri"]
