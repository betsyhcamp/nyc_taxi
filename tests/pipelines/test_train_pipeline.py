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

import pytest
import yaml
from kfp import compiler

from fcstnyctaxi.lib.config.bindings import resolve_model_names
from fcstnyctaxi.lib.utils import get_project_root_dir
from fcstnyctaxi.pipelines.train_pipeline import build_train_pipeline

CONFIG_DIR = get_project_root_dir() / "config"

# Three, none of them in config/: a hardcoded loop, a wrong-key read and a dropped
# entry all fail against a set the shipped tree cannot supply.
SYNTHETIC_MODEL_NAMES = ("model_a", "model_b", "model_c")


def _compiled_ir(tmp_path: Path, model_names: tuple[str, ...]) -> dict[str, Any]:
    """Compile the DAG for one model set to tmp_path and return the parsed IR."""
    out = tmp_path / "pipeline.yaml"
    compiler.Compiler().compile(
        pipeline_func=build_train_pipeline(model_names=model_names),
        package_path=str(out),
    )
    return yaml.safe_load(out.read_text())


def _backtest_tasks(ir: dict[str, Any]) -> dict[str, Any]:
    """Tasks carrying a model_name, selected by input rather than by name.

    set_display_name changes taskInfo.name only, so IR keys stay positional and
    keying off them would assert KFP's numbering instead of this DAG's fan-out.
    """
    return {
        name: task
        for name, task in ir["root"]["dag"]["tasks"].items()
        if "model_name" in task.get("inputs", {}).get("parameters", {})
    }


def _model_names_of(tasks: dict[str, Any]) -> list[str]:
    """The model_name constant each task was compiled with."""
    return [
        task["inputs"]["parameters"]["model_name"]["runtimeValue"]["constant"]
        for task in tasks.values()
    ]


def _executor_of(ir: dict[str, Any], task_name: str) -> dict[str, Any]:
    """The deployment spec that runs one task; its name alone cannot tell an importer
    from an ordinary component."""
    component_name = ir["root"]["dag"]["tasks"][task_name]["componentRef"]["name"]
    executor_label = ir["components"][component_name]["executorLabel"]
    return ir["deploymentSpec"]["executors"][executor_label]


def test_train_pipeline_compiles_with_the_compose_configs_task(tmp_path: Path) -> None:
    """Test that compiling raises nothing without GCP and the first task is present."""
    tasks = _compiled_ir(tmp_path, SYNTHETIC_MODEL_NAMES)["root"]["dag"]["tasks"]

    assert "compose-configs" in tasks


def test_the_dag_emits_one_backtest_task_per_model_it_was_built_for(
    tmp_path: Path,
) -> None:
    """Test that the fan-out follows its argument, not the tree or a hardcoded loop."""
    ir = _compiled_ir(tmp_path, SYNTHETIC_MODEL_NAMES)

    assert _model_names_of(_backtest_tasks(ir)) == list(SYNTHETIC_MODEL_NAMES)


def test_each_backtest_task_runs_its_own_executor(tmp_path: Path) -> None:
    """Test that the tasks are distinct executors rather than one task run thrice."""
    ir = _compiled_ir(tmp_path, SYNTHETIC_MODEL_NAMES)
    backtest_tasks = _backtest_tasks(ir)

    labels = {
        ir["components"][task["componentRef"]["name"]]["executorLabel"]
        for task in backtest_tasks.values()
    }
    assert len(labels) == len(backtest_tasks)
    # One image across all of them, which is why _require_pinned_project_images
    # still sees a singleton project-owned set.
    images = {
        ir["deploymentSpec"]["executors"][label]["container"]["image"]
        for label in labels
    }
    assert images == {os.environ["FCST_TRAIN_IMAGE"]}


def test_each_backtest_task_is_display_named_for_its_model(tmp_path: Path) -> None:
    """Test that the operator sees model-named tasks rather than positional keys.

    IR keys stay positional whatever set_display_name does, so dropping the call
    leaves a DAG that runs correctly and is unreadable in the Vertex UI.
    """
    backtest_tasks = _backtest_tasks(_compiled_ir(tmp_path, SYNTHETIC_MODEL_NAMES))

    assert backtest_tasks  # non-vacuity: an empty mapping satisfies the loop
    for task in backtest_tasks.values():
        model_name = task["inputs"]["parameters"]["model_name"]["runtimeValue"][
            "constant"
        ]
        assert task["taskInfo"]["name"] == f"backtest-{model_name}"


def test_the_shipped_tree_is_a_legal_model_set(tmp_path: Path) -> None:
    """Test the real config composes into a DAG, without naming any model."""
    model_names = resolve_model_names(CONFIG_DIR)
    ir = _compiled_ir(tmp_path, model_names)

    assert _model_names_of(_backtest_tasks(ir)) == list(model_names)


@pytest.mark.parametrize(
    ("model_names", "match"),
    [((), "empty"), (("model_a", "model_a"), "repeats a name")],
    ids=["empty", "duplicated"],
)
def test_a_model_set_the_dag_cannot_fan_out_over_is_refused(
    model_names: tuple[str, ...], match: str
) -> None:
    """Test that an empty or repeating model set is refused rather than compiled.

    Neither reaches the builder from `resolve_model_names`, and neither is loud in
    KFP: duplicates compile to two tasks racing on one sidecar directory.
    """
    with pytest.raises(ValueError, match=match):
        build_train_pipeline(model_names=model_names)


def test_each_artifact_input_comes_from_its_own_importer(tmp_path: Path) -> None:
    """Test that panel and calendar each arrive from an importer of their own URI.

    Both miswirings are silent until runtime: a shared importer reaches the impl's
    same-filepath guard, and a transposed pair reaches nothing until pandas.
    """
    ir = _compiled_ir(tmp_path, SYNTHETIC_MODEL_NAMES)
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


def test_every_backtest_task_reuses_the_compose_importer_handles(
    tmp_path: Path,
) -> None:
    """Test that one importer per artifact feeds compose and every backtest task.

    A second importer would let a backtest task read different bytes than compose
    validated, which no check inside a task can see: a task cannot know its siblings.
    """
    ir = _compiled_ir(tmp_path, SYNTHETIC_MODEL_NAMES)
    tasks = ir["root"]["dag"]["tasks"]
    backtest_names = set(_backtest_tasks(ir))

    for artifact_name in ("panel", "calendar"):
        producer_of = {
            name: task["inputs"]["artifacts"][artifact_name]["taskOutputArtifact"][
                "producerTask"
            ]
            for name, task in tasks.items()
            if artifact_name in task.get("inputs", {}).get("artifacts", {})
        }
        # Non-vacuity: compose and every backtest task must be among the readers.
        assert backtest_names | {"compose-configs"} <= set(producer_of)
        assert len(set(producer_of.values())) == 1


def test_every_backtest_task_takes_its_run_prefix_from_compose(tmp_path: Path) -> None:
    """Test that run_prefix arrives from the compose task rather than being re-derived.

    Re-deriving it would need the environment destination composed a second time,
    which is the exception the compose wrapper was granted and this one was not.
    """
    ir = _compiled_ir(tmp_path, SYNTHETIC_MODEL_NAMES)

    for task in _backtest_tasks(ir).values():
        source = task["inputs"]["parameters"]["run_prefix"]["taskOutputParameter"]
        assert source["producerTask"] == "compose-configs"
        assert source["outputParameterKey"] == "run_prefix"


def test_each_importer_takes_its_uri_as_a_runtime_parameter(tmp_path: Path) -> None:
    """Test that both artifact URIs stay runtime parameters, not compile-time values.

    A constant would bake one Feature run into the spec, so every submission would
    consume that panel whatever --panel-uri said.
    """
    executors = _compiled_ir(tmp_path, SYNTHETIC_MODEL_NAMES)["deploymentSpec"][
        "executors"
    ]
    importers = [spec["importer"] for spec in executors.values() if "importer" in spec]

    # Non-vacuity: an empty list would satisfy the loop below.
    assert importers
    for importer in importers:
        assert "runtimeParameter" in importer["artifactUri"]


def test_the_model_set_costs_no_pipeline_parameter(tmp_path: Path) -> None:
    """Test that fanning out adds no root parameter, so the submit CLI is unchanged.

    model_name is a compile-time constant; a hoisted parameter would reach the
    submitter and every scheduled caller's parameterValues.
    """
    ir = _compiled_ir(tmp_path, SYNTHETIC_MODEL_NAMES)
    roots = set(ir["root"]["inputDefinitions"]["parameters"])

    assert "model_name" not in roots
    assert roots == {
        "env",
        "train_run_id",
        "feature_run_id",
        "panel_uri",
        "calendar_uri",
    }
