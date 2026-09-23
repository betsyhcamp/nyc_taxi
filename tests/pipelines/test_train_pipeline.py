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

from fcstnyctaxi.lib.config.bindings import resolve_model_names, resolve_model_roles
from fcstnyctaxi.lib.utils import get_project_root_dir
from fcstnyctaxi.pipelines.train_pipeline import build_train_pipeline
from fcstnyctaxi.schemas.config.train import ModelRoles

CONFIG_DIR = get_project_root_dir() / "config"

# Three, none of them in config/: a hardcoded loop, a wrong-key read and a dropped
# entry all fail against a set the shipped tree cannot supply.
SYNTHETIC_MODEL_NAMES = ("model_a", "model_b", "model_c")

# Chosen so a positional read fails: the challenger is the middle name and model_c
# holds no role, so neither edge can be reached from the tuple's ends.
SYNTHETIC_MODEL_ROLES = ModelRoles(benchmark="model_a", challenger="model_b")


def _compiled_ir(
    tmp_path: Path,
    model_names: tuple[str, ...],
    model_roles: ModelRoles = SYNTHETIC_MODEL_ROLES,
) -> dict[str, Any]:
    """Compile the DAG for one model set to tmp_path and return the parsed IR."""
    out = tmp_path / "pipeline.yaml"
    compiler.Compiler().compile(
        pipeline_func=build_train_pipeline(
            model_names=model_names, model_roles=model_roles
        ),
        package_path=str(out),
    )
    return yaml.safe_load(out.read_text())


def _output_artifacts_of(ir: dict[str, Any], task: dict[str, Any]) -> set[str]:
    """The artifact keys one task's component declares as outputs."""
    component = ir["components"][task["componentRef"]["name"]]
    return set(component.get("outputDefinitions", {}).get("artifacts", {}))


def _backtest_tasks(ir: dict[str, Any]) -> dict[str, Any]:
    """Tasks producing a sidecar. Not by model_name, which final_fit takes too; not by
    IR key, which is KFP's positional numbering rather than this DAG's fan-out."""
    return {
        name: task
        for name, task in ir["root"]["dag"]["tasks"].items()
        if "sidecar" in _output_artifacts_of(ir, task)
    }


def _tasks_taking_parameter(ir: dict[str, Any], name: str) -> dict[str, Any]:
    """Every task whose compiled inputs carry a parameter of that name."""
    return {
        task_name: task
        for task_name, task in ir["root"]["dag"]["tasks"].items()
        if name in task.get("inputs", {}).get("parameters", {})
    }


def _tasks_reading_artifact(ir: dict[str, Any], name: str) -> dict[str, Any]:
    """Every task whose compiled inputs carry an input artifact of that name."""
    return {
        task_name: task
        for task_name, task in ir["root"]["dag"]["tasks"].items()
        if name in task.get("inputs", {}).get("artifacts", {})
    }


def _model_names_of(tasks: dict[str, Any]) -> list[str]:
    """The model_name constant each task was compiled with."""
    return [
        task["inputs"]["parameters"]["model_name"]["runtimeValue"]["constant"]
        for task in tasks.values()
    ]


def _compose_parameters(ir: dict[str, Any]) -> dict[str, Any]:
    """The parameter wiring of the compose task every backtest task depends on."""
    return ir["root"]["dag"]["tasks"]["compose-configs"]["inputs"]["parameters"]


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


def test_the_compose_task_is_told_the_model_set_the_dag_was_built_for(
    tmp_path: Path,
) -> None:
    """Test that the claim compose checks is built from this DAG's own model set."""
    ir = _compiled_ir(tmp_path, SYNTHETIC_MODEL_NAMES)

    # compose_configs refuses a claim disagreeing with the image's baked tree.
    declared = _compose_parameters(ir)["declared_model_names"]
    assert declared["runtimeValue"]["constant"] == list(SYNTHETIC_MODEL_NAMES)


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
    ir = _compiled_ir(tmp_path, model_names, resolve_model_roles(CONFIG_DIR))

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
        build_train_pipeline(model_names=model_names, model_roles=SYNTHETIC_MODEL_ROLES)


def test_each_artifact_input_comes_from_its_own_importer(tmp_path: Path) -> None:
    """Test that each of the three artifacts arrives from an importer of its own URI.

    Both miswirings are silent until runtime: a shared importer reaches the impl's
    three-different-files guard, and a transposed pair reaches nothing until pandas.
    """
    ir = _compiled_ir(tmp_path, SYNTHETIC_MODEL_NAMES)
    tasks = ir["root"]["dag"]["tasks"]

    for artifact_name, expected_parameter in (
        ("panel", "panel_uri"),
        ("calendar", "calendar_uri"),
        ("additional_exog", "additional_exog_uri"),
    ):
        producer = tasks["compose-configs"]["inputs"]["artifacts"][artifact_name][
            "taskOutputArtifact"
        ]["producerTask"]

        assert "importer" in _executor_of(ir, producer)
        uri = tasks[producer]["inputs"]["parameters"]["uri"]
        assert uri["componentInputParameter"] == expected_parameter


def test_the_compose_task_takes_each_selector_from_its_own_parameter(
    tmp_path: Path,
) -> None:
    """Test that env and the two run ids each arrive from the parameter of that name."""
    parameters = _compose_parameters(_compiled_ir(tmp_path, SYNTHETIC_MODEL_NAMES))

    # A transposition compiles and runs, writing to a prefix named for the Feature run.
    for name in ("env", "train_run_id", "feature_run_id"):
        assert parameters[name]["componentInputParameter"] == name


def test_every_panel_and_calendar_reader_shares_one_importer(tmp_path: Path) -> None:
    """Test that one importer per artifact feeds every reader: a second would hand a
    task bytes compose never validated, and a task cannot see its siblings."""
    ir = _compiled_ir(tmp_path, SYNTHETIC_MODEL_NAMES)
    backtest_names = set(_backtest_tasks(ir))

    for artifact_name in ("panel", "calendar"):
        producer_of = {
            name: task["inputs"]["artifacts"][artifact_name]["taskOutputArtifact"][
                "producerTask"
            ]
            for name, task in _tasks_reading_artifact(ir, artifact_name).items()
        }
        # Non-vacuity: every task known to read it, so a renamed input cannot drop one.
        assert backtest_names | {"compose-configs", "final-fit"} <= set(producer_of)
        assert len(set(producer_of.values())) == 1


def test_every_task_taking_a_run_prefix_takes_it_from_compose(tmp_path: Path) -> None:
    """Test that no task re-derives run_prefix: that would compose the environment
    destination a second time, the exception only the compose wrapper was granted."""
    ir = _compiled_ir(tmp_path, SYNTHETIC_MODEL_NAMES)
    takers = _tasks_taking_parameter(ir, "run_prefix")

    # Non-vacuity, naming every non-backtest reader: a renamed input would drop one.
    assert set(_backtest_tasks(ir)) | {"evaluate", "final-fit"} <= set(takers)
    for task in takers.values():
        source = task["inputs"]["parameters"]["run_prefix"]["taskOutputParameter"]
        assert source["producerTask"] == "compose-configs"
        assert source["outputParameterKey"] == "run_prefix"


def test_every_task_reading_composed_configs_reads_what_compose_wrote(
    tmp_path: Path,
) -> None:
    """Test that every task reading composed_configs takes it from the compose task."""
    ir = _compiled_ir(tmp_path, SYNTHETIC_MODEL_NAMES)
    readers = _tasks_reading_artifact(ir, "composed_configs")

    # KFP accepts a Dataset for Input[Artifact], so a misrouting compiles silently.
    # Non-vacuity, naming every non-backtest reader: a renamed input would drop one.
    assert set(_backtest_tasks(ir)) | {
        "evaluate",
        "final-fit",
        "register-model",
    } <= set(readers)
    for task in readers.values():
        source = task["inputs"]["artifacts"]["composed_configs"]["taskOutputArtifact"]
        assert source["producerTask"] == "compose-configs"
        assert source["outputArtifactKey"] == "composed_configs"


def test_each_evaluate_edge_reaches_the_model_its_role_names(tmp_path: Path) -> None:
    """Test each role edge resolves to that role's model: a transposition inverts
    every skill ratio and still writes a complete, plausible table.
    """
    ir = _compiled_ir(tmp_path, SYNTHETIC_MODEL_NAMES)
    tasks = ir["root"]["dag"]["tasks"]

    for role_input, model_name in (
        ("challenger_sidecar", SYNTHETIC_MODEL_ROLES.challenger),
        ("benchmark_sidecar", SYNTHETIC_MODEL_ROLES.benchmark),
    ):
        source = tasks["evaluate"]["inputs"]["artifacts"][role_input][
            "taskOutputArtifact"
        ]

        assert source["outputArtifactKey"] == "sidecar"
        assert _model_names_of({"producer": tasks[source["producerTask"]]}) == [
            model_name
        ]


def test_one_model_in_both_roles_puts_both_edges_on_one_task(tmp_path: Path) -> None:
    """Test that a one-model set emits one backtest task feeding both role edges."""
    roles = ModelRoles(benchmark="model_a", challenger="model_a")
    ir = _compiled_ir(tmp_path, ("model_a",), roles)
    tasks = ir["root"]["dag"]["tasks"]

    backtest_names = set(_backtest_tasks(ir))
    assert len(backtest_names) == 1

    producers = {
        tasks["evaluate"]["inputs"]["artifacts"][role]["taskOutputArtifact"][
            "producerTask"
        ]
        for role in ("challenger_sidecar", "benchmark_sidecar")
    }
    assert producers == backtest_names


def test_the_final_fit_task_fits_the_model_the_role_names(tmp_path: Path) -> None:
    """Test that final_fit fits the challenger: the benchmark would register cleanly."""
    tasks = _compiled_ir(tmp_path, SYNTHETIC_MODEL_NAMES)["root"]["dag"]["tasks"]

    assert _model_names_of({"final-fit": tasks["final-fit"]}) == [
        SYNTHETIC_MODEL_ROLES.challenger
    ]


def test_the_final_fit_task_runs_after_scoring_without_reading_it(
    tmp_path: Path,
) -> None:
    """Test the ordering edge, which nothing else in the template records, and that
    it stays ordering: a data edge would draw lineage where no bytes flow."""
    task = _compiled_ir(tmp_path, SYNTHETIC_MODEL_NAMES)["root"]["dag"]["tasks"][
        "final-fit"
    ]

    assert "evaluate" in task["dependentTasks"]
    producers = {
        artifact["taskOutputArtifact"]["producerTask"]
        for artifact in task["inputs"]["artifacts"].values()
    }
    assert "evaluate" not in producers


def test_the_final_fit_task_is_display_named_for_its_model(tmp_path: Path) -> None:
    """Test that the operator sees which model the bundle holds."""
    tasks = _compiled_ir(tmp_path, SYNTHETIC_MODEL_NAMES)["root"]["dag"]["tasks"]

    expected = f"final_fit-{SYNTHETIC_MODEL_ROLES.challenger}"
    assert tasks["final-fit"]["taskInfo"]["name"] == expected


def test_the_register_task_registers_the_bundle_final_fit_wrote(tmp_path: Path) -> None:
    """Test the bundle edge: KFP takes a bare Artifact for Input[Model], so any other
    output compiles, and the registry would record bytes final_fit never wrote."""
    tasks = _compiled_ir(tmp_path, SYNTHETIC_MODEL_NAMES)["root"]["dag"]["tasks"]

    source = tasks["register-model"]["inputs"]["artifacts"]["bundle"]
    assert source["taskOutputArtifact"] == {
        "producerTask": "final-fit",
        "outputArtifactKey": "bundle",
    }


def test_the_register_task_records_the_image_it_runs_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test the serving image is a constant equal to the task's own executor image,
    read from the component: the environment is made to disagree here to prove it."""
    monkeypatch.setenv(
        "FCST_TRAIN_IMAGE",
        "us-central1-docker.pkg.dev/nyc-taxi-ehc/fcst-ml-containers/train@sha256:"
        + "b" * 64,
    )
    ir = _compiled_ir(tmp_path, SYNTHETIC_MODEL_NAMES)

    parameters = ir["root"]["dag"]["tasks"]["register-model"]["inputs"]["parameters"]
    own_image = _executor_of(ir, "register-model")["container"]["image"]
    # Self-check: the component bound its image at import, before the change above.
    assert own_image != os.environ["FCST_TRAIN_IMAGE"]
    assert parameters["serving_container_image_uri"]["runtimeValue"] == {
        "constant": own_image
    }


def test_the_register_task_is_display_named_for_its_model(tmp_path: Path) -> None:
    """Test that the operator sees which model the task registers."""
    tasks = _compiled_ir(tmp_path, SYNTHETIC_MODEL_NAMES)["root"]["dag"]["tasks"]

    expected = f"register_model-{SYNTHETIC_MODEL_ROLES.challenger}"
    assert tasks["register-model"]["taskInfo"]["name"] == expected


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
        "additional_exog_uri",
    }
