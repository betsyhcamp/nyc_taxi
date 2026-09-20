# No `__future__.annotations`: PEP 563 strings break KFP's annotation read at compile.

from kfp import dsl

from fcstnyctaxi.components.train.backtest_component import backtest
from fcstnyctaxi.components.train.compose_configs_component import compose_configs
from fcstnyctaxi.components.train.evaluate_component import evaluate
from fcstnyctaxi.schemas.config.train import ModelRoles


def _require_fannable_model_names(model_names: tuple[str, ...]) -> None:
    """Reject a model set the DAG cannot fan out over faithfully.

    Empty backs no model, writes nothing and exits clean. Duplicates do not raise
    in KFP: it emits a second positional task carrying one model_name, and both
    resolve to one sidecar directory, so two tasks race on eight files.

    Args:
        model_names (tuple[str, ...]): The set the DAG would emit a task per.

    Raises:
        ValueError: If `model_names` is empty or repeats a name.
    """
    if not model_names:
        raise ValueError(
            "model_names is empty, so this pipeline would compose every config and "
            "back no model, which looks like a successful run."
        )
    if len(set(model_names)) != len(model_names):
        raise ValueError(
            f"model_names {model_names} repeats a name: each task derives its "
            "sidecar directory from the model name, so two would write to one."
        )


def build_train_pipeline(
    *, model_names: tuple[str, ...], model_roles: ModelRoles
) -> dsl.base_component.BaseComponent:
    """The Training DAG for one compiled model set.

    Takes the model set rather than reading it: @dsl.pipeline traces the body when
    the decorator runs, so a module-level read would freeze one set into the
    import, where no test could vary it.

    Args:
        model_names (tuple[str, ...]): One backtest task per entry, in order.
        model_roles (ModelRoles): The roles evaluate's two edges are wired from;
            both must name entries in `model_names`.

    Raises:
        ValueError: If `model_names` is empty or repeats a name.

    Returns:
        BaseComponent: The traced pipeline, for `Compiler().compile()`.
    """
    _require_fannable_model_names(model_names)

    @dsl.pipeline(name="fcst-train-pipeline")
    def train_pipeline(
        env: str,
        train_run_id: str,
        feature_run_id: str,
        panel_uri: str,
        calendar_uri: str,
    ) -> None:
        """Compose every Training config for one run, back each model, then score.

        Feature is a separate pipeline, so its two artifacts arrive as URIs rather than
        from an upstream task; dsl.importer types each and registers it in ML Metadata.
        No run_prefix parameter: it needs bucket_name from a destination composed at
        runtime, so only the wrapper can resolve it.
        """
        panel = dsl.importer(
            artifact_uri=panel_uri, artifact_class=dsl.Dataset, reimport=False
        )
        calendar = dsl.importer(
            artifact_uri=calendar_uri, artifact_class=dsl.Dataset, reimport=False
        )
        # type: ignore since a type checker sees the undecorated function, whose
        # Output[Artifact] parameter the decorator supplies.
        compose = compose_configs(  # type: ignore[call-arg]
            env=env,
            train_run_id=train_run_id,
            feature_run_id=feature_run_id,
            declared_model_names=list(model_names),
            panel=panel.output,
            calendar=calendar.output,
        )
        # Plain Python at decoration time: one task per model. Both importer
        # handles are reused, so every task reads the artifact compose validated.
        # Kept by name so evaluate can select two of them by role.
        backtest_tasks = {}
        for model_name in model_names:
            backtest_tasks[model_name] = backtest(  # type: ignore[call-arg]
                run_prefix=compose.outputs["run_prefix"],
                model_name=model_name,
                composed_configs=compose.outputs["composed_configs"],
                panel=panel.output,
                calendar=calendar.output,
            ).set_display_name(f"backtest-{model_name}")

        # No importer handles: the sidecars hold every input. No set_display_name:
        # one task, so taskInfo.name already reads "evaluate".
        challenger = backtest_tasks[model_roles.challenger]
        benchmark = backtest_tasks[model_roles.benchmark]
        evaluate(  # type: ignore[call-arg]
            run_prefix=compose.outputs["run_prefix"],
            composed_configs=compose.outputs["composed_configs"],
            challenger_sidecar=challenger.outputs["sidecar"],
            benchmark_sidecar=benchmark.outputs["sidecar"],
        )

    return train_pipeline
