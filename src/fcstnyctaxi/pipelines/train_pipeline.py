# No `__future__.annotations` import: KFP reads annotations as objects at decoration
# time and PEP 563 makes them strings, which fails this pipeline at compile.

from kfp import dsl

from fcstnyctaxi.components.train.compose_configs_component import compose_configs


@dsl.pipeline(name="fcst-train-pipeline")
def train_pipeline(
    env: str,
    train_run_id: str,
    feature_run_id: str,
    panel_uri: str,
    calendar_uri: str,
) -> None:
    """Compose every Training config for one run: the Training DAG's first step.

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
    compose_configs(  # type: ignore[call-arg]
        env=env,
        train_run_id=train_run_id,
        feature_run_id=feature_run_id,
        panel=panel.output,
        calendar=calendar.output,
    )
