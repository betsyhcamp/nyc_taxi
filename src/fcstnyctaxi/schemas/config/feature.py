"""Feature's project-owned configuration destinations — not yet built.

Two destinations belong here, specified in configs_schemas_images §4.2:

    FeatureInfraConfig      display_name_prefix · output (ExtractOutput)
    FeatureModelingConfig   source_query (SourceQuery: filename · params)

They are deliberately **not** written in PR 0a. Both encode the
``ExtractDbToBucketConfig`` split — ``sql_filename`` and ``sql_params`` are
modeling config (a different query means different data means different
numbers) while ``gcs_prefix`` and ``output_filename`` are infrastructure — and
monthly_revenue_training_pipeline §14 assigns that split to the Feature
pipeline's own build. §9 of configs_schemas_images additionally flags
``ExtractOutput.gcs_prefix`` as already scheduled to change with the object
migration.

Writing them now would ship a known-stale value into a file nothing reads,
creating a second live source for Feature's configuration alongside
``config/configs_zone_demand_pipeline.yaml`` — the exact duplication the config
tree exists to remove. See spec §4.8 and ``config/feature/README.md``.
"""
