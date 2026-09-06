# Feature pipeline configuration

This directory is intentionally empty of fragments. It is reserved, and this
file records what belongs here and why it was not written.

## Destinations that belong here

| destination             | fragment                | contents                                                           |
| ----------------------- | ----------------------- | ------------------------------------------------------------------ |
| `FeatureInfraConfig`    | `feature/infra.yaml`    | `display_name_prefix`, `output` (`gcs_prefix` · `output_filename`) |
| `FeatureModelingConfig` | `feature/modeling.yaml` | `source_query` (`filename` · `params`)                             |

`EnvironmentConfig` is composed from `../environments/<env>.yaml`, shared with
Training and Inference. Feature has no tsbricks-owned destination — it does not
backtest.

The schemas are specified in `planning_context__configs_schemas_images.md` §4.2
and stubbed in `src/fcstnyctaxi/schemas/config/feature.py`.

## Why nothing is written yet

Both destinations encode the `ExtractDbToBucketConfig` split. Today
`sql_filename`, `sql_params`, `gcs_prefix`, and `output_filename` live in one
block in `config/configs_zone_demand_pipeline.yaml`. The first two are
**modeling** configuration — a different query means different data means
different numbers — and the last two are **infrastructure**. Splitting them is
what gives Feature a modeling config at all.

`planning_context__monthly_revenue_training_pipeline.md` §14 assigns that split
to the Feature pipeline's own build, and §9 of the configs document flags
`ExtractOutput.gcs_prefix` as already scheduled to change with the object
migration. Writing these fragments now would ship a known-stale value into a
file nothing reads, creating a second live source for Feature's configuration
alongside `config/configs_zone_demand_pipeline.yaml` — the exact duplication
this tree exists to remove.

## What to do when you build Feature

1. Define `FeatureInfraConfig` and `FeatureModelingConfig` in
   `src/fcstnyctaxi/schemas/config/feature.py`, replacing the module docstring.
1. Write `infra.yaml` and `modeling.yaml` here, one fragment per destination.
1. Add `feature_bindings()` entries in `src/fcstnyctaxi/lib/config/bindings.py`,
   which currently declares Feature as having no project-owned destinations.
   Absence there is a deliberate fact, not an oversight.
1. Retire the corresponding keys from
   `config/configs_zone_demand_pipeline.yaml` so there is one live source.
1. Rename `PipelineConfig` to `FeaturePipelineConfig` in
   `src/fcstnyctaxi/schemas/config_schemas.py`, per §14.
