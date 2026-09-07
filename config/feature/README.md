# TODO

Once conventions are understood by the team, trim down comments and remove README's
within directory `config/`.

# Feature pipeline configuration

This directory is intentionally empty of fragments. It is reserved, and this
file records what belongs here and why it was not written.

Read `config/README.md` first for the rules this tree follows — the two axes, the
parity rule, and why `environments/<env>.yaml` is the only environment-varying
file. What's needed to build Feature's configuration is stated here.

## Destinations that belong here

| destination             | fragment                | contents                                                           |
| ----------------------- | ----------------------- | ------------------------------------------------------------------ |
| `FeatureInfraConfig`    | `feature/infra.yaml`    | `display_name_prefix`, `output` (`gcs_prefix` · `output_filename`) |
| `FeatureModelingConfig` | `feature/modeling.yaml` | `source_query` (`filename` · `params`)                             |

`EnvironmentConfig` is composed from `../environments/<env>.yaml`, shared with
Training and Inference. Feature has no tsbricks-owned destination — it does not
backtest.

The schemas are stubbed, with their intended fields, in
`src/fcstnyctaxi/schemas/config/feature.py`.

## Why nothing is written yet

Both destinations encode the `ExtractDbToBucketConfig` split. Today
`sql_filename`, `sql_params`, `gcs_prefix`, and `output_filename` live in one
block in `config/configs_zone_demand_pipeline.yaml`. The first two are
**modeling** configuration — a different query means different data means
different numbers — and the last two are **infrastructure**. Splitting them is
what gives Feature a modeling config at all.

That split is assigned to the Feature pipeline's own build, and
`ExtractOutput.gcs_prefix` is separately already scheduled to change with the
object migration. Writing these fragments now would ship a known-stale value into a
file nothing reads, creating a second live source for Feature's configuration
alongside `config/configs_zone_demand_pipeline.yaml` — the exact duplication
this tree exists to remove.

## What `modeling.yaml` is expected to collect

`source_query` is the first member of this category, not the whole of it — and
because it is SQL-shaped, it makes the category look narrower than it is.
"Modeling" here is **Axis 1** — *would changing this value change the forecast
numbers?* — not *is this about a model?* Feature has no model. The category is
**the rules that decide what the numbers are before anyone models them**. Axis 1
and Axis 2 are both defined in `config/README.md`.

Everything Feature derives is a candidate. The panel carries `ds` · `unique_id` ·
`y`, and the fiscal calendar carries `fiscal_year_month`,
`origin_month_fraction_elapsed`, `fiscal_week_of_month`, `fiscal_month`,
`weeks_in_month`, and `count_workdays` — so every rule producing those columns
belongs here:

| candidate                                                                              | why it belongs, and where it lives today                                                                                                                    |
| -------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Fiscal calendar rules** — year start, 4-4-5 vs 4-5-4, week start day                 | Changing the period boundaries changes every monthly total. Implicit in SQL today                                                                           |
| **`weeks_in_month` derivation**                                                        | The observed-max derivation is correct only because of a `WHERE` clause in a different file that nothing states or checks. A stated rule is the durable fix |
| **Calendar horizon** — how far past the panel the calendar extends                     | Training's `_build_future_calendar_df` and runtime assertion 2 both need it. This is a live entry condition for PR 3                                        |
| **Workday and holiday definitions** — region, which days count                         | `count_workdays` ships today; the holiday source defining it is unstated                                                                                    |
| **Panel scope** — zone or borough grain, Manhattan-only, the month cutoff              | Hardcoded in `queries/initial_daily_taxi_rides.sql`'s `month_cutoff_cte`                                                                                    |
| **Series admission rules** — minimum history, first-active month, gap versus zero fill | Currently expressed in notebook code rather than config                                                                                                     |
| **Exogenous feature toggles** — holiday flags, event calendars                         | Anything Feature precomputes and ships as extra columns                                                                                                     |

Judge the category name against this list rather than against `source_query`
alone. If `FeatureModelingConfig` still reads wrong once these are real, that is
the moment to rename it. The alternative under consideration is renaming the
category across all three slices — `<Slice>MethodConfig` or
`<Slice>DerivationConfig` — rather than renaming Feature's alone, which would
leave no shared name for the second category at all.

## Source-data configuration does not live here

The BigQuery source project and location belong in
`EnvironmentConfig.source_data`, in `config/environments/<env>.yaml`. Do **not**
add a `source_data:` block to `infra.yaml` or `modeling.yaml` here, and do not
add a corresponding field to `FeatureInfraConfig`.

The reason is structural rather than stylistic. The parity rule gives each slice
**exactly one environment-independent fragment** per category — there is no
`feature/infra.dev.yaml` — so `config/environments/<env>.yaml` is the only file
in the tree that can hold a value differing between dev and prod. A source
dataset that differs by environment has nowhere else to go.

The BigQuery **job** project is a separate value and is already there too, as
`compute.project_id` — the project that pays for the query, which need not be
the project holding the tables. `source_data.location` is not a free choice: a
query job must run in its dataset's location.

## What to do when you build Feature

1. Define `FeatureInfraConfig` and `FeatureModelingConfig` in
   `src/<project_package_name>/schemas/config/feature.py`, replacing the module docstring.
1. Write `infra.yaml` and `modeling.yaml` here, one fragment per destination.
1. Add `feature_bindings()` entries in `src/<project_package_name>/lib/config/bindings.py`,
   which currently declares Feature as having no project-owned destinations.
   Absence there is a deliberate fact, not an oversight.
1. Retire the corresponding keys from upstream existing configs (if any) so
   there is one live source.
1. Rename `PipelineConfig` to `FeaturePipelineConfig` in
   `src/<project_package_name>/schemas/config_schemas.py`. It is a flat schema mixing project settings, an image URI, Vertex settings, and one step's
   parameters, and the rename is owed once this tree replaces it.

## If Feature publishes a latest-run pointer

Training resolves the panel and calendar by `feature_run_id`, deriving
`gs://<bucket>/<env>/feature/<feature_run_id>/data_prep/<filename>`. That is rung 2
of a three-rung resolution ladder — rung 1 is explicit `--panel-uri` /
`--calendar-uri` overrides, deferred; rung 2 is the derivation above; rung 3 is the
pointer below. Rung 2 works whenever the caller already knows an id, which Cloud
Workflows does because it minted it.

Rung 3 is a `_latest.json` pointer that Feature rewrites after each successful run,
naming its newest `feature_run_id`. It serves the case where **nobody knows an id** —
a scheduled Training run not chained to a specific Feature run, or a developer who
wants whatever is current. The `_latest.json` pointer is unbuilt because it needs a commitment from Feature,
not from Training.

**If you build it, the pointer must live at a fixed, non-run-scoped path** — something
like `gs://<bucket>/<env>/feature/_latest.json`. It cannot use `build_feature_uri`'s
convention, because you would need the `feature_run_id` to find the file whose only
job is naming the `feature_run_id`. A pointer and the artifacts it points at cannot
share a path convention, since that convention is exactly what the pointer shortcuts.

An earlier draft of `config/train/infra.yaml` carried `pointer_filename: _latest.json`
inside `feature_source`, beside the two run-scoped filenames. It was removed: its
placement implied `build_feature_uri` would construct its URI, which run-scopes it.
When Feature commits to writing a pointer, Training gains rung 3 and a config entry
naming **both** the fixed prefix and the filename.

One property worth preserving: a pointer is mutable, and it would be the only mutable
object in a storage layout that is otherwise immutable by construction. Runs stay
reproducible anyway, because whatever resolution produces is stamped into
`run_identity.json` as `panel_uri`, `calendar_uri`, and `feature_run_id` — the run
records what it read, not how it found it.
