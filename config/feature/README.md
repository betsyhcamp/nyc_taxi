# TODO

Once conventions are understood by the team, trim comments and remove the README's inside
`config/`.

# Feature pipeline configuration

Intentionally empty of fragments. This file records what belongs here and why it is not
written yet. Read `config/README.md` first for the rules this tree follows: the two axes,
the parity rule, and why `environments/<env>.yaml` is the only environment-varying file.

## Destinations that belong here

| destination             | fragment                | contents                                                           |
| ----------------------- | ----------------------- | ------------------------------------------------------------------ |
| `FeatureInfraConfig`    | `feature/infra.yaml`    | `display_name_prefix`, `output` (`gcs_prefix` · `output_filename`) |
| `FeatureModelingConfig` | `feature/modeling.yaml` | `source_query` (`filename` · `params`)                             |

`EnvironmentConfig` is composed from `../environments/<env>.yaml`, shared with Training and
Inference. Feature has no tsbricks-owned destination, since it does not backtest. The
schemas are stubbed, with their intended fields, in
`src/fcstnyctaxi/schemas/config/feature.py`.

## What `modeling.yaml` is expected to collect

`source_query` is the first member of this category, not the whole of it, and being
SQL-shaped it makes the category look narrower than it is. "Modeling" here is **Axis 1**,
*would changing this value change the forecast numbers?*, not *is this about a model?*
Feature has no model. The category is **the rules that decide what the numbers are before
anyone models them**. Both axes are defined in `config/README.md`.

Everything Feature derives is a candidate, so every rule producing a column in the panel
(`ds` · `unique_id` · `y`) or in the fiscal calendar belongs here:

| candidate                                                                             | why it belongs, and where it lives today                                                                                                                    |
| ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Fiscal calendar rules**: year start, 4-4-5 vs 4-5-4, week start day                 | Changing the period boundaries changes every monthly total. Implicit in SQL today                                                                           |
| **`weeks_in_month` derivation**                                                       | The observed-max derivation is correct only because of a `WHERE` clause in a different file that nothing states or checks. A stated rule is the durable fix |
| **Calendar horizon**: how far past the panel the calendar extends                     | Training's `_build_future_calendar_df` and runtime assertion 2 both need it                                                                                 |
| **Workday and holiday definitions**: region, which days count                         | `count_workdays` ships today; the holiday source defining it is unstated                                                                                    |
| **Panel scope**: zone or borough grain, Manhattan-only, the month cutoff              | Hardcoded in `queries/initial_daily_taxi_rides.sql`'s `month_cutoff_cte`                                                                                    |
| **Series admission rules**: minimum history, first-active month, gap versus zero fill | Currently expressed in notebook code rather than config                                                                                                     |
| **Exogenous feature toggles**: holiday flags, event calendars                         | Anything Feature precomputes and ships as extra columns                                                                                                     |

Judge the category name against this list rather than against `source_query` alone. If
`FeatureModelingConfig` still reads wrong once these are real, rename it then. The
alternative under consideration is renaming the category across all three slices, to
`<Slice>MethodConfig` or `<Slice>DerivationConfig`, rather than Feature's alone, which
would leave no shared name for the second category.

## Source-data configuration does not live here

The BigQuery source project and location belong in `EnvironmentConfig.source_data`, in
`config/environments/<env>.yaml`. Do **not** add a `source_data:` block to `infra.yaml` or
`modeling.yaml` here, and do not add a corresponding field to `FeatureInfraConfig`.

The reason is structural. The parity rule gives each slice **exactly one
environment-independent fragment** per category, so there is no `feature/infra.dev.yaml`,
and `config/environments/<env>.yaml` is the only file in the tree that can hold a value
differing between dev and prod. A source dataset that varies by environment has nowhere
else to go.

The BigQuery **job** project is a separate value and is already there as
`compute.project_id`: the project that pays for the query need not be the one holding the
tables. `source_data.location` is not a free choice, since a query job must run in its
dataset's location.

## What to do when you build Feature

1. Define `FeatureInfraConfig` and `FeatureModelingConfig` in
   `src/<project_package_name>/schemas/config/feature.py`, replacing the module docstring.
1. Write `infra.yaml` and `modeling.yaml` here, one fragment per destination.
1. Add `feature_bindings()` entries in `src/<project_package_name>/lib/config/bindings.py`,
   which currently declares Feature as having no project-owned destinations. That absence
   is deliberate, not an oversight.
1. Retire the corresponding keys from existing upstream configs, so there is one live
   source.
1. Rename `PipelineConfig` to `FeaturePipelineConfig` in
   `src/<project_package_name>/schemas/config_schemas.py`. It is a flat schema mixing
   project settings, an image URI, Vertex settings and one step's parameters; the rename is
   owed once this tree replaces it.

## How Training finds Feature's output

**Rung 1: Training is told.** Both training callers take explicit `--panel-uri`,
`--calendar-uri` and `--additional-exog-uri`. They survive as optional overrides, all three
required together.

**Rung 2: Training resolves, and this ships.** Given a `--feature-run-id` alone, both
callers read the outputs file below and take the three URIs by role. Resolution is
caller-side only, so the pipeline is still *told* where its inputs are; what changed is who
computes the URIs. The reader is `src/<project_package_name>/lib/run_outputs.py`; the
contract as a typed model is `src/<project_package_name>/schemas/run_outputs.py`.

### The rule that constrains it

> **Training must not need to know Feature's internal step names.**

An earlier design had Training derive
`gs://<bucket>/<env>/feature/<feature_run_id>/data_prep/<filename>`. `data_prep` is not a
Feature step name; it is named after `notebooks/data_prep.py`, the notebook the stand-in
imitates, so Training would have been coupled to a token that never described your
pipeline. More generally, a path template makes Training's correctness depend on your
layout: rename a step, or split calendar derivation into its own step, and Training breaks
with a 404, at runtime, in someone else's code.

So that design was dropped rather than handed over half-built. `config/train/infra.yaml`
still carries **no** `feature_source` block and no filenames, and the rule is now
mechanically checked: `grep -rn "data_prep" src/ scripts/` returns only the stand-in.

### The design: a run-root outputs file

Feature writes one small file per run, at the **run root**:

```
gs://<bucket>/<env>/feature/<feature_run_id>/run_output.json
```

Training builds that one path from `<bucket>`, `<env>`, `feature` and the `feature_run_id`
it was given, reads it, and takes the URIs by name. It learns no step name and no filename.
Four properties make this work, and each was a mistake we made first:

1. **Written as the pipeline's final act, never at step 1.** A URI stamped by the first
   step is a *promise*, not a record: if a later step fails, the file names an object that
   does not exist and the consumer gets an authoritative-looking dangling pointer.
1. **Its presence is the completion signal.** Absent means the run did not finish, so a
   pipeline that completes must write it **even with nothing to declare**, or absence is
   ambiguous between *failed* and *nothing to say*.
1. **At the run root, not inside a step directory.** Under a step it inherits the problem
   it exists to solve: you would need the step name to find the file whose job is telling
   you the step name.
1. **It carries its own `feature_run_id`.** Training compares that against the id it was
   given, catching a file copied between run directories or a wrong id passed.

Properties 1 and 2 work only together. Training reports a missing manifest as *this run did
not complete*, never as a missing object, and that reading is valid only if the file really
is written last.

### The schema, as chosen

The schema was yours to choose and you chose it. What ships is recorded here so both sides
have one statement of it. `src/<project_package_name>/schemas/run_outputs.py` is the same
contract as a typed model, and is the authority if the two ever disagree.

| key                                              | type                        | read by Training                            |
| ------------------------------------------------ | --------------------------- | ------------------------------------------- |
| `feature_run_id`                                 | string                      | guard, compared against the id it was given |
| `published.panel_uri`                            | `gs://` string              | **load-bearing**                            |
| `published.calendar_uri`                         | `gs://` string              | **load-bearing**                            |
| `published.exogenous_uri`                        | `gs://` string              | **load-bearing**                            |
| `env`                                            | string                      | guard, compared when present                |
| `git_hash`                                       | string, 7 characters        | no                                          |
| `completed_at`                                   | ISO 8601 string with offset | no                                          |
| `panel.rows` · `panel.series`                    | int                         | no                                          |
| `panel.first_ds` · `panel.last_ds`               | date string                 | no                                          |
| `panel.series_admitted` · `panel.series_dropped` | int                         | no                                          |
| `panel.exogenous_columns`                        | list of strings             | no                                          |

The keys under `published` are role names; the first two match what `TrainRunIdentity`
already calls them. `env` was volunteered rather than asked for, and earns its place as a
second copy-detector.

**What Training reads, and what it merely tolerates.** Only the three load-bearing keys,
the two guards and `schema_version`, an optional string, are typed. Everything else is
opaque and unvalidated, and unknown keys are ignored at both levels. **Adding a key is
always safe.** The only change that breaks Training is renaming or removing something under
`published`, or writing `""` as `published.exogenous_uri`: Training refuses it until every
step can train without that file.

**Two departures from the design above.**

1. **No `sql_sha256`.** `FeatureRunIdentity` declares it and nothing writes it, so the
   query that produced a run's data is captured nowhere. `source_query` is modeling
   configuration because *a different query means different data means different numbers*,
   which makes the query hash the field that says whether two Feature runs are comparable.
   Still owed, and yours to add, as one field here rather than a second file.
1. **A `panel` statistics block** this design never anticipated, and a real improvement.
   Training reads nothing under it, ever, and types the whole block opaque rather than as a
   mapping: a consumer comparing `exogenous_columns` against an expected list would turn
   adding a feature to Feature into a failure in a Training step that never reads it.

### One field asked for, with one rule

`schema_version` is not in the file today. It is the single addition this project asks for,
with one sentence scoping when it moves:

> Bump when any key under `published` is added, removed or renamed. Changes to the `panel`
> block do not need a bump.

The narrow scope is what makes it cheap: no bump for statistics changes, and it protects
exactly the keys resolution depends on. Training types it as an optional string, such as
`"0.1.0"`, and treats its value as a diagnostic, never a gate: an unrecognized version
warns and proceeds, and an absent one is not an error, so the reader works whether or not
the field ever arrives.

### Where this fits the placement rule

> A file belongs at the **run root** iff a reader **outside this pipeline** must find it
> with only `<bucket>`, `<env>`, `<slice>` and `<run_id>`. Everything else lives in the
> directory of the step that produced it.

`run_output.json` qualifies: its reader is the next pipeline, which knows only the
`feature_run_id` it was handed. `run_identity.json` qualifies too, and Training already
writes it that way:

```
gs://<bucket>/<env>/train/<train_run_id>/run_identity.json
gs://<bucket>/<env>/train/<train_run_id>/compose_configs/   the first step's own outputs
```

The rule is about **readers, not content**. `run_identity.json` was always run-level and
belonged in the step directory for as long as its only readers were downstream steps;
gaining one outside reader, the submission script, is what moved it. The two files still
have **different jobs**: identity is what a run *is* and what it *read*, true at step 1;
outputs are what it *produced*, true only at the end. Do not merge them, since a failed run
should still record its identity.

### Rung 3, given the pointer you publish

Rung 3 serves the case where **nobody knows an id**: a scheduled Training run not chained
to a specific Feature run, or a developer who wants whatever is current. It resolves from
`_latest.json`, which you rewrite after each successful run. The default itself is not
built and nothing here needs it; it was waiting on the pointer's shape, not on you.

The pointer is **one per environment, not one per slice**, at the environment root:

```
gs://<bucket>/<env>/_latest.json
```

Its keys are the slice names `feature`, `train` and `inference`, and **each holds that
slice's whole `run_output.json` document**, not an id. Each pipeline rewrites its own key
and leaves the others equal in value and in order.

Training's `register_model` rewrites `train` after writing its completion marker, so the
pointer never names a run that did not finish. It drops `feature_run_id` from that block
alone: `feature` already names the newest Feature run, while `train.feature_run_id` would
name the one this model trained on, and the two diverge the moment Feature runs again.

The old objection, that a pointer cannot share a path convention with the run-scoped
artifacts it names, is answered by placement, not content: it sits above every slice.

One property worth preserving: a pointer is mutable, and is the only mutable object in a
layout that is otherwise immutable by construction. Runs stay reproducible anyway, because
whatever resolution produces is stamped into `run_identity.json` as `panel_uri`,
`calendar_uri` and `feature_run_id`. A run records what it read, not how it found it.
