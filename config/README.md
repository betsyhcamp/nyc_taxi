# TODO

Once conventions are understood by the team, trim down comments and remove README's
within directory `config/`.

# Configuration

This tree is the single source of configuration for the Feature, Training, and
Inference pipelines. It is **baked into every Docker image** as the final layer,
copied to `/app/config`, so the config version is the image tag, which is the git
hash — one identifier covering code and config together.

Code never reads a hardcoded path. `core/` and `lib/` receive `config_dir: Path`
as a parameter, so the same function runs identically under `tmp_path` in a test,
from the working tree locally, and at `/app/config` inside a container.

This file states the rules the tree follows. The per-directory READMEs
(`feature/`, `inference/`) explain what is *missing* from those directories and
why.

## Where a fragment lives, and what validates it

**The directory says who reads it. The filename says which schema validates it.**

These two are perpendicular to each other, but they are a different pair from the
numbered **Axis 1** and **Axis 2** below. Those decide *what kind of value* you are
holding; these decide *where the file goes* once you know.

`train/` holds both infrastructure-destined and modeling-destined keys, so that
scoping lives entirely in filenames — `train/infra.yaml` beside
`train/backtest.yaml`. The directory cannot do that work and should not be asked
to.

No path feeds more than one destination.

## Axis 1 — modeling or infrastructure

> **Would changing this value change the forecast numbers?**
> Yes → *modeling* config. No → *infrastructure* config.

`freq`, `lags`, `forecast_origins`, `period_col`, `tier_labels` are modeling.
`bucket_name`, `display_name_prefix`, an image tag are infrastructure.

The real question this answers is *"does changing it invalidate comparison with
prior runs?"* — which is why the split exists at all.

**"Modeling" is a blast radius, not a subject.** It does not mean "pertaining to
a model." Feature has no model — it does not backtest — yet a SQL source query
is modeling configuration, because a different query means different data means
different numbers. Judge a value by what changing it does, not by what it is
about.

## Axis 2 — who owns the schema

Axis 1 alone cannot separate `evaluation_periods` from `cross_validation`: both
change the numbers. The second axis splits modeling config into:

- **tsbricks-owned** — `BacktestConfig` and its sub-models. Not ours to
  constrain; they keep whatever they declare.
- **project-owned** — parameters driving functionality that lives in `lib/`.
  tsbricks backtests *weekly* series; the monthly rollup, tiering, weighting, and
  origin derivation layered on top are ours.

Project-owned schemas set `extra="forbid"`. tsbricks' do not, which is why
composition carries a round-trip check for keys that were accepted and silently
discarded.

## The tree

Each fragment names its destination, and each destination names the module that
defines it. Schema paths are relative to `src/fcstnyctaxi/`.

```
config/
  README.md                       this file

  base/data.yaml                  → BacktestConfig, keys {data}    (tsbricks)
  environments/dev.yaml           → EnvironmentConfig         schemas/config/environment.py
  train/infra.yaml                → TrainInfraConfig          schemas/config/train.py
  train/modeling.yaml             → TrainModelingConfig       schemas/config/train.py
  train/backtest.yaml             → BacktestConfig            (tsbricks)
  train/models/<name>.yaml        → BacktestConfig, keys {model}   (tsbricks)
                                    naive.yaml, xgboost.yaml today

  feature/README.md               reserved; see that file    schemas/config/feature.py
  inference/README.md             reserved; see that file    schemas/config/inference.py

  configs_zone_demand_pipeline.yaml   legacy — see below      schemas/config_schemas.py
```

`schemas/config/common.py` holds concepts shared across those modules. The two
`(tsbricks)` destinations are defined in `tsbricks.backtesting.schema` and are not
ours to change — which is why the composition step carries a round-trip check for
keys they accept and silently discard.

The model files are named by `model_roles` in `train/modeling.yaml`, so
`<name>.yaml` is the shape; `naive` and `xgboost` are what it currently resolves
to.

## Composition, and why most fragments cannot be validated alone

Every destination but one composes from a **single** file:

| destination                 | fragments, in precedence order                                                  |
| --------------------------- | ------------------------------------------------------------------------------- |
| `EnvironmentConfig`         | `environments/<env>.yaml`                                                       |
| `TrainInfraConfig`          | `train/infra.yaml`                                                              |
| `TrainModelingConfig`       | `train/modeling.yaml`                                                           |
| `BacktestConfig` × N models | `base/data.yaml` → `train/backtest.yaml` → `train/models/<name>.yaml` → runtime |

**Layered merging is confined to `BacktestConfig`.** Mappings deep-merge, scalars
later-win, lists replace atomically.

**Fragments are partial by construction.** `train/backtest.yaml` has no `data`
block and no `model` block; `base/data.yaml` has only `data`. None of the three
`BacktestConfig` fragments can validate against that schema individually — a
fragment may legitimately omit a required field that arrives from a later layer
or at runtime. They are checked at fragment level instead: the file parses into a
non-empty mapping, and its top-level keys are within the set that fragment is
allowed to declare.

**That allowed set is usually the destination's whole field list, but two
fragments narrow it.** The rule that decides which:

> A fragment narrows when it is **defined by the block it carries**.

`base/data.yaml` is the data contract and nothing else, so `{data}`.
`train/models/<name>.yaml` is one model and nothing else, so `{model}`.
`train/backtest.yaml` does **not** narrow: its set would have to be everything
`BacktestConfig` declares minus what the other two layers own — a definition by
subtraction that goes stale the moment tsbricks adds a field, and then rejects a
valid fragment. Narrowing by what a file positively *is* cannot rot that way;
narrowing by what it is not can.

**`base/data.yaml`'s narrowing is load-bearing, not tidy.** `BacktestConfig` does
not set `extra="forbid"`, so without a narrower set this fragment may legally
declare *any* field of that schema. A stray `model:` block there would apply to
every model — and silently, because `base/data.yaml` is the **first** layer and
each model file merges on top, so it would surface only through whichever keys no
model happened to override. That is the same failure the round-trip drop check
exists for, arriving one layer earlier and with no error at all.

The narrower sets live in `src/fcstnyctaxi/lib/config/bindings.py` as data on the
binding, not as a branch inside a check, so relaxing one later is an edit.

**One value is injected at runtime**: `cross_validation.forecast_origins`,
derived from the panel's last complete month. Nothing else is.

**Pointers to bytes are not configuration.** Which panel and calendar a run read
are *lineage*, recorded in that run's `run_identity.json`, not in any file here.
The test is: *is this a pointer to bytes?* If yes, it is lineage. Note that
data-*derived* is not the same as lineage — `forecast_origins` is computed from
the panel and still belongs in the config, because it is an instruction.

## `environments/<env>.yaml` is the only environment-varying file

Every other fragment is **environment-independent**: there is no
`train/infra.dev.yaml`. So a value that must differ between dev and prod has
exactly one legal home, whichever slice it belongs to. That is why
`EnvironmentConfig` carries the BigQuery source project — a Feature concern — and
image references for all three slices, in a file Training also composes.

`EnvironmentConfig` is therefore not "the GCP block." It is **everything that
varies by environment, for all three slices**, grouped into operational planes:
`compute`, `storage`, `vertex`, `artifact_registry`, `source_data`.

**There is no `env` field inside it.** The environment is the *selector* that
chose the file, so declaring it inside would be a second source of a derived
fact: copy `dev.yaml` to `prod.yaml`, forget the field, and a file at the prod
path declares itself dev. The selected environment is recorded in each run's
emitted `manifest.json` instead.

**Adding an environment is adding a file** — no code edit. The allowed set is
discovered by globbing `environments/*.yaml`, so it cannot drift from the tree.
The tree is `.yaml` only, deliberately narrower than the loader's `.yaml`/`.yml`.

## The parity rule

> Every slice may have **up to** two project-owned configuration categories,
> `infra` and `modeling`. When a category exists it has exactly one destination
> and exactly one environment-independent fragment. Absence is made explicit in
> `src/fcstnyctaxi/lib/config/bindings.py`, which enumerates each slice's
> destinations — so a slice with no modeling config is a visible fact in code,
> not an inference from a missing file.

We do not manufacture empty placeholder files for symmetry. A slice gains a
tsbricks-owned destination only if it uses a tsbricks schema.

## One slice vocabulary

The three slices are **`feature`, `train`, `inference`**, and those exact tokens are
used everywhere a slice appears:

```
config/<slice>/                     these directories
core/<slice>/, components/<slice>/  code layout
schemas/config/<slice>.py           destination schemas
<env>/<slice>/<run_id>/<step>/      the storage convention
<slice>_run_id                      the run-id prefix
artifact_registry.images.<slice>    image references
```

`train`, not `training`. GCP resource names follow the vocabulary too: the Vertex
pipeline and its `display_name_prefix` are `fcst-train-pipeline`, and the
container images are the flat names `feature`, `train`, `inference` inside one
Docker repository, `fcst-ml-containers`. One repository rather than one per
slice, because Training is a single image serving five components — a
step-shaped image name would be structurally wrong for it.

**One grandfathered exception, and it no longer lives in this tree**: the ingress
code still names `fcst-data-ingress-pipeline` and `extract-db-to-bucket` — a
repository and image that already exist with a pushed tag, so renaming them would
mean a new registry repository and a re-push. Those names survive in the
Taskfile's build and push targets, the extract component's hardcoded fallback,
and `configs_zone_demand_pipeline.yaml`; the Feature rebuild replaces that code.
`environments/*.yaml` no longer carries them.

Ordinary English prose is outside the vocabulary as well — a training panel,
training rows. The canonical declaration is `SliceName` in
`src/fcstnyctaxi/schemas/config/common.py`.

## Conventions worth knowing before editing

- **Duplicate mapping keys are rejected.** PyYAML silently collapses `a: 1` /
  `a: 2` to `{"a": 2}` before any validation stage can observe the loss, so the
  loader raises instead.
- **Empty files and non-mapping documents are rejected**, with an error naming
  the file rather than four missing fields.
- **`null` carries meaning in exactly two places**, and both are commented at the
  point of use: `evaluation_periods.start_months: null` means *derive*, and
  `aggregation.calendar_source: null` means *declared but never injected*.
- **Image references carry repository and name only** — no tag, no digest. The
  tag is the git hash, resolved at build and submit time, so committing one here
  would put a second source of truth for a value the release process owns into a
  file that is itself baked into the image.
- **The registry is independent of compute**, so prod pulls the bytes dev ran and one git hash cannot mean two images.
- **Service accounts are not in this tree.** They are supplied at submit time via
  `FCST_{FEATURE,TRAIN,INFERENCE}_SERVICE_ACCOUNT`; see `.env.example`. This is an
  interim pending an infosec determination on whether identifiers may live in
  source control.

## The legacy file

`configs_zone_demand_pipeline.yaml` is the data-ingress pipeline's configuration
and predates this tree. It validates against `PipelineConfig` in
`src/fcstnyctaxi/schemas/config_schemas.py`, which is a flat schema mixing
project settings, an image URI, Vertex settings, and one step's parameters.

It is scheduled to be split and folded into `feature/` when the Feature pipeline
is rebuilt — see `feature/README.md`. Until then it is the live source for that
pipeline, and nothing here duplicates it.
