# Inference pipeline configuration

This directory is intentionally empty of fragments. It is reserved, and this
file records what belongs here and why it was not written.

## Destinations that belong here

| destination               | fragment                  | contents    |
| ------------------------- | ------------------------- | ----------- |
| `InferenceInfraConfig`    | `inference/infra.yaml`    | provisional |
| `InferenceModelingConfig` | `inference/modeling.yaml` | provisional |

`EnvironmentConfig` is composed from `../environments/<env>.yaml`, shared with
Feature and Training.

The sketch is in `planning_context__configs_schemas_images.md` §4.4, which is
marked **provisional**, and stubbed in
`src/fcstnyctaxi/schemas/config/inference.py`.

## Why nothing is written yet

Inference has a fourth configuration layer with no counterpart in Feature or
Training: **the registered model's own composed config**, read from the Model
Registry rather than from a file. That layer is what makes Inference's
composition structurally different from the other two slices, and it is unsolved.

Merging it needs three categories, not one:

- **inherited** — what the model is, carried forward from training
- **overridden** — the current run's own settings
- **asserted-equal** — a mismatch invalidates the comparison and must *raise*

Deep merge cannot express the third. It can make either side win; it cannot
refuse. Until that is settled — `planning_context__monthly_revenue_training_pipeline.md`
§10 — writing the first two fragments would fix a precedence order that the
fourth layer may well change.

## What to do when you build Inference

1. Settle the three-category merge for the registered model's config. This is a
   design decision, not an implementation detail, and it is the blocker.
1. Define the two schemas in `src/fcstnyctaxi/schemas/config/inference.py`.
1. Write the fragments here, one per destination.
1. Add `inference_bindings()` entries in
   `src/fcstnyctaxi/lib/config/bindings.py`, which currently declares Inference
   as having no project-owned destinations.

Read the parity rule in §4.6 of the configs document first: a slice may have
**up to** two project-owned categories, and absence is stated explicitly in
`bindings.py` rather than inferred from a missing file. Do not add empty
placeholder fragments for symmetry with Training.
