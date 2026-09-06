"""Inference's project-owned configuration destinations — not yet built.

Two destinations belong here, sketched provisionally in
configs_schemas_images §4.4:

    InferenceInfraConfig
    InferenceModelingConfig

Inference also has a fourth configuration layer with no counterpart in Feature
or Training: the registered model's own composed config, read from the Model
Registry rather than from a file. Merging it needs three categories —
*inherited* (what the model is), *overridden* (the current run), and
*asserted-equal* (a mismatch invalidates the comparison) — which deep merge
cannot express, since it can make either side win but cannot refuse.

That is monthly_revenue_training_pipeline §10's open work, and it is why §4.4
is marked provisional. Nothing here is written until it is settled. See spec
§4.8 and ``config/inference/README.md``.
"""
