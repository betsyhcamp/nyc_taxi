# TODO: Once conventions are understood by the team, trim down comments
"""Inference's project-owned configuration destinations — not yet built.

The rules this tree follows are in ``config/README.md``.

Two destinations belong here, sketched only provisionally so far:

    InferenceInfraConfig
    InferenceModelingConfig

Inference also has a **fourth configuration layer with no counterpart in Feature
or Training**: the registered model's own composed config, read from the Model
Registry rather than from a file. That layer is what makes Inference's
composition structurally different from the other two slices, and it is unsolved.

Merging it needs three categories, not one:

    inherited        what the model is, carried forward from training
    overridden       the current run's own settings
    asserted-equal   a mismatch invalidates the comparison and must raise

Deep merge cannot express the third. It can make either side win; it cannot
refuse. Until that is settled, writing the first two destinations would fix a
precedence order the fourth layer may well change — so nothing here is defined,
and ``config/inference/README.md`` records the same for the YAML side.
"""
