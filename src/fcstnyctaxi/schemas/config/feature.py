# TODO: Once conventions are understood by the team, trim down comments
"""Feature's project-owned configuration destinations — not yet built.

The rules this tree follows are in ``config/README.md``, including the two that
decide where a value belongs: **Axis 1** — *would changing this value change the
forecast numbers?* — and **Axis 2** — *who owns the schema, tsbricks or this
project?* Section references below point to design notes kept outside this
repository; everything needed to work on this module is stated here.

Two destinations belong here:

    FeatureInfraConfig
        display_name_prefix: str      flat — a one-field nested block is ceremony
        output: ExtractOutput         gcs_prefix · output_filename

    FeatureModelingConfig
        source_query: SourceQuery     filename · params
        # expected to accrete — see config/feature/README.md

They are deliberately **not** written at present. By Axis 1 the first two are
**modeling** — a different query means different data means different numbers —
and the last two are **infrastructure**. Splitting them is what gives Feature a
modeling config at all, and that split is assigned to the Feature pipeline's own
build. ``ExtractOutput.gcs_prefix`` is separately already scheduled to change
with the object migration.

**"Modeling" here is a blast radius, not a subject.** Axis 1 asks *would changing
this value change the forecast numbers?* — not *is this about a model?* Feature
has **no model at all**: it does not backtest, and no tsbricks schema validates
anything it owns. So ``FeatureModelingConfig`` holds the rules that decide what the
numbers are before anyone models them: a SQL query today, calendar derivation
rules and panel-admission rules tomorrow. ``config/feature/README.md`` lists what
the category is expected to collect, and judging the name against that list
rather than against ``source_query`` alone is the point — the name is provisional
and these classes are a forecast, not a commitment.

**Source-data configuration does not belong in this module.** The BigQuery source
project and location live in ``EnvironmentConfig.source_data`` — *not* as a field
on ``FeatureInfraConfig``, and not as a ``source_data:`` block in
``config/feature/infra.yaml``. The reason is structural: the parity rule gives
each slice exactly one **environment-independent** fragment per category, so
``config/environments/<env>.yaml`` is the only file in the tree that can hold a
value differing between dev and prod. A source dataset that differs by
environment has nowhere else to go.
"""
