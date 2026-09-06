"""Run identity — runtime-derived provenance, one frozen model per slice.

Each stage stamps everything upstream of it plus its own id
(monthly_revenue_training_pipeline §7.3). These are **contracts read across a
file boundary**: ``run_identity.json`` is emitted by a slice's
``compose_configs`` and read by every downstream impl in that slice, which is
what earns them a place in ``schemas/`` rather than ``lib/``.

Three flat models, **not** a hierarchy. The shared surface is one field
(``git_hash``); Feature cannot carry the URIs because it *produces* them; and
only one of the three exists to be written today. ``InferenceRunIdentity`` is
not built — its ``model_version`` depends on registration that does not exist
until PR 4.

**Pydantic rather than a frozen dataclass, decided by one field.**
``get_git_hash()`` shells out to ``git rev-parse HEAD``, and the container has
neither a repo (``.dockerignore`` excludes ``.git``) nor a git binary. On
failure it returns ``None`` with a warning rather than raising. A dataclass
accepts ``git_hash=None``, serialises ``null``, and stamps a null column on
every downstream table — the failure that already produced two registered
benchmarks whose ``git_hash`` cannot reproduce their artifacts. Typed ``str``
with ``min_length=1``, construction raises at compose time instead.

The ``gs://`` patterns close a second live slip: the local runner holds both
the original URI and the staged local ``Path``, and passing the wrong one would
stamp ``/tmp/scratch/panel.parquet`` as provenance with nothing raising.
"""

from pydantic import BaseModel, ConfigDict, Field


class FeatureRunIdentity(BaseModel):
    """Provenance of one Feature run's published artifacts.

    Built in PR 0a although Feature does not exist yet: §7.3 specifies it
    exactly, and run-scoped writes plus ``feature_run_id`` stamping is
    precisely what ``scripts/publish_feature_stand_in.py`` imitates.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    git_hash: str = Field(min_length=1)
    feature_run_id: str = Field(min_length=1)
    sql_sha256: str = Field(min_length=1)


class TrainRunIdentity(BaseModel):
    """Provenance of one Training run — which artifacts it read, and who made them.

    ``feature_run_id`` is the **join key** relating a training run's tables to a
    feature run's; ``panel_uri`` and ``calendar_uri`` are the **human pointer**.
    Both are cheap and they verify each other once Feature publishes
    run-scoped paths, since the URI then contains the id.

    An output type, not an input contract: a KFP wrapper cannot construct one,
    because ``feature_run_id`` comes from reading the panel and §3.0 forbids the
    wrapper from opening files.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    git_hash: str = Field(min_length=1)
    feature_run_id: str = Field(min_length=1)
    training_run_id: str = Field(min_length=1)
    panel_uri: str = Field(pattern=r"^gs://")
    calendar_uri: str = Field(pattern=r"^gs://")
