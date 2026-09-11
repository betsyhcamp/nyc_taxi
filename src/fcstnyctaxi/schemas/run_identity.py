# TODO: Once conventions are understood by the team, trim down comments
"""Run identity — runtime-derived provenance, one frozen model per slice.

Section references below point to design notes kept outside this repository;
everything needed to work on this module is stated here.

Each stage stamps everything upstream of it plus its own id. These are
**contracts read across a file boundary**: ``run_identity.json`` is emitted by a
slice's ``compose_configs`` and read by every downstream impl in that slice,
which is what earns them a place in ``schemas/`` rather than ``lib/``.

The id prefixes are the **slice tokens** — ``feature``, ``train``, ``inference``
— the same three that name ``config/<slice>/``, ``core/<slice>/``, and the
``<env>/<slice>/<run_id>/<step>/`` storage convention. One vocabulary, used
everywhere. (note we are using ``train``, **NOT** ``training``)

Three flat models, **not** a hierarchy. The shared surface is one field
(``git_hash``); Feature cannot carry the URIs because it *produces* them; and
only one of the three exists to be written today. ``InferenceRunIdentity`` is
not built — its ``model_version`` depends on registration that does not exist
until later.

The ``gs://`` patterns close a second live slip: the local runner holds both
the original URI and the staged local ``Path``, and passing the wrong one would
stamp ``/tmp/scratch/panel.parquet`` as the source with nothing raising.

``sql_sha256``'s hex pattern closes a third slip and is lowercase only, because
``hexdigest()`` is lowercase and accepting both cases would let two spellings of one
hash compare unequal. The prefixed ``"sha256:…"`` form is the convention used for
config-file hashes in each run's emitted ``manifest.json``, and is deliberately not
accepted here — one field, one spelling.
"""

from pydantic import BaseModel, ConfigDict, Field


class FeatureRunIdentity(BaseModel):
    """Provenance of one Feature run's published artifacts.

    Built although Feature does not exist yet: the three fields are
    already specified, and run-scoped writes plus ``feature_run_id`` stamping is
    precisely what ``scripts/publish_feature_stand_in.py`` imitates.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    git_hash: str = Field(min_length=1)
    feature_run_id: str = Field(min_length=1)
    sql_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class TrainRunIdentity(BaseModel):
    """Provenance of one Training run — which artifacts it read, and who made them.

    ``feature_run_id`` is the **join key** relating a training run's tables to a
    feature run's; ``panel_uri`` and ``calendar_uri`` are the **human pointer**.
    Both are cheap and they verify each other once Feature publishes
    run-scoped paths, since the URI then contains the id.

    An output type, not an input contract: a KFP wrapper cannot construct one,
    because ``feature_run_id`` comes from reading the panel, and the impl/wrapper
    contract puts all file IO in the impl — a wrapper never opens the files it
    causes to be written.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    git_hash: str = Field(min_length=1)
    feature_run_id: str = Field(min_length=1)
    train_run_id: str = Field(min_length=1)
    panel_uri: str = Field(pattern=r"^gs://")
    calendar_uri: str = Field(pattern=r"^gs://")
