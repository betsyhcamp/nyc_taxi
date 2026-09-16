"""Feature's run-root outputs manifest run_outputs.json, as a typed input contract.

Written by the Feature pipeline as last action and read by Training's callers to
resolve two artifact URIs from a ``feature_run_id`` alone.

No ``extra="forbid"``, unlike ``run_identity.py``: that is right for a record this
repo writes and wrong for a third party's, where an added field would break a
consumer that never reads it. ``frozen=True`` does carry over, since a
``FeatureArtifacts`` becomes provenance. The freeze is shallow.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FeatureArtifacts(BaseModel):
    """The artifacts one Feature run published, by role.

    The ``gs://`` patterns close the slip ``TrainRunIdentity`` closes, and newly
    validate hand-typed ``--panel-uri`` overrides.
    """

    model_config = ConfigDict(frozen=True)

    panel_uri: str = Field(pattern=r"^gs://")
    calendar_uri: str = Field(pattern=r"^gs://")


class FeatureRunOutputs(BaseModel):
    """At Feature ``run_outputs.json``in run root. Presence is the completion signal"""

    model_config = ConfigDict(frozen=True)

    feature_run_id: str = Field(min_length=1)
    published: FeatureArtifacts
    # Optional: the four properties never asked for it, so it is checked when
    # present and its absence is not an error.
    env: str | None = None
    schema_version: str | None = None
    # Opaque by design: a 7-character hash, an ISO 8601 string with offset, and a
    # statistics mapping today, none of them read here. `Any`, not a declared
    # type, which would validate the container even while ignoring the contents.
    git_hash: Any = None
    completed_at: Any = None
    panel: Any = None
