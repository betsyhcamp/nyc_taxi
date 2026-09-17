from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# An allowlist rather than a drop-list; each new metadata column would otherwise

PANEL_MODELING_COLUMNS: tuple[str, ...] = ("unique_id", "ds", "y")
"""What the panel carries into a model; required and allowed are one list."""

CALENDAR_REQUIRED_COLUMNS: tuple[str, ...] = (
    "ds",
    "fiscal_year_month",
    "origin_month_fraction_elapsed",
    "fiscal_week_of_month",
    "fiscal_month",
    "weeks_in_month",
    "count_workdays",
)
"""Calendar columns with a named consumer, so absence is a failure."""

CALENDAR_ALLOWED_COLUMNS: tuple[str, ...] = CALENDAR_REQUIRED_COLUMNS + (
    "fiscal_year",
    "fiscal_year_week",
)
"""Everything the contract declares. The two extras are allowed but not required:
`exog_features` could name them, and the contract lets them change uncoordinated.
"""


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
