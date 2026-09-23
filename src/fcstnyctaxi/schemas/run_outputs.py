"""Each slice's `run_outputs.json`, Feature's for Training and Training's for
Inference, the shared pointer holding one record per slice, and the columns
Feature promises.

Training's records forbid extras, since this repo writes and reads them; Feature's
and the pointer's don't, since a third party's added field must not break a reader
that ignores it. All are frozen, as provenance.

The column tuples are an allowlist, not a validation: a new column on a
consultant-owned artifact cannot become a model input. Feature owns shape.
"""

from datetime import date
from typing import Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

PANEL_REQUIRED_COLUMNS: tuple[str, ...] = ("unique_id", "ds", "y")
"""Everything the panel must contain; required and allowed are one list."""

JOIN_KEYS: tuple[str, ...] = ("unique_id", "ds")
"""The key an assembled exogenous frame carries; `exog_features` never names these."""

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

ADDITIONAL_EXOG_REQUIRED_COLUMNS: tuple[str, ...] = (
    "unique_id",
    "ds",
    "holiday_days_in_week",
    "week_sin",
    "week_cos",
)
"""Everything the exogenous features artifact must contain; required and allowed are
one list. The file exists to carry the three features, so each is required."""


class FeatureArtifacts(BaseModel):
    """The artifacts one Feature run published, by role.

    The ``gs://`` patterns close the slip ``TrainRunIdentity`` closes, and newly
    validate hand-typed ``--panel-uri`` overrides.
    """

    model_config = ConfigDict(frozen=True)

    panel_uri: str = Field(pattern=r"^gs://")
    calendar_uri: str = Field(pattern=r"^gs://")
    # Feature's key. Everything Training owns calls this artifact additional_exog.
    exogenous_uri: str = Field(pattern=r"^gs://")


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


class RegisteredModel(BaseModel):
    """The registered version, and the bundle it was uploaded from.

    ``model_tag`` is ``Model.versioned_resource_name`` verbatim: ``resource_name``
    carries no version and would resolve to whatever ``default`` points at. The slot
    after ``@`` takes a version or an alias, so ``@champion`` fits unchanged.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_tag: str = Field(
        pattern=r"^projects/[^/]+/locations/[^/]+/models/[^/@]+@[^/@]+$"
    )
    bundle_uri: str = Field(pattern=r"^gs://")


class TrainingData(BaseModel):
    """Copied verbatim from ``final_fit_manifest.json``'s block of the same name."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    train_end_ds: date
    n_series: int
    n_obs: int


class TrainRunOutputs(BaseModel):
    """At Training's run root. Presence marks the pipeline finished, and
    ``published.model_tag`` is what Inference passes to ``aiplatform.Model()``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    train_run_id: str = Field(min_length=1)
    published: RegisteredModel
    feature_run_id: str = Field(min_length=1)
    env: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    git_hash: str = Field(min_length=1)
    completed_at: AwareDatetime
    training_data: TrainingData


class LatestRunPointer(BaseModel):
    """At the environment root, ``_latest.json``: each slice's newest run record.

    A key holds that slice's whole ``run_outputs.json`` document, and no value is
    validated: a sibling's record is not Train's to reject, and checking ``train``
    would let the record being replaced block its replacement. The one check left
    is pydantic's own, that the document is an object.
    """

    model_config = ConfigDict(frozen=True)

    feature: Any = None
    train: Any = None
    inference: Any = None
