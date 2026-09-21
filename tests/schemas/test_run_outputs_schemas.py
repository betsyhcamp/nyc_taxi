import pytest
from pydantic import ValidationError

from fcstnyctaxi.schemas.run_outputs import (
    CALENDAR_ALLOWED_COLUMNS,
    CALENDAR_REQUIRED_COLUMNS,
    JOIN_KEYS,
    PANEL_REQUIRED_COLUMNS,
    FeatureArtifacts,
    FeatureRunOutputs,
)

PANEL_URI = "gs://BUCKET/dev/feature/F1/data_prep/time_series.parquet"
CALENDAR_URI = "gs://BUCKET/dev/feature/F1/data_prep/fiscal_calendar.parquet"


def _payload(**overrides: object) -> dict:
    """The minimum a manifest must carry, with any key replaced or added."""
    payload: dict = {
        "feature_run_id": "F1",
        "published": {"panel_uri": PANEL_URI, "calendar_uri": CALENDAR_URI},
    }
    payload.update(overrides)
    return payload


def test_extra_keys_are_ignored_at_both_levels() -> None:
    """The day the producer adds a field, Training must not break in a step that
    never reads it."""
    outputs = FeatureRunOutputs.model_validate(
        _payload(
            sql_sha256="a" * 64,
            published={
                "panel_uri": PANEL_URI,
                "calendar_uri": CALENDAR_URI,
                "features_uri": "gs://BUCKET/dev/feature/F1/step/features.parquet",
            },
        )
    )

    assert outputs.published.panel_uri == PANEL_URI


@pytest.mark.parametrize("field", ["panel_uri", "calendar_uri"])
def test_a_non_gcs_uri_is_refused(field: str) -> None:
    """Neither role may hold a local path; a typo otherwise reaches
    download_from_gcs."""
    published = {"panel_uri": PANEL_URI, "calendar_uri": CALENDAR_URI}
    published[field] = "/tmp/scratch/panel.parquet"

    with pytest.raises(ValidationError, match=field):
        FeatureRunOutputs.model_validate(_payload(published=published))


def test_env_and_schema_version_are_both_optional() -> None:
    """The shipped producer file carries no schema_version, and the four properties
    never asked for env."""
    outputs = FeatureRunOutputs.model_validate(_payload())

    assert outputs.env is None
    assert outputs.schema_version is None


def test_opaque_fields_accept_a_shape_no_consumer_reads() -> None:
    """Typed even as a container, a restructured block would fail a consumer that
    never opens it."""
    outputs = FeatureRunOutputs.model_validate(
        _payload(panel=[1, 2], git_hash=12345, completed_at={"iso": "2026-09-14"})
    )

    assert outputs.panel == [1, 2]


def test_feature_artifacts_refuses_rebinding() -> None:
    """A resolved URI altered between resolution and the parameter_values carrying
    it into the job would be provenance that lies."""
    artifacts = FeatureArtifacts(panel_uri=PANEL_URI, calendar_uri=CALENDAR_URI)

    with pytest.raises(ValidationError):
        artifacts.panel_uri = "gs://BUCKET/other.parquet"


# ================================================
# Column allowlists
# ================================================


def test_every_required_calendar_column_is_also_allowed() -> None:
    """A required column outside the allowlist would be demanded and then dropped.

    The two tuples are separate facts, one from consumption and one from the
    contract, so nothing but this stops them drifting into that contradiction.
    """
    assert set(CALENDAR_REQUIRED_COLUMNS) <= set(CALENDAR_ALLOWED_COLUMNS)


def test_the_calendar_allows_more_than_it_requires() -> None:
    """Collapsing the two tuples into one is the conformance check this project avoids.

    If they ever became equal, a column the contract calls optional would take the
    pipeline down when Feature changed it without coordinating.
    """
    assert set(CALENDAR_ALLOWED_COLUMNS) > set(CALENDAR_REQUIRED_COLUMNS)


# ================================================
# JOIN_KEYS
# ================================================


def test_the_join_keys_are_columns_the_panel_carries() -> None:
    """A key the panel does not carry makes the merge unwritable."""
    assert set(JOIN_KEYS) <= set(PANEL_REQUIRED_COLUMNS)


def test_exactly_one_join_key_is_also_a_calendar_column() -> None:
    """Why exog_features needs two checks: `ds` passes an unknown-column test."""
    assert set(JOIN_KEYS) & set(CALENDAR_ALLOWED_COLUMNS) == {"ds"}
