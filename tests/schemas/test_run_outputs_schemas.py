import pytest
from pydantic import ValidationError

from fcstnyctaxi.schemas.run_outputs import (
    ADDITIONAL_EXOG_REQUIRED_COLUMNS,
    CALENDAR_ALLOWED_COLUMNS,
    CALENDAR_REQUIRED_COLUMNS,
    JOIN_KEYS,
    PANEL_REQUIRED_COLUMNS,
    FeatureArtifacts,
    FeatureRunOutputs,
    TrainRunOutputs,
)

PANEL_URI = "gs://BUCKET/dev/feature/F1/data_prep/time_series.parquet"
CALENDAR_URI = "gs://BUCKET/dev/feature/F1/data_prep/fiscal_calendar.parquet"
EXOG_URI = "gs://BUCKET/dev/feature/F1/data_prep/exogenous_features.parquet"
MODEL_RESOURCE = "projects/123456789/locations/us-central1/models/fcst-a-lightgbm"
BUNDLE_URI = "gs://BUCKET/dev/train/t1/final_fit/lightgbm/"


def _published(**overrides: object) -> dict:
    """The three URIs a manifest publishes, with any replaced or added."""
    return {
        "panel_uri": PANEL_URI,
        "calendar_uri": CALENDAR_URI,
        "exogenous_uri": EXOG_URI,
        **overrides,
    }


def _payload(**overrides: object) -> dict:
    """The minimum a manifest must carry, with any key replaced or added."""
    payload: dict = {
        "feature_run_id": "F1",
        "published": _published(),
    }
    payload.update(overrides)
    return payload


def test_extra_keys_are_ignored_at_both_levels() -> None:
    """The day the producer adds a field, Training must not break in a step that
    never reads it."""
    outputs = FeatureRunOutputs.model_validate(
        _payload(
            sql_sha256="a" * 64,
            published=_published(
                features_uri="gs://BUCKET/dev/feature/F1/step/features.parquet"
            ),
        )
    )

    assert outputs.published.panel_uri == PANEL_URI


@pytest.mark.parametrize("field", ["panel_uri", "calendar_uri", "exogenous_uri"])
def test_a_non_gcs_uri_is_refused(field: str) -> None:
    """No role may hold a local path; a typo otherwise reaches download_from_gcs."""
    published = _published(**{field: "/tmp/scratch/panel.parquet"})

    with pytest.raises(ValidationError, match=field):
        FeatureRunOutputs.model_validate(_payload(published=published))


def test_a_manifest_without_exogenous_uri_is_refused() -> None:
    """A run published before the artifact, or a renamed key, must fail rather than
    validate."""
    published = _published()
    del published["exogenous_uri"]

    with pytest.raises(ValidationError, match="exogenous_uri"):
        FeatureRunOutputs.model_validate(_payload(published=published))


def test_an_empty_exogenous_uri_is_refused() -> None:
    """Feature's no-file value; accepted only once every consumer can train without
    the file."""
    with pytest.raises(ValidationError, match="exogenous_uri"):
        FeatureRunOutputs.model_validate(
            _payload(published=_published(exogenous_uri=""))
        )


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
    artifacts = FeatureArtifacts(
        panel_uri=PANEL_URI, calendar_uri=CALENDAR_URI, exogenous_uri=EXOG_URI
    )

    with pytest.raises(ValidationError):
        artifacts.panel_uri = "gs://BUCKET/other.parquet"


# ================================================
# TrainRunOutputs
# ================================================


def _train_payload() -> dict:
    """A complete Training record, shaped as register_model will write it."""
    return {
        "train_run_id": "t1",
        "published": {"model_tag": f"{MODEL_RESOURCE}@3", "bundle_uri": BUNDLE_URI},
        "feature_run_id": "F1",
        "env": "dev",
        "schema_version": "0.1.0",
        "git_hash": "a" * 40,
        "completed_at": "2026-09-20T22:26:03.114927+00:00",
        "training_data": {"train_end_ds": "2025-10-19", "n_series": 3, "n_obs": 60},
    }


def test_train_run_outputs_reads_back_what_its_writer_serializes() -> None:
    """Inference reads what register_model writes, with the writer's own flags."""
    outputs = TrainRunOutputs.model_validate(_train_payload())

    written = outputs.model_dump_json(indent=2, exclude_none=True)

    assert TrainRunOutputs.model_validate_json(written) == outputs


@pytest.mark.parametrize("level", [None, "published", "training_data"])
def test_train_run_outputs_refuses_an_extra_key_at_every_level(
    level: str | None,
) -> None:
    """Unlike Feature's record, this one is ours, so an extra key is a typo."""
    payload = _train_payload()
    (payload if level is None else payload[level])["unexpected"] = "x"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TrainRunOutputs.model_validate(payload)


@pytest.mark.parametrize(
    "field", ["train_run_id", "feature_run_id", "env", "schema_version", "git_hash"]
)
def test_an_empty_string_is_refused(field: str) -> None:
    """An empty value names nothing, so a record carrying one is as useless as null."""
    payload = _train_payload()
    payload[field] = ""

    with pytest.raises(ValidationError, match=field):
        TrainRunOutputs.model_validate(payload)


def test_a_mount_path_is_refused_as_the_bundle_uri() -> None:
    """The wrapper holds the bundle's gcsfuse path beside its URI; this is the slip."""
    payload = _train_payload()
    payload["published"]["bundle_uri"] = "/gcs/BUCKET/dev/train/t1/final_fit/lightgbm"

    with pytest.raises(ValidationError, match="bundle_uri"):
        TrainRunOutputs.model_validate(payload)


@pytest.mark.parametrize(
    "model_tag",
    [MODEL_RESOURCE, "fcst-a-lightgbm@3"],
    ids=["unversioned", "unqualified"],
)
def test_a_model_tag_that_pins_no_version_or_no_project_is_refused(
    model_tag: str,
) -> None:
    """Unversioned resolves to `default`; unqualified needs aiplatform.init to find."""
    payload = _train_payload()
    payload["published"]["model_tag"] = model_tag

    with pytest.raises(ValidationError, match="model_tag"):
        TrainRunOutputs.model_validate(payload)


def test_an_alias_is_a_legal_model_tag() -> None:
    """The approval gate will write `@champion`, and must not need a schema change."""
    payload = _train_payload()
    payload["published"]["model_tag"] = f"{MODEL_RESOURCE}@champion"

    outputs = TrainRunOutputs.model_validate(payload)

    assert outputs.published.model_tag.endswith("@champion")


def test_a_completed_at_with_no_offset_is_refused() -> None:
    """A naive timestamp's zone is a guess; the runners log in UTC."""
    payload = _train_payload()
    payload["completed_at"] = "2026-09-20T22:26:03.114927"

    with pytest.raises(ValidationError, match="completed_at"):
        TrainRunOutputs.model_validate(payload)


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


def test_the_join_keys_are_columns_the_additional_exog_file_carries() -> None:
    """A key the file does not carry makes its join unwritable."""
    assert set(JOIN_KEYS) <= set(ADDITIONAL_EXOG_REQUIRED_COLUMNS)


def test_exactly_one_join_key_is_also_a_calendar_column() -> None:
    """Why exog_features needs two checks: `ds` passes an unknown-column test."""
    assert set(JOIN_KEYS) & set(CALENDAR_ALLOWED_COLUMNS) == {"ds"}
