from __future__ import annotations

import pytest
from pydantic import ValidationError

from fcstnyctaxi.schemas.config.environment import EnvironmentConfig


@pytest.fixture
def valid_environment_dict() -> dict:
    """A complete, valid EnvironmentConfig dict. Each test gets a fresh copy."""
    return {
        "gcp": {
            "project_id": "test-project",
            "location": "us-central1",
            "bucket_name": "test-bucket",
        },
        "vertex": {"pipeline_root": "gs://test-bucket/vertex-pipeline-roots/"},
        "artifact_registry": {
            "project_id": "test-registry-project",
            "location": "us-east1",
            "images": {
                "feature": {"repository": "repo-f", "image": "img-f"},
                "train": {"repository": "repo-t", "image": "img-t"},
                "inference": {"repository": "repo-i", "image": "img-i"},
            },
        },
    }


def test_valid_dict_constructs_environment_config(valid_environment_dict: dict) -> None:
    """A complete dict constructs EnvironmentConfig and every nested model."""
    config = EnvironmentConfig(**valid_environment_dict)

    assert config.gcp.bucket_name == "test-bucket"
    assert config.vertex.pipeline_root == "gs://test-bucket/vertex-pipeline-roots/"
    assert config.artifact_registry.images.train.repository == "repo-t"


def test_artifact_registry_is_independent_of_gcp(valid_environment_dict: dict) -> None:
    """The registry carries its own project and location, not gcp.*'s.

    A shared registry commonly lives in a separate project, and registry region
    diverges from compute region more often than project does.
    """
    config = EnvironmentConfig(**valid_environment_dict)

    assert config.artifact_registry.project_id != config.gcp.project_id
    assert config.artifact_registry.location != config.gcp.location


def test_env_field_is_rejected(valid_environment_dict: dict) -> None:
    """An `env` key is an error: the file path is the binding, not a field.

    Declaring env inside would be a second source of a derived fact — copy
    dev.yaml to prod.yaml, forget the field, and a file at the prod path
    declares itself dev.
    """
    valid_environment_dict["env"] = "dev"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EnvironmentConfig(**valid_environment_dict)


def test_service_accounts_block_is_rejected(valid_environment_dict: dict) -> None:
    """A service_accounts block is an error for PR 0a.

    Deleted rather than made optional, pending an infosec determination.
    Accounts arrive via FCST_{SLICE}_SERVICE_ACCOUNT at submit time.
    """
    valid_environment_dict["service_accounts"] = {
        "feature": "f@p.iam.gserviceaccount.com",
        "train": "t@p.iam.gserviceaccount.com",
        "inference": "i@p.iam.gserviceaccount.com",
    }

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EnvironmentConfig(**valid_environment_dict)


def test_non_gcs_pipeline_root_raises(valid_environment_dict: dict) -> None:
    """pipeline_root must be a gs:// URI."""
    valid_environment_dict["vertex"]["pipeline_root"] = "s3://test-bucket/roots/"

    with pytest.raises(ValidationError, match="pipeline_root"):
        EnvironmentConfig(**valid_environment_dict)


def test_missing_image_slice_raises(valid_environment_dict: dict) -> None:
    """A missing slice is a validation error here, not a KeyError at submit time.

    This is why SliceImages is a named class rather than dict[str, ImageRef].
    """
    del valid_environment_dict["artifact_registry"]["images"]["inference"]

    with pytest.raises(ValidationError, match="inference"):
        EnvironmentConfig(**valid_environment_dict)


def test_image_ref_rejects_a_tag(valid_environment_dict: dict) -> None:
    """ImageRef carries repository and image only — no tag, no digest.

    The tag is the git hash, resolved at build and submit time.
    """
    valid_environment_dict["artifact_registry"]["images"]["train"]["tag"] = "abc1234"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EnvironmentConfig(**valid_environment_dict)


def test_environment_config_is_frozen(valid_environment_dict: dict) -> None:
    """Composed configuration is immutable once validated."""
    config = EnvironmentConfig(**valid_environment_dict)

    with pytest.raises(ValidationError):
        config.gcp.bucket_name = "other-bucket"  # type: ignore[misc]
