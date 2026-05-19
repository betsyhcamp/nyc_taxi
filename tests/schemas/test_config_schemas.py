from __future__ import annotations

import pytest
from pydantic import ValidationError

from fcstnyctaxi.schemas.config_schemas import (
    PipelineConfig,
)


@pytest.fixture
def valid_config_dict() -> dict:
    """A complete, valid raw config dict. Each test receives a fresh
    instance that may be mutated.
    """
    return {
        "project_settings": {
            "project_id": "nyc_taxi_ehc",
            "location": "us-central1",
            "env": "dev",
            "bucket_name": "nyc-taxi-ehc--modeling",
        },
        "docker": {"extract_db_to_bucket": "extractdbimagename:sha1234"},
        "extract_db_to_bucket": {
            "sql_filename": "initial_daily_taxi_rides.sql",
            "sql_params": {},
            "gcs_prefix": "dev/initial_datapull",
            "output_filename": "manhattan_daily_zone_pickups.parquet",
        },
    }


def test_valid_dict_constructs_pipeline_config(valid_config_dict: dict) -> None:
    """A complete, well typed dict constructs PipelineConfig and nested models"""
    config = PipelineConfig(**valid_config_dict)

    assert (
        config.project_settings.project_id
        == valid_config_dict["project_settings"]["project_id"]
    )
    assert (
        config.extract_db_to_bucket.sql_filename
        == valid_config_dict["extract_db_to_bucket"]["sql_filename"]
    )
    assert (
        config.extract_db_to_bucket.sql_params
        == valid_config_dict["extract_db_to_bucket"]["sql_params"]
    )


def test_missing_required_field_raises_validation_error(
    valid_config_dict: dict,
) -> None:
    """A dict missing a required field raises ValidationError naming that field."""
    bad_dict = valid_config_dict
    bad_dict["project_settings"].pop("bucket_name")
    with pytest.raises(ValidationError, match="bucket_name"):
        PipelineConfig(**bad_dict)


def test_extra_field_raises_validation_error(valid_config_dict: dict) -> None:
    """An unknown field in a nested config section raises ValidationError
    (extra="forbid").
    """
    extra_dict = valid_config_dict
    extra_dict["project_settings"]["not_a_real_field"] = "value"

    with pytest.raises(ValidationError, match="not_a_real_field"):
        PipelineConfig(**extra_dict)


def test_wrong_type_raises_validation_error(valid_config_dict: dict) -> None:
    """A value of the wrong type for a declared field raises ValidationError."""
    wrong_type_dict = valid_config_dict
    wrong_type_dict["project_settings"]["bucket_name"] = 1

    with pytest.raises(ValidationError, match="bucket_name"):
        PipelineConfig(**wrong_type_dict)


def test_frozen_model_rejects_mutation(valid_config_dict: dict) -> None:
    """Field assignment on a frozen nested model raises ValidationError
    (frozen=True).
    """
    config = PipelineConfig(**valid_config_dict)

    with pytest.raises(ValidationError, match="frozen"):
        config.project_settings.env = "prod"


def test_empty_output_filename_raises_validation_error(valid_config_dict: dict) -> None:
    """Empty string for output_filename violates min_length; raises ValidationError."""

    bad_filename_config = valid_config_dict
    bad_filename_config["extract_db_to_bucket"]["output_filename"] = ""

    with pytest.raises(ValidationError, match="output_filename"):
        PipelineConfig(**bad_filename_config)
