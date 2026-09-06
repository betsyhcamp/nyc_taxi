from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from fcstnyctaxi.lib.config.loading import load_pipeline_config


@pytest.mark.parametrize("extension", [".yaml", ".yml"])
def test_valid_yaml_file_returns_pipeline_config(
    tmp_path: Path, extension: str
) -> None:
    """A well formed YAML loads and validates into a PipelineConfig."""

    yaml_path = tmp_path / f"config{extension}"
    yaml_path.write_text(
        """
        project_settings:
          project_id: "p"
          location: "l"
          env: "dev"
          bucket_name: "b"
        docker:
          extract_db_to_bucket: "t"
        vertex:
          pipeline_service_account: "test-acct-name@testproject.iam.gserviceaccount.com"
          pipeline_root: "gs://test-bucket/test-pipeline-root/"
          display_name_prefix: "test-gcs-prefix"
        extract_db_to_bucket:
          sql_filename: "x.sql"
          sql_params: {}
          gcs_prefix: "p"
          output_filename: "y.parquet"
    """
    )
    config = load_pipeline_config(yaml_path)
    assert config.project_settings.bucket_name == "b"


def test_attempt_to_load_nonexistent_yaml_config(tmp_path: Path) -> None:
    """A path to a nonexistent YAML file raises FileNotFoundError."""
    missing_file_path = tmp_path / "nonexistent_file.yaml"

    with pytest.raises(FileNotFoundError):
        load_pipeline_config(missing_file_path)


def test_attempt_to_use_wrong_config_extension_path(tmp_path: Path) -> None:
    """An unsupported file extension raises ValueError."""
    wrong_ext_path = tmp_path / "wrong_ext.json"
    with pytest.raises(ValueError, match="Unsupported config type"):
        load_pipeline_config(wrong_ext_path)


def test_invalid_config_file_structure(tmp_path: Path) -> None:
    """An invalid YAML structure (extra/missing fields) raises ValidationError."""
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        """
        not_a_valid_key:
          not_a_valid_nested_key: "not_a_valid_value"
    """
    )
    with pytest.raises(ValidationError, match="project_settings"):
        load_pipeline_config(yaml_path)
