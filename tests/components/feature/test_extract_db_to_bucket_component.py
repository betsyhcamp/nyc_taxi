import re
from typing import Any

import pytest
from kfp.dsl import Dataset
from pytest_mock import MockerFixture
from tsbricks.blocks.dataio import BigQueryQueryStats

from fcstnyctaxi.components.feature.extract_db_to_bucket_component import (
    extract_db_to_bucket,
)

PATCH_TARGET_IMPL = (
    "fcstnyctaxi.core.feature.extract_db_to_bucket_impl.extract_db_to_bucket_impl"
)
PATCH_TARGET_WRITE_TEXT = "fcstnyctaxi.lib.io.write_text_to_gcs"
PATCH_TARGET_BQ_CLIENT = "google.cloud.bigquery.Client"


@pytest.fixture
def mock_extract_db_impl(mocker: MockerFixture) -> Any:
    """Mock extract_db_to_bucket_impl returning a successful
    (BigQueryQueryStats, write_meta) tuple."""
    mock = mocker.patch(PATCH_TARGET_IMPL)
    mock.return_value = (
        BigQueryQueryStats(
            job_id="test-job-123",
            total_rows=100,
            total_bytes_processed=1024,
            total_bytes_billed=2048,
            cache_hit=False,
            elapsed_seconds=0.5,
            conversion_seconds=0.1,
        ),
        {"size": 12345},
    )
    return mock


@pytest.fixture
def mock_write_text_to_gcs(mocker: MockerFixture) -> Any:
    """Mock write_text_to_gcs (sidecar writer); prevents real GCS writes during
    tests."""
    mock = mocker.patch(PATCH_TARGET_WRITE_TEXT)
    return mock


@pytest.fixture
def mock_bq_client_class(mocker: MockerFixture) -> Any:
    """Mock the bigquery.Client class so its constructor returns a benign mock
    instance."""
    mock = mocker.patch(PATCH_TARGET_BQ_CLIENT)
    return mock


@pytest.fixture
def valid_component_kwargs() -> dict[str, str]:
    """Valid dict of component kwargs for invoking the wrapper; fresh per test."""
    return {
        "project_id": "test-proj",
        "bq_location": "US",
        "sql_query": "SELECT 1",
        "sql_sha256": "deadbeef" * 8,  # any 64-hex string
        "output_gcs_uri": "gs://BUCKET/PREFIX/RUN_ID/snapshot.parquet",
        "sql_sidecar_gcs_uri": "gs://BUCKET/PREFIX/RUN_ID/query.sql",
        "run_id": "RUN_ID",
    }


def test_invokes_impl_with_passed_inputs(
    mock_extract_db_impl: Any,
    mock_write_text_to_gcs: Any,
    mock_bq_client_class: Any,
    valid_component_kwargs: dict[str, str],
) -> None:
    """Test that wrapper invokes extract_db_to_bucket_impl with the passed sql_query,
    output_gcs_uri, and constructed bq_client."""

    snapshot = Dataset(name="snapshot", uri="")

    extract_db_to_bucket.python_func(**valid_component_kwargs, snapshot=snapshot)

    mock_extract_db_impl.assert_called_once()

    assert mock_extract_db_impl.call_args.kwargs["sql_query"] == "SELECT 1"
    assert (
        mock_extract_db_impl.call_args.kwargs["output_gcs_uri"]
        == "gs://BUCKET/PREFIX/RUN_ID/snapshot.parquet"
    )
    assert (
        mock_extract_db_impl.call_args.kwargs["bq_client"]
        is mock_bq_client_class.return_value
    )


def test_writes_sql_sidecar_with_passed_inputs(
    mock_extract_db_impl: Any,
    mock_write_text_to_gcs: Any,
    mock_bq_client_class: Any,
    valid_component_kwargs: dict[str, str],
) -> None:
    """Test that wrapper writes the SQL sidecar via write_text_to_gcs with the passed
    text and gcs_uri."""
    snapshot = Dataset(name="snapshot", uri="")

    extract_db_to_bucket.python_func(**valid_component_kwargs, snapshot=snapshot)

    mock_write_text_to_gcs.assert_called_once()

    assert mock_write_text_to_gcs.call_args.kwargs["text"] == "SELECT 1"
    assert (
        mock_write_text_to_gcs.call_args.kwargs["gcs_uri"]
        == "gs://BUCKET/PREFIX/RUN_ID/query.sql"
    )


def test_sets_snapshot_uri_to_output_gcs_uri(
    mock_extract_db_impl: Any,
    mock_write_text_to_gcs: Any,
    mock_bq_client_class: Any,
    valid_component_kwargs: dict[str, str],
) -> None:
    """Test that wrapper assigns snapshot.uri to the caller-provided output_gcs_uri."""
    snapshot = Dataset(name="snapshot", uri="")
    assert snapshot.uri == ""  # assert baseline value of uri

    extract_db_to_bucket.python_func(**valid_component_kwargs, snapshot=snapshot)

    # pulled from fixture rather than hardcoded — test asserts the passthrough,
    # not a specific URI value.
    assert snapshot.uri == valid_component_kwargs["output_gcs_uri"]


def test_populates_metadata_with_expected_keys(
    mock_extract_db_impl: Any,
    mock_write_text_to_gcs: Any,
    mock_bq_client_class: Any,
    valid_component_kwargs: dict[str, str],
) -> None:
    """Test that wrapper populates snapshot.metadata with all 9 metadata keys
    at their expected values."""
    snapshot = Dataset(name="snapshot", uri="")

    extract_db_to_bucket.python_func(**valid_component_kwargs, snapshot=snapshot)

    assert snapshot.metadata.keys() == {
        "row_count",
        "bq_query_job_id",
        "file_size_bytes",
        "run_id",
        "gcs_uri",
        "extracted_at",
        "sql_sha256",
        "sql_sidecar_gcs_uri",
        "git_sha",
    }

    assert snapshot.metadata["row_count"] == 100
    assert snapshot.metadata["bq_query_job_id"] == "test-job-123"
    assert snapshot.metadata["file_size_bytes"] == 12345
    assert snapshot.metadata["run_id"] == "RUN_ID"
    assert snapshot.metadata["gcs_uri"] == "gs://BUCKET/PREFIX/RUN_ID/snapshot.parquet"
    assert re.match(
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$",
        snapshot.metadata["extracted_at"],
    )
    assert snapshot.metadata["sql_sha256"] == ("deadbeef" * 8)
    assert (
        snapshot.metadata["sql_sidecar_gcs_uri"]
        == "gs://BUCKET/PREFIX/RUN_ID/query.sql"
    )
    assert snapshot.metadata["git_sha"] is None


def test_propagates_impl_errors_and_skips_sidecar(
    mock_extract_db_impl: Any,
    mock_write_text_to_gcs: Any,
    mock_bq_client_class: Any,
    valid_component_kwargs: dict[str, str],
) -> None:
    """Test that wrapper propagates core/impl function errors and skips the sidecar
    write on failure."""
    snapshot = Dataset(name="snapshot", uri="")

    mock_extract_db_impl.side_effect = RuntimeError("zero rows")

    with pytest.raises(RuntimeError, match="zero rows"):
        extract_db_to_bucket.python_func(**valid_component_kwargs, snapshot=snapshot)

    mock_extract_db_impl.assert_called_once()
    mock_write_text_to_gcs.assert_not_called()
