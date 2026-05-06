from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from google.cloud import bigquery
from pytest_mock import MockerFixture

from fcstnyctaxi.core.feature.extract_db_to_bucket_impl import extract_db_to_bucket_impl

PATCH_TARGET_WRITE = (
    "fcstnyctaxi.core.feature.extract_db_to_bucket_impl.write_df_to_gcs_parquet"
)


@pytest.fixture
def mock_write_df_to_gcs_parquet(mocker: MockerFixture) -> Any:
    """Patch write_df_to_gcs_parquet at the module's import binding."""
    mock = mocker.patch(PATCH_TARGET_WRITE)
    mock.return_value = {
        "uri": "gs://BUCKET/PREFIX/RUN_ID/OUTPUT_FILE.parquet",
        "size": 12345,
        "generation": "1",
        "crc32c": "abc",
        "etag": "etag",
        "updated": "2026-01-01",
    }
    return mock


@pytest.fixture
def mock_bq_client_with_df(
    mock_bq_client: Any,
    mock_row_iterator: Any,
    sample_pandas_df: pd.DataFrame,
) -> Any:
    """mock_bq_client wired to return sample_pandas_df and a nonzero row count."""
    mock_row_iterator.total_rows = len(sample_pandas_df)
    mock_row_iterator.to_dataframe.return_value = sample_pandas_df
    return mock_bq_client


def test_happy_path_bq_stats_and_write_meta(
    mock_bq_client_with_df: Any, mock_write_df_to_gcs_parquet: Any
) -> None:
    """Happy path; returns tuple of (BigQueryQueryStats, write_meta)."""
    bq_stats, write_meta = extract_db_to_bucket_impl(
        sql_query="SELECT 1",
        output_gcs_uri="gs://BUCKET/PREFIX/RUN_ID/OUTPUT_FILE.parquet",
        bq_client=mock_bq_client_with_df,
    )

    assert bq_stats.job_id == "test-job-123"
    assert bq_stats.total_rows == 3
    assert write_meta["size"] == 12345


def test_raises_runtime_error_when_query_returns_zero_rows(
    mock_bq_client: Any,
    mock_row_iterator: Any,
    sample_pandas_df: pd.DataFrame,
    mock_write_df_to_gcs_parquet: Any,
) -> None:
    """Zero rows from BigQuery must raise RuntimeError before writing to GCS."""
    mock_row_iterator.total_rows = 0
    mock_row_iterator.to_dataframe.return_value = sample_pandas_df.iloc[0:0]

    with pytest.raises(RuntimeError, match="zero rows"):
        extract_db_to_bucket_impl(
            sql_query="SELECT 1",
            output_gcs_uri="gs://BUCKET/PREFIX/RUN_ID/OUTPUT_FILE.parquet",
            bq_client=mock_bq_client,
        )

    mock_write_df_to_gcs_parquet.assert_not_called()


def test_write_df_to_gcs_parquet_called_with_df_and_uri(
    mock_bq_client_with_df: Any,
    sample_pandas_df: pd.DataFrame,
    mock_write_df_to_gcs_parquet: Any,
) -> None:
    """The writer is called with the queried DataFrame and the caller-supplied URI."""
    extract_db_to_bucket_impl(
        sql_query="SELECT 1",
        output_gcs_uri="gs://BUCKET/PREFIX/RUN_ID/OUTPUT_FILE.parquet",
        bq_client=mock_bq_client_with_df,
    )

    mock_write_df_to_gcs_parquet.assert_called_once()
    df_arg, uri_arg = mock_write_df_to_gcs_parquet.call_args.args
    pd.testing.assert_frame_equal(df_arg, sample_pandas_df)
    assert uri_arg == "gs://BUCKET/PREFIX/RUN_ID/OUTPUT_FILE.parquet"


def test_query_inputs_forwarded_to_bq_client(
    mock_bq_client_with_df: Any, mock_write_df_to_gcs_parquet: Any
) -> None:
    """job_config and SQL query forwarded to query_to_dataframe to bq_client.query."""
    job_config = bigquery.QueryJobConfig(maximum_bytes_billed=10_000_000)

    extract_db_to_bucket_impl(
        sql_query="SELECT 1",
        output_gcs_uri="gs://BUCKET/PREFIX/RUN_ID/OUTPUT_FILE.parquet",
        bq_client=mock_bq_client_with_df,
        job_config=job_config,
    )

    assert mock_bq_client_with_df.query.call_args.kwargs["job_config"] is job_config
    assert mock_bq_client_with_df.query.call_args.args[0] == "SELECT 1"
