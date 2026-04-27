from __future__ import annotations

from typing import Any

import pandas as pd
import pyarrow as pa
import pytest
from google.api_core.exceptions import DeadlineExceeded, GoogleAPIError

from fcstnyctaxi.lib.dataio import BigQueryQueryStats, query_to_dataframe

#################################################################
# tests for query_to_dataframe()
#################################################################


def test_pandas_path_calls_to_dataframe_not_to_arrow(
    mock_bq_client: Any,
    mock_row_iterator: Any,
    sample_pandas_df: pd.DataFrame,
) -> None:
    """Verify pandas path calls to_dataframe() and never touches to_arrow()."""
    mock_row_iterator.to_dataframe.return_value = sample_pandas_df

    df, stats = query_to_dataframe(
        "SELECT * FROM test",
        client=mock_bq_client,
        dataframe_type="pandas",
    )

    mock_row_iterator.to_dataframe.assert_called_once_with(create_bqstorage_client=True)
    mock_row_iterator.to_arrow.assert_not_called()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["col1", "col2"]
    assert len(df) == 3


def test_polars_path_calls_to_arrow_not_to_dataframe(
    mock_bq_client: Any,
    mock_row_iterator: Any,
    sample_arrow_table: pa.Table,
) -> None:
    """Verify polars path calls to_arrow() and never touches to_dataframe()."""
    import polars as pl

    mock_row_iterator.to_arrow.return_value = sample_arrow_table

    df, stats = query_to_dataframe(
        "SELECT * FROM test",
        client=mock_bq_client,
        dataframe_type="polars",
    )

    mock_row_iterator.to_arrow.assert_called_once_with(create_bqstorage_client=True)
    mock_row_iterator.to_dataframe.assert_not_called()
    assert isinstance(df, pl.DataFrame)
    assert df.columns == ["col1", "col2"]
    assert len(df) == 3


def test_stats_populated_from_job_and_result(
    mock_bq_client: Any,
    mock_query_job: Any,
    mock_row_iterator: Any,
    sample_pandas_df: pd.DataFrame,
) -> None:
    """Verify all stats fields are populated from job and result objects."""
    mock_query_job.job_id = "job-abc-123"
    mock_query_job.total_bytes_processed = 5000
    mock_query_job.total_bytes_billed = 10000
    mock_query_job.cache_hit = True
    mock_row_iterator.total_rows = 42
    mock_row_iterator.to_dataframe.return_value = sample_pandas_df

    _, stats = query_to_dataframe("SELECT 1", client=mock_bq_client)

    assert stats.job_id == "job-abc-123"
    assert stats.total_rows == 42
    assert stats.total_bytes_processed == 5000
    assert stats.total_bytes_billed == 10000
    assert stats.cache_hit is True
    assert stats.elapsed_seconds >= 0
    assert stats.conversion_seconds >= 0


def test_google_api_error_propagates_from_query(
    mock_bq_client: Any,
) -> None:
    """Verify GoogleAPIError from client.query() propagates unchanged."""
    mock_bq_client.query.side_effect = GoogleAPIError("Permission denied")

    with pytest.raises(GoogleAPIError, match="Permission denied"):
        query_to_dataframe("SELECT * FROM table", client=mock_bq_client)


def test_google_api_error_propagates_from_result(
    mock_bq_client: Any,
    mock_query_job: Any,
) -> None:
    """Verify GoogleAPIError from job.result() propagates unchanged."""
    mock_query_job.result.side_effect = DeadlineExceeded("Query timed out")

    with pytest.raises(DeadlineExceeded, match="Query timed out"):
        query_to_dataframe("SELECT * FROM table", client=mock_bq_client)


def test_timeout_passed_to_job_result(
    mock_bq_client: Any,
    mock_query_job: Any,
    mock_row_iterator: Any,
) -> None:
    """Verify timeout parameter is forwarded to job.result()."""
    mock_row_iterator.to_dataframe.return_value = pd.DataFrame()

    query_to_dataframe("SELECT 1", client=mock_bq_client, timeout=42.5)

    mock_query_job.result.assert_called_once_with(timeout=42.5)


def test_invalid_dataframe_type_raises_valueerror(
    mock_bq_client: Any,
) -> None:
    """Verify ValueError raised for unsupported dataframe_type."""
    with pytest.raises(ValueError, match=r"Unsupported dataframe_type"):
        query_to_dataframe(
            "SELECT 1",
            client=mock_bq_client,
            dataframe_type="duckdb",  # type: ignore[arg-type]
        )


def test_use_bqstorage_false_forwarded_to_pandas(
    mock_bq_client: Any,
    mock_row_iterator: Any,
) -> None:
    """Verify use_bqstorage=False is passed to to_dataframe()."""
    mock_row_iterator.to_dataframe.return_value = pd.DataFrame()

    query_to_dataframe(
        "SELECT 1",
        client=mock_bq_client,
        dataframe_type="pandas",
        use_bqstorage=False,
    )

    mock_row_iterator.to_dataframe.assert_called_once_with(
        create_bqstorage_client=False
    )


def test_use_bqstorage_false_forwarded_to_polars(
    mock_bq_client: Any,
    mock_row_iterator: Any,
    sample_arrow_table: pa.Table,
) -> None:
    """Verify use_bqstorage=False is passed to to_arrow()."""
    mock_row_iterator.to_arrow.return_value = sample_arrow_table

    query_to_dataframe(
        "SELECT 1",
        client=mock_bq_client,
        dataframe_type="polars",
        use_bqstorage=False,
    )

    mock_row_iterator.to_arrow.assert_called_once_with(create_bqstorage_client=False)


#################################################################
# BigQueryQueryStats tests
#################################################################


def test_stats_to_dict_excludes_none_by_default() -> None:
    """Verify to_dict() excludes None values by default."""
    stats = BigQueryQueryStats(
        job_id="job-123",
        total_rows=100,
        total_bytes_processed=None,
        total_bytes_billed=None,
        cache_hit=None,
        elapsed_seconds=1.5,
        conversion_seconds=0.5,
    )

    result = stats.to_dict()

    assert result == {
        "job_id": "job-123",
        "total_rows": 100,
        "elapsed_seconds": 1.5,
        "conversion_seconds": 0.5,
    }
    assert "total_bytes_processed" not in result
    assert "cache_hit" not in result


def test_stats_to_dict_includes_none_when_requested() -> None:
    """Verify to_dict(exclude_none=False) includes None values."""
    stats = BigQueryQueryStats(
        job_id="job-123",
        total_rows=None,
        total_bytes_processed=None,
        total_bytes_billed=None,
        cache_hit=None,
        elapsed_seconds=1.0,
        conversion_seconds=0.2,
    )

    result = stats.to_dict(exclude_none=False)

    assert result["total_rows"] is None
    assert result["cache_hit"] is None
    assert len(result) == 7
