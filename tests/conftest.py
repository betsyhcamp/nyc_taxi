from __future__ import annotations

from typing import Any
import pandas as pd
import pyarrow as pa
import pytest
from pytest_mock import MockerFixture

from google.cloud import bigquery
from google.cloud.bigquery.job import QueryJob
from google.cloud.bigquery.table import RowIterator


@pytest.fixture
def mock_row_iterator(mocker: MockerFixture) -> Any:
    """Create mock RowIterator"""
    result = mocker.MagicMock(spec=RowIterator)
    result.total_rows = 100
    return result


@pytest.fixture
def mock_query_job(mocker: MockerFixture, mock_row_iterator: Any) -> Any:
    """
    Create mock QueryJob, wired to row iterator. Using `Any`
    to avoid importing unittest.mock for return type
    """
    job = mocker.MagicMock(spec=QueryJob)
    job.job_id = "test-job-123"
    job.total_bytes_processed = 1024
    job.total_bytes_billed = 2048
    job.cache_hit = False
    job.result.return_value = mock_row_iterator
    return job


@pytest.fixture
def mock_bq_client(mocker: MockerFixture, mock_query_job: Any) -> Any:
    """
    Create mock BigQuery client, wired to job. Using `Any`
    to avoid importing unittest.mock for return type
    """
    client = mocker.MagicMock(spec=bigquery.Client)
    client.query.return_value = mock_query_job
    return client


@pytest.fixture
def sample_pandas_df() -> pd.DataFrame:
    """Sample Pandas DataFrame for testing."""
    return pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})


@pytest.fixture
def sample_arrow_table() -> pa.Table:
    """Sample Arrow table for testing."""
    return pa.table({"col1": [1, 2, 4], "col2": ["a", "b", "c"]})
