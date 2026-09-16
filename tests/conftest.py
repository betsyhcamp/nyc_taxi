from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pytest
from fsspec.implementations.local import LocalFileSystem
from google.cloud import bigquery
from google.cloud.bigquery.job import QueryJob
from google.cloud.bigquery.table import RowIterator
from pytest_mock import MockerFixture


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


@pytest.fixture
def fake_gcs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Resolve gs:// URIs onto a LocalFileSystem under tmp_path; return its root.

    Runs the real fsspec calls in CI with no credentials. Two details are
    load-bearing: auto_mkdir=True is GCS's implicit-parents behaviour, and the path
    is built by string substitution because _strip_protocol strips the trailing "/"
    that marks a prefix rather than an object name.
    """
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    fs = LocalFileSystem(auto_mkdir=True)

    def _fake_url_to_fs(url: str, **kwargs: object) -> tuple[LocalFileSystem, str]:
        return fs, url.replace("gs://", f"{remote_root}/", 1)

    monkeypatch.setattr("fsspec.url_to_fs", _fake_url_to_fs)
    return remote_root
