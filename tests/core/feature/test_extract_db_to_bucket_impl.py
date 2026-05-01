from __future__ import annotations

from pathlib import Path
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
def sql_file(tmp_path: Path) -> Path:
    """A SQL file with no Jinja placeholders."""
    path = tmp_path / "query.sql"
    path.write_text("SELECT pickup_date FROM `proj.dataset.table`")
    return path


@pytest.fixture
def sql_file_with_params(tmp_path: Path) -> Path:
    """A SQL file with Jinja placeholders."""
    path = tmp_path / "query_with_params.sql"
    path.write_text(
        "SELECT * FROM `proj.{{ schema }}.{{ table }}` WHERE year = {{ year }}"
    )
    return path


@pytest.fixture
def mock_write_df_to_gcs_parquet(mocker: MockerFixture) -> Any:
    """Patch write_df_to_gcs_parquet at the module's import binding."""
    mock = mocker.patch(PATCH_TARGET_WRITE)
    mock.return_value = {
        "uri": "gs://test-bucket/test-prefix/manhattan_daily_zone_pickups.parquet",
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


def test_happy_path_no_sql_params_returns_uri_stats_and_write_meta(
    sql_file: Path,
    mock_bq_client_with_df: Any,
    mock_write_df_to_gcs_parquet: Any,
) -> None:
    """With no params, returns tuple of (gcs_uri, BigQueryQueryStats, write_meta)."""
    gcs_uri, bq_stats, write_meta = extract_db_to_bucket_impl(
        sql_path=sql_file,
        sql_params={},
        bq_client=mock_bq_client_with_df,
        bucket_name="test-bucket",
        gcs_prefix="test-prefix",
    )

    assert gcs_uri == (
        "gs://test-bucket/test-prefix/manhattan_daily_zone_pickups.parquet"
    )
    assert bq_stats.job_id == "test-job-123"
    assert bq_stats.total_rows == 3
    assert write_meta["size"] == 12345


def test_skips_rendering_when_sql_params_empty(
    sql_file: Path,
    mock_bq_client_with_df: Any,
    mock_write_df_to_gcs_parquet: Any,
    mocker: MockerFixture,
) -> None:
    """Empty sql_params dict must not invoke render_sql_template."""
    render_spy = mocker.patch(
        "fcstnyctaxi.core.feature.extract_db_to_bucket_impl.render_sql_template"
    )

    extract_db_to_bucket_impl(
        sql_path=sql_file,
        sql_params={},
        bq_client=mock_bq_client_with_df,
        bucket_name="test-bucket",
        gcs_prefix="test-prefix",
    )

    render_spy.assert_not_called()


def test_rendered_sql_is_passed_to_bq_when_params_provided(
    sql_file_with_params: Path,
    mock_bq_client_with_df: Any,
    mock_write_df_to_gcs_parquet: Any,
) -> None:
    """When sql_params are provided, the rendered SQL is what BQ receives."""
    extract_db_to_bucket_impl(
        sql_path=sql_file_with_params,
        sql_params={"schema": "sales", "table": "events", "year": 2026},
        bq_client=mock_bq_client_with_df,
        bucket_name="test-bucket",
        gcs_prefix="test-prefix",
    )

    sql_arg = mock_bq_client_with_df.query.call_args.args[0]
    assert "{{ schema }}" not in sql_arg
    assert "{{ table }}" not in sql_arg
    assert "{{ year }}" not in sql_arg
    assert "proj.sales.events" in sql_arg
    assert "year = 2026" in sql_arg


def test_raises_runtime_error_when_query_returns_zero_rows(
    sql_file: Path,
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
            sql_path=sql_file,
            sql_params={},
            bq_client=mock_bq_client,
            bucket_name="test-bucket",
            gcs_prefix="test-prefix",
        )

    mock_write_df_to_gcs_parquet.assert_not_called()


def test_gcs_uri_constructed_from_bucket_and_prefix(
    sql_file: Path,
    mock_bq_client_with_df: Any,
    mock_write_df_to_gcs_parquet: Any,
) -> None:
    """GCS URI must compose as gs://<bucket>/<prefix>/<static_filename>."""
    gcs_uri, _, _ = extract_db_to_bucket_impl(
        sql_path=sql_file,
        sql_params={},
        bq_client=mock_bq_client_with_df,
        bucket_name="my-bucket",
        gcs_prefix="dev/snapshots/daily-zone",
    )

    assert gcs_uri == (
        "gs://my-bucket/dev/snapshots/daily-zone/manhattan_daily_zone_pickups.parquet"
    )


def test_write_df_to_gcs_parquet_called_with_df_and_uri(
    sql_file: Path,
    mock_bq_client_with_df: Any,
    sample_pandas_df: pd.DataFrame,
    mock_write_df_to_gcs_parquet: Any,
) -> None:
    """The queried DataFrame and constructed URI are forwarded to the writer."""
    extract_db_to_bucket_impl(
        sql_path=sql_file,
        sql_params={},
        bq_client=mock_bq_client_with_df,
        bucket_name="test-bucket",
        gcs_prefix="test-prefix",
    )

    mock_write_df_to_gcs_parquet.assert_called_once()
    df_arg, uri_arg = mock_write_df_to_gcs_parquet.call_args.args
    pd.testing.assert_frame_equal(df_arg, sample_pandas_df)
    assert uri_arg == (
        "gs://test-bucket/test-prefix/manhattan_daily_zone_pickups.parquet"
    )


def test_job_config_forwarded_to_bq_client(
    sql_file: Path,
    mock_bq_client_with_df: Any,
    mock_write_df_to_gcs_parquet: Any,
) -> None:
    """job_config must be forwarded through query_to_dataframe to bq_client.query."""
    job_config = bigquery.QueryJobConfig(maximum_bytes_billed=10_000_000)

    extract_db_to_bucket_impl(
        sql_path=sql_file,
        sql_params={},
        bq_client=mock_bq_client_with_df,
        bucket_name="test-bucket",
        gcs_prefix="test-prefix",
        job_config=job_config,
    )

    assert mock_bq_client_with_df.query.call_args.kwargs["job_config"] is job_config


def test_missing_sql_file_raises_file_not_found_error(
    tmp_path: Path,
    mock_bq_client_with_df: Any,
    mock_write_df_to_gcs_parquet: Any,
) -> None:
    """A non-existent SQL file propagates FileNotFoundError from read_sql."""
    missing = tmp_path / "does_not_exist.sql"

    with pytest.raises(FileNotFoundError):
        extract_db_to_bucket_impl(
            sql_path=missing,
            sql_params={},
            bq_client=mock_bq_client_with_df,
            bucket_name="test-bucket",
            gcs_prefix="test-prefix",
        )

    mock_bq_client_with_df.query.assert_not_called()
    mock_write_df_to_gcs_parquet.assert_not_called()
