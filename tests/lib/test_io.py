from __future__ import annotations

from pathlib import Path

import pytest

from fcstnyctaxi.lib.io import build_run_scoped_uri, prepare_sql

EXPECTED_HASH = "e004ebd5b5532a4b85984a62f8ad48a81aa3460c1ca07701f386135d72cdecf5"


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
    path.write_text("SELECT * FROM `proj.schema.table` WHERE year = {{ year }}")
    return path


# ================================================
# prepare_sql tests
# ================================================


def test_prep_sql_with_param_rendering(sql_file_with_params: Path) -> None:
    """Test that prepare_sql will render sql with parameters"""
    sql_path = sql_file_with_params
    params = {"year": 1980}

    result = prepare_sql(sql_path=sql_path, sql_params=params)
    assert result.sql_text == "SELECT * FROM `proj.schema.table` WHERE year = 1980"


def test_prep_sql_raises_on_extra_param(sql_file: Path) -> None:
    "Test that prepare_sql raises ValueError if extra params supplied not in template."
    sql_path = sql_file
    params = {"foo": 1}

    with pytest.raises(ValueError, match="unused"):
        prepare_sql(sql_path=sql_path, sql_params=params)


def test_prep_sql_raises_on_missing_param(sql_file_with_params: Path) -> None:
    "Test that prepare_sql raises ValueError if param in template missing from params"
    sql_path = sql_file_with_params
    params: dict = {}

    with pytest.raises(ValueError, match="missing"):
        prepare_sql(sql_path=sql_path, sql_params=params)


def test_prep_sql_raises_filenotfound_on_missing_file(tmp_path: Path) -> None:
    """Test that nonexistant SQL file raises FileNotFoundError"""

    missing_file = tmp_path / "nonexistent_file.sql"

    with pytest.raises(FileNotFoundError):
        prepare_sql(sql_path=missing_file, sql_params={})


def test_prep_sql_hash_deterministic(tmp_path: Path) -> None:
    """Test that prepare_sql returns deterministic SHA-256 hashes."""
    sql_path = tmp_path / "q.sql"
    sql_path.write_text("SELECT 1")
    result = prepare_sql(sql_path=sql_path, sql_params={})
    assert result.sha256 == EXPECTED_HASH

    result2 = prepare_sql(sql_path=sql_path, sql_params={})
    assert result.sha256 == result2.sha256


# ================================================
# build_run_scoped_uri tests
# ================================================


def test_build_run_scoped_id_constructs_expected_string() -> None:
    bucket = "BUCKET"
    prefix = "PREFIX"
    run_id = "RUNID"
    filename = "FILE.parquet"
    uri = build_run_scoped_uri(
        bucket=bucket, prefix=prefix, run_id=run_id, filename=filename
    )
    assert uri == "gs://BUCKET/PREFIX/RUNID/FILE.parquet"
