from __future__ import annotations

from pathlib import Path

import pytest

from fcstnyctaxi.lib.io import prepare_sql


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


def test_prep_sql_with_param_rendering(sql_file_with_params: Path) -> None:
    """Test that prepare_sql will render sql with parameters"""
    sql_path = sql_file_with_params
    params = {"year": 1980}

    result = prepare_sql(sql_path=sql_path, sql_params=params)
    assert result.sql_text == "SELECT * FROM `proj.schema.table` WHERE year = 1980"


def test_prep_sql_raises_on_extra_param(sql_file: Path) -> None:
    "Test that prepare_sql raises a ValueError if extra params supplied."
    sql_path = sql_file
    params = {"foo": 1}

    with pytest.raises(ValueError, match="unused"):
        prepare_sql(sql_path=sql_path, sql_params=params)
