from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import duckdb

from local import load_local_config


@dataclass(frozen=True)
class YearMonth:
    year: int
    month: int

    @property
    def as_file_stem(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"


def _parse_year_month(*, value: str) -> YearMonth:
    parts = value.split(sep="-")
    if len(parts) != 2:
        raise ValueError(f"Invalid year-month format: {value}. Expected YYYY-MM.")
    return YearMonth(year=int(parts[0]), month=int(parts[1]))


def _last_three_full_months(*, today: date) -> list[YearMonth]:
    months: list[YearMonth] = []
    for i in range(1, 4):
        target_date = today - timedelta(days=31 * i)
        months.append(YearMonth(year=target_date.year, month=target_date.month))
    return months


def _range_year_months(
    *, start: YearMonth, end_inclusive: YearMonth
) -> list[YearMonth]:
    months: list[YearMonth] = []
    year = start.year
    month = start.month
    while (year, month) <= (end_inclusive.year, end_inclusive.month):
        months.append(YearMonth(year=year, month=month))
        month += 1
        if month > 12:
            month = 1
            year += 1
    return months


def _normalize_optional_arg(*, value: str | None) -> str | None:
    if value is None:
        return None
    if value.strip() == "":
        return None
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge NYC Taxi rides into DuckDB.")
    parser.add_argument("--config-path", default="config.yaml")
    parser.add_argument("--year-month-start", default=None)
    parser.add_argument("--year-month-end-inclusive", default=None)
    return parser.parse_args()


def _build_file_list(*, output_dir: Path, year_months: Iterable[YearMonth]) -> list[str]:
    files: list[str] = []
    for year_month in year_months:
        file_path = output_dir / f"{year_month.as_file_stem}.parquet"
        if file_path.exists():
            files.append(str(file_path))
        else:
            print(f"Missing file, skipping: {file_path}")
    return files


def _load_merge_sqls(*, query_path: Path) -> tuple[str, str]:
    raw_sql = query_path.read_text(encoding="utf-8")
    marker = "-- MERGE STATEMENT"
    if marker not in raw_sql:
        raise ValueError(f"Expected marker not found in SQL: {marker}")
    staging_sql, merge_sql = raw_sql.split(marker, maxsplit=1)
    return staging_sql.strip(), merge_sql.strip()


def _sql_literal(*, value: str | None) -> str:
    if value is None:
        return "NULL"
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def _sql_string_list(*, values: list[str]) -> str:
    escaped_values = [value.replace("'", "''") for value in values]
    quoted = [f"'{value}'" for value in escaped_values]
    return "[" + ", ".join(quoted) + "]"


def _merge_into_raw(
    *,
    connection: duckdb.DuckDBPyConnection,
    parquet_files: list[str],
    query_path: Path,
    year_month_start: str | None,
    year_month_end_inclusive: str | None,
) -> None:
    staging_sql, merge_sql = _load_merge_sqls(query_path=query_path)
    rendered_staging_sql = (
        staging_sql.replace("{{parquet_files}}", _sql_string_list(values=parquet_files))
        .replace("{{year_month_start}}", _sql_literal(value=year_month_start))
        .replace(
            "{{year_month_end_inclusive}}",
            _sql_literal(value=year_month_end_inclusive),
        )
    )
    connection.execute(query=rendered_staging_sql)
    connection.execute(query=merge_sql)


def main() -> None:
    args = _parse_args()
    config = load_local_config(path=Path(args.config_path))

    year_month_start = _normalize_optional_arg(value=args.year_month_start)
    year_month_end = _normalize_optional_arg(value=args.year_month_end_inclusive)

    if year_month_start and year_month_end:
        start = _parse_year_month(value=year_month_start)
        end = _parse_year_month(value=year_month_end)
        target_months = _range_year_months(start=start, end_inclusive=end)
    elif year_month_start or year_month_end:
        raise ValueError("Provide both --year-month-start and --year-month-end-inclusive.")
    else:
        target_months = _last_three_full_months(today=date.today())

    output_dir = config.data_dir / config.taxi_data_subdir / "yellow"
    parquet_files = _build_file_list(output_dir=output_dir, year_months=target_months)
    if not parquet_files:
        print("No parquet files found for the requested range.")
        return

    repo_root = Path(__file__).resolve().parents[2]
    merge_query_path = (
        repo_root / "local" / "queries" / "merge__raw__taxi_yellow_tripdata_fact.sql"
    )
    config.duckdb_location.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(database=str(config.duckdb_location)) as connection:
        _merge_into_raw(
            connection=connection,
            parquet_files=parquet_files,
            query_path=merge_query_path,
            year_month_start=year_month_start,
            year_month_end_inclusive=year_month_end,
        )


if __name__ == "__main__":
    main()

