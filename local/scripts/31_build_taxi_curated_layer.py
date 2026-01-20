from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import duckdb

from local import load_local_config


@dataclass(frozen=True)
class YearMonth:
    year: int
    month: int

    @property
    def as_year_month(self) -> str:
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


def _month_start(*, year_month: YearMonth) -> date:
    return date(year_month.year, year_month.month, 1)


def _month_end(*, year_month: YearMonth) -> date:
    if year_month.month == 12:
        next_month = date(year_month.year + 1, 1, 1)
    else:
        next_month = date(year_month.year, year_month.month + 1, 1)
    return next_month - timedelta(days=1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build curated layer tables for a year-month range."
    )
    parser.add_argument("--config-path", default="config.yaml")
    parser.add_argument("--year-month-start", default=None)
    parser.add_argument("--year-month-end-inclusive", default=None)
    return parser.parse_args()


def _sql_literal(*, value: str) -> str:
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def _render_date_dim_sql(*, query_path: Path, start_date: date, end_date: date) -> str:
    raw_sql = query_path.read_text(encoding="utf-8")
    return (
        raw_sql.replace("{{start_date}}", _sql_literal(value=start_date.isoformat()))
        .replace("{{end_date}}", _sql_literal(value=end_date.isoformat()))
        .strip()
    )


def _execute_sql_file(
    *, connection: duckdb.DuckDBPyConnection, query_path: Path
) -> None:
    sql = query_path.read_text(encoding="utf-8").strip()
    if sql == "":
        return
    connection.execute(query=sql)


def main() -> None:
    args = _parse_args()
    config = load_local_config(path=Path(args.config_path))

    year_month_start = args.year_month_start
    year_month_end = args.year_month_end_inclusive
    if year_month_start and year_month_end:
        start = _parse_year_month(value=year_month_start)
        end = _parse_year_month(value=year_month_end)
        start_date = _month_start(year_month=start)
        end_date = _month_end(year_month=end)
    elif year_month_start or year_month_end:
        raise ValueError("Provide both --year-month-start and --year-month-end-inclusive.")
    else:
        recent_months = _last_three_full_months(today=date.today())
        start_date = _month_start(year_month=recent_months[-1])
        end_date = _month_end(year_month=recent_months[0])

    repo_root = Path(__file__).resolve().parents[2]
    queries_dir = repo_root / "local" / "queries"
    date_dim_path = queries_dir / "merge__curated__date_dim.sql"
    staging_path = queries_dir / "view__staging__taxi_yellow_tripdata_fact.sql"
    curated_fact_path = queries_dir / "view__curated__taxi_yellow_tripdata_fact.sql"
    curated_zone_path = queries_dir / "view__curated__taxi_zone_dim.sql"

    if not date_dim_path.exists():
        raise FileNotFoundError(f"Query file not found: {date_dim_path}")

    config.duckdb_location.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(database=str(config.duckdb_location)) as connection:
        date_dim_sql = _render_date_dim_sql(
            query_path=date_dim_path,
            start_date=start_date,
            end_date=end_date,
        )
        connection.execute(query=date_dim_sql)
        _execute_sql_file(connection=connection, query_path=staging_path)
        _execute_sql_file(connection=connection, query_path=curated_zone_path)
        _execute_sql_file(connection=connection, query_path=curated_fact_path)


if __name__ == "__main__":
    main()

