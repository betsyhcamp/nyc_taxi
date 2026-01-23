from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import duckdb

from local import load_local_config


# Weather stations in priority order
WEATHER_STATION_IDS = ["KNYC0", "KTEB0", "KJRB0", "72502"]


def _parse_date(*, value: str) -> date:
    return date.fromisoformat(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge weather data into DuckDB raw tables."
    )
    parser.add_argument("--config-path", default="config.yaml")
    parser.add_argument(
        "--start-date",
        default=None,
        help="Start date (inclusive) filter in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="End date (inclusive) filter in YYYY-MM-DD format.",
    )
    return parser.parse_args()


def _sql_literal(*, value: str | None) -> str:
    if value is None:
        return "NULL"
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def _sql_string_list(*, values: list[str]) -> str:
    escaped_values = [value.replace("'", "''") for value in values]
    quoted = [f"'{value}'" for value in escaped_values]
    return "[" + ", ".join(quoted) + "]"


def _load_merge_sqls(*, query_path: Path) -> tuple[str, str]:
    raw_sql = query_path.read_text(encoding="utf-8")
    marker = "-- MERGE STATEMENT"
    if marker not in raw_sql:
        raise ValueError(f"Expected marker not found in SQL: {marker}")
    staging_sql, merge_sql = raw_sql.split(marker, maxsplit=1)
    return staging_sql.strip(), merge_sql.strip()


def _build_file_list(*, data_dir: Path, station_ids: list[str]) -> list[str]:
    files: list[str] = []
    for station_id in station_ids:
        file_path = data_dir / f"{station_id}.parquet"
        if file_path.exists():
            files.append(str(file_path))
        else:
            print(f"Warning: Missing parquet file, skipping: {file_path}")
    return files


def _merge_weather_data(
    *,
    connection: duckdb.DuckDBPyConnection,
    parquet_files: list[str],
    query_path: Path,
    start_date: str | None,
    end_date: str | None,
) -> None:
    staging_sql, merge_sql = _load_merge_sqls(query_path=query_path)
    rendered_staging_sql = (
        staging_sql.replace("{{parquet_files}}", _sql_string_list(values=parquet_files))
        .replace("{{start_date}}", _sql_literal(value=start_date))
        .replace("{{end_date}}", _sql_literal(value=end_date))
    )
    connection.execute(query=rendered_staging_sql)
    connection.execute(query=merge_sql)


def main() -> None:
    args = _parse_args()
    config = load_local_config(path=Path(args.config_path))

    start_date = args.start_date
    end_date = args.end_date

    if start_date:
        _parse_date(value=start_date)  # Validate format
    if end_date:
        _parse_date(value=end_date)  # Validate format

    repo_root = Path(__file__).resolve().parents[2]
    queries_dir = repo_root / "local" / "queries"

    # Merge hourly data
    hourly_dir = config.data_dir / "weather" / "hourly"
    hourly_files = _build_file_list(data_dir=hourly_dir, station_ids=WEATHER_STATION_IDS)
    if hourly_files:
        hourly_query_path = queries_dir / "merge__raw__weather_hourly_fact.sql"
        config.duckdb_location.parent.mkdir(parents=True, exist_ok=True)
        with duckdb.connect(database=str(config.duckdb_location)) as connection:
            _merge_weather_data(
                connection=connection,
                parquet_files=hourly_files,
                query_path=hourly_query_path,
                start_date=start_date,
                end_date=end_date,
            )
        print(f"Merged hourly weather data from {len(hourly_files)} files")
    else:
        print("No hourly parquet files found. Run 42_download_weather_data.py first.")

    # Merge daily data
    daily_dir = config.data_dir / "weather" / "daily"
    daily_files = _build_file_list(data_dir=daily_dir, station_ids=WEATHER_STATION_IDS)
    if daily_files:
        daily_query_path = queries_dir / "merge__raw__weather_daily_fact.sql"
        with duckdb.connect(database=str(config.duckdb_location)) as connection:
            _merge_weather_data(
                connection=connection,
                parquet_files=daily_files,
                query_path=daily_query_path,
                start_date=start_date,
                end_date=end_date,
            )
        print(f"Merged daily weather data from {len(daily_files)} files")
    else:
        print("No daily parquet files found. Run 42_download_weather_data.py first.")


if __name__ == "__main__":
    main()
