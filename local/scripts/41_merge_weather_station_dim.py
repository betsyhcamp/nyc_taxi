from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from local import load_local_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge weather station dimension into DuckDB."
    )
    parser.add_argument("--config-path", default="config.yaml")
    return parser.parse_args()


def _sql_literal(*, value: str) -> str:
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def _load_merge_sqls(*, query_path: Path) -> tuple[str, str]:
    raw_sql = query_path.read_text(encoding="utf-8")
    marker = "-- MERGE STATEMENT"
    if marker not in raw_sql:
        raise ValueError(f"Expected marker not found in SQL: {marker}")
    staging_sql, merge_sql = raw_sql.split(marker, maxsplit=1)
    return staging_sql.strip(), merge_sql.strip()


def _merge_into_raw(
    *,
    connection: duckdb.DuckDBPyConnection,
    csv_path: str,
    query_path: Path,
) -> None:
    staging_sql, merge_sql = _load_merge_sqls(query_path=query_path)
    rendered_staging_sql = staging_sql.replace(
        "{{weather_station_csv}}", _sql_literal(value=csv_path)
    )
    connection.execute(query=rendered_staging_sql)
    connection.execute(query=merge_sql)


def main() -> None:
    args = _parse_args()
    config = load_local_config(path=Path(args.config_path))

    csv_path = config.data_dir / "weather" / "station_dim" / "weather_stations.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Weather station CSV not found: {csv_path}. "
            "Run 40_download_weather_station_dim.py first."
        )

    repo_root = Path(__file__).resolve().parents[2]
    merge_query_path = repo_root / "local" / "queries" / "merge__raw__weather_station_dim.sql"

    config.duckdb_location.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(database=str(config.duckdb_location)) as connection:
        _merge_into_raw(
            connection=connection,
            csv_path=str(csv_path),
            query_path=merge_query_path,
        )
    print(f"Merged weather station dimension from {csv_path}")


if __name__ == "__main__":
    main()
