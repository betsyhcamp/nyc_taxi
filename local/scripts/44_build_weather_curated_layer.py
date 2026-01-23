from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from local import load_local_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build curated weather tables with priority-based coalescing."
    )
    parser.add_argument("--config-path", default="config.yaml")
    return parser.parse_args()


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

    repo_root = Path(__file__).resolve().parents[2]
    queries_dir = repo_root / "local" / "queries"

    hourly_path = queries_dir / "view__curated__weather_hourly_fact.sql"
    daily_path = queries_dir / "view__curated__weather_daily_fact.sql"

    if not hourly_path.exists():
        raise FileNotFoundError(f"Query file not found: {hourly_path}")
    if not daily_path.exists():
        raise FileNotFoundError(f"Query file not found: {daily_path}")

    config.duckdb_location.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(database=str(config.duckdb_location)) as connection:
        print("Building curated.weather_hourly_fact...")
        _execute_sql_file(connection=connection, query_path=hourly_path)

        print("Building curated.weather_daily_fact...")
        _execute_sql_file(connection=connection, query_path=daily_path)

    print("Curated weather tables built successfully.")


if __name__ == "__main__":
    main()
