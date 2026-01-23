from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import duckdb

from local import load_local_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge curated.date_dim and curated.datetime_hour_dim for a specified date range."
    )
    parser.add_argument("--config-path", default="config.yaml")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    return parser.parse_args()


def _sql_literal(*, value: str) -> str:
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def _render_merge_sql(*, query_path: Path, start_date: str, end_date: str) -> str:
    raw_sql = query_path.read_text(encoding="utf-8")
    return (
        raw_sql.replace("{{start_date}}", _sql_literal(value=start_date))
        .replace("{{end_date}}", _sql_literal(value=end_date))
        .strip()
    )


def _validate_date(*, value: str) -> None:
    date.fromisoformat(value)


def main() -> None:
    args = _parse_args()
    _validate_date(value=args.start_date)
    _validate_date(value=args.end_date)
    config = load_local_config(path=Path(args.config_path))

    repo_root = Path(__file__).resolve().parents[2]
    queries_dir = repo_root / "local" / "queries"

    date_dim_path = queries_dir / "merge__curated__date_dim.sql"
    datetime_hour_dim_path = queries_dir / "merge__curated__datetime_hour_dim.sql"

    if not date_dim_path.exists():
        raise FileNotFoundError(f"Query file not found: {date_dim_path}")
    if not datetime_hour_dim_path.exists():
        raise FileNotFoundError(f"Query file not found: {datetime_hour_dim_path}")

    date_dim_sql = _render_merge_sql(
        query_path=date_dim_path,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    datetime_hour_dim_sql = _render_merge_sql(
        query_path=datetime_hour_dim_path,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    config.duckdb_location.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(database=str(config.duckdb_location)) as connection:
        print("Merging curated.date_dim...")
        connection.execute(query=date_dim_sql)
        print("Merging curated.datetime_hour_dim...")
        connection.execute(query=datetime_hour_dim_sql)
    print("Done.")


if __name__ == "__main__":
    main()

