from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import duckdb

from local import load_local_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ddl_* queries against the local DuckDB database."
    )
    parser.add_argument("--config-path", default="config.yaml")
    return parser.parse_args()


def _load_ddl_files(*, queries_dir: Path) -> list[Path]:
    ddl_files = [path for path in queries_dir.glob(pattern="ddl_*.sql")]
    ddl_files.sort()
    return ddl_files


def _execute_ddl_files(
    *, connection: duckdb.DuckDBPyConnection, ddl_files: Iterable[Path]
) -> None:
    for ddl_file in ddl_files:
        sql = ddl_file.read_text(encoding="utf-8")
        if sql.strip() == "":
            logging.info("Skipping empty query file: %s", ddl_file.name)
            continue
        logging.info("Executing: %s", ddl_file.name)
        connection.execute(query=sql)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    config = load_local_config(path=Path(args.config_path))

    repo_root = Path(__file__).resolve().parents[2]
    queries_dir = repo_root / "local" / "queries"
    if not queries_dir.exists():
        raise FileNotFoundError(f"Queries directory not found: {queries_dir}")

    ddl_files = _load_ddl_files(queries_dir=queries_dir)
    if not ddl_files:
        logging.info("No ddl_* queries found in: %s", queries_dir)
        return

    config.duckdb_location.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(database=str(config.duckdb_location)) as connection:
        _execute_ddl_files(connection=connection, ddl_files=ddl_files)


if __name__ == "__main__":
    main()

