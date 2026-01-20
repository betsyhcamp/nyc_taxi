from __future__ import annotations

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from local import load_local_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge taxi zone centroids CSV into DuckDB."
    )
    parser.add_argument("--config-path", default="config.yaml")
    return parser.parse_args()


def _sql_literal(*, value: str) -> str:
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def _shape_source_from_zip(*, zip_path: Path) -> str:
    return f"/vsizip/{zip_path.as_posix()}/taxi_zones.shp"


def _render_sql(*, query_path: Path, zip_path: Path) -> str:
    raw_sql = query_path.read_text(encoding="utf-8")
    shape_source = _shape_source_from_zip(zip_path=zip_path)
    return raw_sql.replace(
        "{{zone_shape_zip}}", _sql_literal(value=shape_source)
    )


def main() -> None:
    args = _parse_args()
    config = load_local_config(path=Path(args.config_path))
    repo_root = Path(__file__).resolve().parents[2]

    zip_path = config.data_dir / config.taxi_data_subdir / "zones" / "taxi_zones.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Zone shapefile zip not found: {zip_path}")

    query_path = (
        repo_root / "local" / "queries" / "merge__raw__taxi_zone_centroids_dim.sql"
    )
    sql = _render_sql(query_path=query_path, zip_path=zip_path)

    config.duckdb_location.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(database=str(config.duckdb_location)) as connection:
        connection.execute(query=sql)


if __name__ == "__main__":
    main()

