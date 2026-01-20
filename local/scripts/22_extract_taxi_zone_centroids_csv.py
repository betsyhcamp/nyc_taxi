from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from local import load_local_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract taxi zone centroids to CSV."
    )
    parser.add_argument("--config-path", default="config.yaml")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _sql_literal(*, value: str) -> str:
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def _shape_source_from_zip(*, zip_path: Path) -> str:
    return f"/vsizip/{zip_path.as_posix()}/taxi_zones.shp"


def _extract_centroids_csv(*, zip_path: Path, csv_path: Path, force: bool) -> None:
    if csv_path.exists() and not force:
        print(f"Already exists, skipping: {csv_path}")
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect() as connection:
        connection.execute(query="INSTALL spatial;")
        connection.execute(query="LOAD spatial;")
        shape_source = _shape_source_from_zip(zip_path=zip_path)
        connection.execute(
            query=(
                "CREATE OR REPLACE TEMP VIEW taxi_zone_shapes AS "
                "SELECT "
                "CAST(locationid AS BIGINT) AS taxi_zone_id, "
                "st_y(st_centroid(geom)) AS centroid_latitude, "
                "st_x(st_centroid(geom)) AS centroid_longitude "
                f"FROM st_read({_sql_literal(value=shape_source)});"
            )
        )
        connection.execute(
            query=(
                "COPY (SELECT taxi_zone_id, centroid_latitude, centroid_longitude "
                "FROM taxi_zone_shapes) "
                f"TO {_sql_literal(value=str(csv_path))} "
                "(HEADER, DELIMITER ',');"
            )
        )


def main() -> None:
    args = _parse_args()
    config = load_local_config(path=Path(args.config_path))

    zip_path = config.data_dir / config.taxi_data_subdir / "zones" / "taxi_zones.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Zone shapefile zip not found: {zip_path}")

    csv_path = (
        config.data_dir
        / config.taxi_data_subdir
        / "zones"
        / "taxi_zone_centroids.csv"
    )
    _extract_centroids_csv(zip_path=zip_path, csv_path=csv_path, force=args.force)


if __name__ == "__main__":
    main()

