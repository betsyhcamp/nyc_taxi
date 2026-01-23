from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
from meteostat.api.stations import Stations

from local import load_local_config


# Central Park coordinates used as anchor for distance calculations
CENTRAL_PARK_LAT = 40.7829
CENTRAL_PARK_LON = -73.9654

# Weather stations in priority order
WEATHER_STATIONS = [
    {"id": "KNYC0", "priority_rank": 1},
    {"id": "KTEB0", "priority_rank": 2},
    {"id": "KJRB0", "priority_rank": 3},
    {"id": "72502", "priority_rank": 4},
]


def _haversine_km(*, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two lat/lon points in kilometers."""
    earth_radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return earth_radius_km * c


def _fetch_station_metadata(*, station_id: str) -> dict:
    """Fetch metadata for a single weather station from Meteostat."""
    stations_db = Stations()
    station = stations_db.meta(station=station_id)
    return {
        "weather_station_id": station.id,
        "name": station.name,
        "country": station.country,
        "region": station.region,
        "latitude": station.latitude,
        "longitude": station.longitude,
        "elevation_meters": station.elevation,
        "timezone": station.timezone,
    }


def _build_station_dataframe() -> pd.DataFrame:
    """Build a DataFrame with metadata for all configured weather stations."""
    rows = []
    for station_config in WEATHER_STATIONS:
        station_id = station_config["id"]
        priority_rank = station_config["priority_rank"]
        try:
            metadata = _fetch_station_metadata(station_id=station_id)
            metadata["priority_rank"] = priority_rank
            if metadata["latitude"] is not None and metadata["longitude"] is not None:
                metadata["distance_from_central_park_km"] = _haversine_km(
                    lat1=CENTRAL_PARK_LAT,
                    lon1=CENTRAL_PARK_LON,
                    lat2=metadata["latitude"],
                    lon2=metadata["longitude"],
                )
            else:
                metadata["distance_from_central_park_km"] = None
            rows.append(metadata)
        except Exception as e:
            print(f"Warning: Failed to fetch metadata for station {station_id}: {e}")
    return pd.DataFrame(data=rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download weather station dimension data from Meteostat."
    )
    parser.add_argument("--config-path", default="config.yaml")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing file if it exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_local_config(path=Path(args.config_path))

    output_dir = config.data_dir / "weather" / "station_dim"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "weather_stations.csv"

    if output_path.exists() and not args.force:
        print(f"File already exists, skipping: {output_path}")
        print("Use --force to overwrite.")
        return

    print("Fetching weather station metadata from Meteostat...")
    df = _build_station_dataframe()

    df.to_csv(path_or_buf=output_path, index=False)
    print(f"Wrote {len(df)} stations to {output_path}")


if __name__ == "__main__":
    main()
