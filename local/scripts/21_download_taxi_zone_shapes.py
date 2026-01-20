from __future__ import annotations

import argparse
from pathlib import Path

import requests

from local import load_local_config

TAXI_ZONE_SHAPES_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"


def _download_file(*, url: str, destination: Path) -> None:
    response = requests.get(url=url, timeout=60)
    response.raise_for_status()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download NYC TLC taxi zone shapefile zip."
    )
    parser.add_argument("--config-path", default="config.yaml")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_local_config(path=Path(args.config_path))
    output_dir = config.data_dir / config.taxi_data_subdir / "zones"
    output_path = output_dir / "taxi_zones.zip"

    if output_path.exists() and not args.force:
        print(f"Already exists, skipping: {output_path}")
        return

    print(f"Downloading {TAXI_ZONE_SHAPES_URL} -> {output_path}")
    _download_file(url=TAXI_ZONE_SHAPES_URL, destination=output_path)


if __name__ == "__main__":
    main()

