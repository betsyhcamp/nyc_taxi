from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path

import meteostat
from meteostat.api.hourly import hourly
from meteostat.api.daily import daily

from local import load_local_config


# Weather stations in priority order
WEATHER_STATION_IDS = ["KNYC0", "KTEB0", "KJRB0", "72502"]

# Meteostat blocks requests > 3 years by default
meteostat.config.block_large_requests = False


def _parse_date(*, value: str) -> date:
    return date.fromisoformat(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download hourly and daily weather data from Meteostat."
    )
    parser.add_argument("--config-path", default="config.yaml")
    parser.add_argument(
        "--start-date",
        default="2015-01-01",
        help="Start date (inclusive) in YYYY-MM-DD format. Default: 2015-01-01",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="End date (inclusive) in YYYY-MM-DD format. Default: today",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files if they exist.",
    )
    return parser.parse_args()


def _download_hourly_data(
    *,
    station_id: str,
    start_date: date,
    end_date: date,
    output_path: Path,
    force: bool,
) -> None:
    if output_path.exists() and not force:
        print(f"  Hourly file exists, skipping: {output_path}")
        return

    print(f"  Fetching hourly data for {station_id}...")
    start_dt = datetime(year=start_date.year, month=start_date.month, day=start_date.day)
    end_dt = datetime(
        year=end_date.year, month=end_date.month, day=end_date.day, hour=23, minute=59
    )

    timeseries = hourly(station=station_id, start=start_dt, end=end_dt)
    df = timeseries.fetch()

    if df.empty:
        print(f"  Warning: No hourly data returned for {station_id}")
        return

    df = df.reset_index()
    df["station_id"] = station_id
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path=output_path, index=False)
    print(f"  Wrote {len(df)} hourly records to {output_path}")


def _download_daily_data(
    *,
    station_id: str,
    start_date: date,
    end_date: date,
    output_path: Path,
    force: bool,
) -> None:
    if output_path.exists() and not force:
        print(f"  Daily file exists, skipping: {output_path}")
        return

    print(f"  Fetching daily data for {station_id}...")
    start_dt = datetime(year=start_date.year, month=start_date.month, day=start_date.day)
    end_dt = datetime(year=end_date.year, month=end_date.month, day=end_date.day)

    timeseries = daily(station=station_id, start=start_dt, end=end_dt)
    df = timeseries.fetch()

    if df.empty:
        print(f"  Warning: No daily data returned for {station_id}")
        return

    df = df.reset_index()
    df["station_id"] = station_id
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path=output_path, index=False)
    print(f"  Wrote {len(df)} daily records to {output_path}")


def main() -> None:
    args = _parse_args()
    config = load_local_config(path=Path(args.config_path))

    start_date = _parse_date(value=args.start_date)
    end_date = (
        _parse_date(value=args.end_date) if args.end_date else date.today()
    )

    hourly_dir = config.data_dir / "weather" / "hourly"
    daily_dir = config.data_dir / "weather" / "daily"

    print(f"Downloading weather data from {start_date} to {end_date}")

    for station_id in WEATHER_STATION_IDS:
        print(f"Processing station: {station_id}")

        hourly_path = hourly_dir / f"{station_id}.parquet"
        _download_hourly_data(
            station_id=station_id,
            start_date=start_date,
            end_date=end_date,
            output_path=hourly_path,
            force=args.force,
        )

        daily_path = daily_dir / f"{station_id}.parquet"
        _download_daily_data(
            station_id=station_id,
            start_date=start_date,
            end_date=end_date,
            output_path=daily_path,
            force=args.force,
        )

    print("Download complete.")


if __name__ == "__main__":
    main()
