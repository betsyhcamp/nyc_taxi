from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import requests

from local import load_local_config


@dataclass(frozen=True)
class YearMonth:
    year: int
    month: int

    @property
    def as_file_stem(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"


def _parse_year_month(*, value: str) -> YearMonth:
    parts = value.split(sep="-")
    if len(parts) != 2:
        raise ValueError(f"Invalid year-month format: {value}. Expected YYYY-MM.")
    return YearMonth(year=int(parts[0]), month=int(parts[1]))


def _last_three_full_months(*, today: date) -> list[YearMonth]:
    months: list[YearMonth] = []
    for i in range(1, 4):
        target_date = today - timedelta(days=31 * i)
        months.append(YearMonth(year=target_date.year, month=target_date.month))
    return months


def _range_year_months(
    *, start: YearMonth, end_inclusive: YearMonth
) -> list[YearMonth]:
    months: list[YearMonth] = []
    year = start.year
    month = start.month
    while (year, month) <= (end_inclusive.year, end_inclusive.month):
        months.append(YearMonth(year=year, month=month))
        month += 1
        if month > 12:
            month = 1
            year += 1
    return months


def _normalize_optional_arg(*, value: str | None) -> str | None:
    if value is None:
        return None
    if value.strip() == "":
        return None
    return value


def _download_file(*, url: str, destination: Path) -> None:
    response = requests.get(url=url, timeout=60)
    response.raise_for_status()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)


def download_taxi_data(*, year_months: Iterable[YearMonth], output_dir: Path) -> None:
    files_downloaded = 0
    files_skipped = 0

    for year_month in year_months:
        file_stem = year_month.as_file_stem
        local_path = output_dir / f"{file_stem}.parquet"
        if local_path.exists():
            print(f"Already exists, skipping: {local_path}")
            files_skipped += 1
            continue

        remote_name = f"yellow_tripdata_{file_stem}.parquet"
        url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{remote_name}"
        print(f"Downloading {url} -> {local_path}")
        _download_file(url=url, destination=local_path)
        files_downloaded += 1

    print(
        f"Download summary: {files_downloaded} files downloaded, {files_skipped} files skipped"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download NYC Taxi rides locally.")
    parser.add_argument("--config-path", default="config.yaml")
    parser.add_argument("--year-month-start", default=None)
    parser.add_argument("--year-month-end-inclusive", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_local_config(path=Path(args.config_path))

    year_month_start = _normalize_optional_arg(value=args.year_month_start)
    year_month_end = _normalize_optional_arg(value=args.year_month_end_inclusive)

    if year_month_start and year_month_end:
        start = _parse_year_month(value=year_month_start)
        end = _parse_year_month(value=year_month_end)
        target_months = _range_year_months(start=start, end_inclusive=end)
    elif year_month_start or year_month_end:
        raise ValueError("Provide both --year-month-start and --year-month-end-inclusive.")
    else:
        target_months = _last_three_full_months(today=date.today())

    output_dir = (
        config.data_dir / config.taxi_data_subdir / "yellow"
    )
    download_taxi_data(year_months=target_months, output_dir=output_dir)


if __name__ == "__main__":
    main()

