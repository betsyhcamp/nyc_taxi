from __future__ import annotations

from pathlib import Path
import yaml
from pydantic import BaseModel, Field


class LocalConfig(BaseModel):
    duckdb_location: Path = Field(..., description="Path to DuckDB database file")
    data_dir: Path = Field(..., description="Base data directory")
    taxi_data_subdir: str = Field(..., description="Subdirectory for taxi data")

    def resolve_paths(self, *, base_dir: Path) -> "LocalConfig":
        return LocalConfig(
            duckdb_location=_resolve_path(path=self.duckdb_location, base_dir=base_dir),
            data_dir=_resolve_path(path=self.data_dir, base_dir=base_dir),
            taxi_data_subdir=self.taxi_data_subdir,
        )


def load_local_config(*, path: Path) -> LocalConfig:
    with path.open(mode="r", encoding="utf-8") as file_object:
        data = yaml.safe_load(stream=file_object) or {}
    config = LocalConfig.model_validate(obj=data)
    return config.resolve_paths(base_dir=path.parent)


def _resolve_path(*, path: Path, base_dir: Path) -> Path:
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()

