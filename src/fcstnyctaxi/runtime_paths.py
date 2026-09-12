"""Dockerfile.train's `COPY config` destination, named once.

Only `components/` may import this; ruff's TID251 `banned-api` entry enforces it,
and an importable name is the point — a bare string cannot be banned. A constant,
not a validating accessor, which would fail at compile time with no tree to check.
"""

from pathlib import Path

CONFIG_DIR: Path = Path("/app/config")
