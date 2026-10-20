"""Minting and checking run ids, in the format `schemas/run_identity.py` defines."""

import re
from datetime import UTC, datetime
from pathlib import Path

from fcstnyctaxi.schemas.config.common import SliceName
from fcstnyctaxi.schemas.run_identity import (
    LOCAL_RUN_ID_SUFFIX,
    RUN_ID_TIMESTAMP_FORMAT,
    TRAIN_RUN_ID_PATTERN,
)
from fcstnyctaxi.schemas.storage.common import RUN_OUTPUT_FILENAME


def _format_run_id(slice_name: SliceName, when: datetime, *, local: bool) -> str:
    """`<slice>-<timestamp>`, suffixed when local. Pure, so tests can pin `when`."""
    run_id = f"{slice_name}-{when.strftime(RUN_ID_TIMESTAMP_FORMAT)}"
    return f"{run_id}{LOCAL_RUN_ID_SUFFIX}" if local else run_id


def mint_run_id(slice_name: SliceName, *, local: bool = False) -> str:
    """A new run id for `slice_name`, read from the UTC clock.

    Args:
        slice_name (SliceName): The slice the run belongs to, which prefixes the id.
        local (bool): Whether the local runner is minting it, which suffixes the id.

    Returns:
        str: `<slice>-YYYYMMDDtHHMMSSffffffz`, ending in `-local` when `local`.
    """
    return _format_run_id(slice_name, datetime.now(UTC), local=local)


def require_train_run_id(run_id: str, flag_name: str, *, local: bool) -> None:
    """Refuse a Training id off the format, or tagged for the other entry point.

    Args:
        run_id (str): The identifier to check.
        flag_name (str): Where it arrived, named in the error.
        local (bool): Whether the caller is the local runner, the one entry point
            whose ids end in `-local`.

    Raises:
        ValueError: `run_id` does not match `TRAIN_RUN_ID_PATTERN`, or its `-local`
            suffix disagrees with `local`.
    """
    # fullmatch rather than match, whose "$" also matches before a trailing newline.
    if not re.fullmatch(TRAIN_RUN_ID_PATTERN, run_id):
        raise ValueError(
            f"{flag_name} {run_id!r} must match {TRAIN_RUN_ID_PATTERN}, the form "
            "mint_run_id('train') writes."
        )
    if run_id.endswith(LOCAL_RUN_ID_SUFFIX) != local:
        entry_point = "the local runner" if local else "a Vertex run"
        rule = "must" if local else "must not"
        raise ValueError(
            f"{flag_name} {run_id!r} {rule} end in {LOCAL_RUN_ID_SUFFIX!r} for "
            f"{entry_point}; mint a new id rather than mixing runners in one run."
        )


def require_path_safe_run_id(run_id: str, flag_name: str) -> None:
    """Reject a run id that would build a malformed path or an unpasteable line.

    Every slice's run id becomes a GCS path segment and a local directory name.
    Call it as soon as the id is resolved, before anything builds or prints it.

    Args:
        run_id (str): The identifier to check.
        flag_name (str): The CLI flag it arrived on, named in the error so a caller
            holding two ids says which one to fix.

    Raises:
        ValueError: run_id does not match `[A-Za-z0-9][A-Za-z0-9._-]*`.
    """
    # The first character is anchored separately because [A-Za-z0-9._-]+ accepts
    # "..", ".hidden" and "-x"; fullmatch rather than match, whose "$" also
    # matches before the trailing newline that $(cat run_id.txt) supplies.
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise ValueError(
            f"{flag_name} {run_id!r} must start with a letter or digit and "
            "contain only letters, digits, '.', '_' or '-'."
        )


def require_unfinished_run(run_dir: Path) -> None:
    """Refuse to write into a run whose completion marker exists.

    Every step that writes under a run root calls it before its first write: a step
    can re-execute under a cached compose_configs, and each deletes its own marker
    before rewriting.

    Args:
        run_dir (Path): The run root, the gcsfuse mount on Vertex or the local mirror.

    Raises:
        ValueError: `run_output.json` exists under `run_dir`.
    """
    marker = run_dir / RUN_OUTPUT_FILENAME
    if marker.is_file():
        raise ValueError(
            f"Run {run_dir.name!r} finished: {marker} exists, and a finished run is "
            "never rewritten. Mint a new id to run again."
        )
