"""A finished run as register_model_impl leaves it, for the steps that must refuse one.

Imported as a top-level module, as ``stand_in_model`` is, by pytest's default import
mode.
"""

from datetime import UTC, date, datetime
from pathlib import Path

from fcstnyctaxi.schemas.run_identity import TrainRunIdentity
from fcstnyctaxi.schemas.run_outputs import (
    RegisteredModel,
    TrainingData,
    TrainRunOutputs,
)
from fcstnyctaxi.schemas.storage.common import (
    RUN_IDENTITY_FILENAME,
    RUN_OUTPUT_FILENAME,
)


def finish_run(run_dir: Path) -> None:
    """Write run_output.json for the run whose identity run_dir holds, through the
    schema register_model_impl writes it with."""
    identity = TrainRunIdentity.model_validate_json(
        (run_dir / RUN_IDENTITY_FILENAME).read_text()
    )
    outputs = TrainRunOutputs(
        train_run_id=identity.train_run_id,
        published=RegisteredModel(
            model_tag="projects/123456789/locations/us-central1/models/m-naive@1",
            bundle_uri=f"gs://bucket/dev/train/{identity.train_run_id}/final_fit/naive/",
        ),
        feature_run_id=identity.feature_run_id,
        env="dev",
        schema_version="0.1.0",
        git_hash=identity.git_hash,
        completed_at=datetime.now(UTC),
        training_data=TrainingData(
            train_end_ds=date(2025, 10, 19), n_series=3, n_obs=60
        ),
    )
    record = run_dir / RUN_OUTPUT_FILENAME
    record.write_text(outputs.model_dump_json(indent=2, exclude_none=True) + "\n")
    # A broken fixture reads as a broken check.
    assert TrainRunOutputs.model_validate_json(record.read_text()) == outputs


def file_bytes(root: Path) -> dict[Path, bytes]:
    """Every file under root, keyed by its path relative to root."""
    return {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
