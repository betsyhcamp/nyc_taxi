from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from fcstnyctaxi.lib.run_identity import (
    _format_run_id,
    mint_run_id,
    require_path_safe_run_id,
    require_train_run_id,
    require_unfinished_run,
)
from fcstnyctaxi.schemas.run_identity import RUN_ID_TIMESTAMP_FORMAT
from fcstnyctaxi.schemas.run_outputs import (
    RegisteredModel,
    TrainingData,
    TrainRunOutputs,
)
from fcstnyctaxi.schemas.storage.common import RUN_OUTPUT_FILENAME

# Zero microseconds, the instant isoformat() prints without a fraction.
PINNED_WHEN = datetime(2026, 10, 9, 8, 32, 29, tzinfo=UTC)
VALID_ID = "train-20261009t083229000000z"

MALFORMED_IDS = {
    "another slice": _format_run_id("feature", PINNED_WHEN, local=False),
    "uppercase T": "train-20261009T083229000000z",
    "uppercase Z": "train-20261009t083229000000Z",
    # What the minter before the slice prefix wrote.
    "bare old format": PINNED_WHEN.strftime(RUN_ID_TIMESTAMP_FORMAT),
    "trailing newline": f"{VALID_ID}\n",
    "non-ASCII digit": VALID_ID.replace("9", "\u0669"),
}

PATH_UNSAFE_RUN_IDS = ["f1\n", "..", ".hidden", "-x", ""]


def test_zero_microseconds_still_print_six_digits() -> None:
    """isoformat() drops the fraction here, and a shorter id sorts out of order."""
    assert _format_run_id("train", PINNED_WHEN, local=False) == VALID_ID


def test_the_suffix_is_appended_only_when_local() -> None:
    """The suffix is the one mark of the local runner, so a Vertex id must lack it."""
    assert _format_run_id("train", PINNED_WHEN, local=True) == f"{VALID_ID}-local"
    assert _format_run_id("train", PINNED_WHEN, local=False) == VALID_ID


def test_ids_minted_in_sequence_sort_in_time_order() -> None:
    """Listings sort run ids as strings, so string order must be time order."""
    run_ids = [mint_run_id("train") for _ in range(20)]

    assert run_ids == sorted(run_ids)


@pytest.mark.parametrize("local", [False, True])
def test_a_minted_train_id_passes_for_its_own_entry_point(local: bool) -> None:
    """The formatter and the pattern are written apart, so each checks the other."""
    require_train_run_id(mint_run_id("train", local=local), "--run-id", local=local)


@pytest.mark.parametrize("run_id", MALFORMED_IDS.values(), ids=MALFORMED_IDS.keys())
def test_an_id_off_the_format_is_refused_naming_the_flag(run_id: str) -> None:
    """Untagged and checked as a Vertex id, so only the format can refuse it."""
    with pytest.raises(ValueError, match="--run-id"):
        require_train_run_id(run_id, "--run-id", local=False)


def test_a_tagged_id_is_refused_for_a_vertex_run() -> None:
    """A Vertex run under a local id would mix runners in one run root."""
    with pytest.raises(ValueError, match="--run-id"):
        require_train_run_id(mint_run_id("train", local=True), "--run-id", local=False)


def test_an_untagged_id_is_refused_for_the_local_runner() -> None:
    """A local run under a Vertex id would mix runners in one run root."""
    with pytest.raises(ValueError, match="--run-id"):
        require_train_run_id(mint_run_id("train"), "--run-id", local=True)


@pytest.mark.parametrize("run_id", PATH_UNSAFE_RUN_IDS)
def test_require_path_safe_run_id_rejects_unsafe_ids(run_id: str) -> None:
    """Each case is a distinct hazard, not a variation on one."""
    with pytest.raises(ValueError, match="--run-id"):
        require_path_safe_run_id(run_id, "--run-id")


def test_require_path_safe_run_id_accepts_a_minted_feature_id() -> None:
    """A guard rejecting mint_run_id's output would fail every unnamed Feature run."""
    require_path_safe_run_id(mint_run_id("feature"), "--feature-run-id")


def _finish(run_dir: Path) -> None:
    """run_output.json as register_model_impl writes it, read back as a self-check."""
    outputs = TrainRunOutputs(
        train_run_id=run_dir.name,
        published=RegisteredModel(
            model_tag="projects/1/locations/us-central1/models/m-naive@1",
            bundle_uri=f"gs://bucket/dev/train/{run_dir.name}/final_fit/naive/",
        ),
        feature_run_id="feature-20261009t000000000000z",
        env="dev",
        schema_version="0.1.0",
        git_hash="a" * 40,
        completed_at=PINNED_WHEN,
        training_data=TrainingData(
            train_end_ds=date(2025, 10, 19), n_series=3, n_obs=60
        ),
    )
    run_dir.mkdir(parents=True)
    record = run_dir / RUN_OUTPUT_FILENAME
    record.write_text(outputs.model_dump_json(indent=2, exclude_none=True) + "\n")
    assert TrainRunOutputs.model_validate_json(record.read_text()) == outputs


def test_require_unfinished_run_refuses_a_finished_run_naming_it(
    tmp_path: Path,
) -> None:
    """A rerun would delete its marker and rewrite what a registered version names."""
    run_dir = tmp_path / VALID_ID
    _finish(run_dir)

    with pytest.raises(ValueError, match=VALID_ID):
        require_unfinished_run(run_dir)


def test_require_unfinished_run_passes_a_run_with_no_record(tmp_path: Path) -> None:
    """An interrupted run has no record, and resuming one is why ids are reused."""
    run_dir = tmp_path / VALID_ID
    run_dir.mkdir()

    require_unfinished_run(run_dir)
