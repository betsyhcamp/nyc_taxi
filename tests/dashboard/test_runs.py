import json
from datetime import UTC, date, datetime
from unittest.mock import Mock

import pytest
from fsspec import AbstractFileSystem
from fsspec.implementations.memory import MemoryFileSystem

from dashboard.shared import load, paths, runs
from fcstnyctaxi.lib.utils import generate_run_id
from fcstnyctaxi.schemas.run_outputs import (
    LatestRunPointer,
    RegisteredModel,
    TrainingData,
    TrainRunOutputs,
)

TRAIN_ROOT = "memory://BUCKET/dev/train/"
POINTER_URI = "memory://BUCKET/dev/_latest.json"

# Same date, with the uppercase id carrying the LATER time.
LATER_UPPERCASE = "20260920T235959000000Z"
EARLIER_LOWERCASE = "20260920t000000000000z"
HAND_SUPPLIED = "finalfit-c7-gate"


def _run_output_json(run_id: str) -> bytes:
    """The completion marker, dumped the way `register_model_impl` dumps it."""
    outputs = TrainRunOutputs(
        train_run_id=run_id,
        published=RegisteredModel(
            model_tag="projects/1/locations/us-central1/models/model-a@1",
            bundle_uri=f"gs://BUCKET/dev/train/{run_id}/final_fit/model_a/",
        ),
        feature_run_id="20260101t000000000000z",
        env="dev",
        schema_version="0.1.0",
        git_hash="0" * 40,
        completed_at=datetime(2026, 1, 1, tzinfo=UTC),
        training_data=TrainingData(
            train_end_ds=date(2025, 12, 28), n_series=3, n_obs=300
        ),
    )
    return outputs.model_dump_json(indent=2, exclude_none=True).encode()


def _pointer_json(run_id: str | None) -> bytes:
    """`_latest.json` as `register_model_impl` writes it, or with `train` unwritten.

    An unwritten slice reads the string "unset", which is why the reader cannot
    assume a record."""
    if run_id is None:
        pointer = LatestRunPointer(train="unset", inference="unset")
    else:
        record = json.loads(
            TrainRunOutputs.model_validate_json(
                _run_output_json(run_id)
            ).model_dump_json(exclude_none=True, exclude={"feature_run_id"})
        )
        pointer = LatestRunPointer(train=record, inference="unset")
    return pointer.model_dump_json(indent=2).encode()


@pytest.fixture
def fake_fs() -> AbstractFileSystem:
    """An empty MemoryFileSystem, cleared of any prior test's keys."""
    MemoryFileSystem.store.clear()
    MemoryFileSystem.pseudo_dirs.clear()
    return MemoryFileSystem()


def _write_run(
    fs: AbstractFileSystem, run_id: str, *, evaluated: bool, registered: bool
) -> None:
    """One run directory in whichever of the three states is wanted."""
    if evaluated:
        fs.pipe(f"{TRAIN_ROOT}{run_id}/evaluate/{paths.EVALUATE_MANIFEST}", b"{}")
    if registered:
        fs.pipe(f"{TRAIN_ROOT}{run_id}/run_output.json", _run_output_json(run_id))
    # Every run reaches compose_configs, so a run that got no further still has
    # a directory and would appear in a listing that did not gate.
    fs.pipe(f"{TRAIN_ROOT}{run_id}/compose_configs/manifest.json", b"{}")


@pytest.fixture
def three_states(fake_fs: AbstractFileSystem) -> dict[str, str]:
    """One run of each state the gate must distinguish, newest id last."""
    ids = {
        "no_evaluate": generate_run_id(),
        "evaluated_only": generate_run_id(),
        "registered": generate_run_id(),
    }
    _write_run(fake_fs, ids["no_evaluate"], evaluated=False, registered=False)
    _write_run(fake_fs, ids["evaluated_only"], evaluated=True, registered=False)
    _write_run(fake_fs, ids["registered"], evaluated=True, registered=True)

    # Self-check: all three directories exist, so an absence below is the gate's
    # doing and not a fixture that never wrote the run.
    listed = {
        entry.rstrip("/").rsplit("/", 1)[-1]
        for entry in fake_fs.ls(TRAIN_ROOT, detail=False)
    }
    assert listed == set(ids.values())
    assert len(set(ids.values())) == len(ids), "generate_run_id collided"
    return ids


# ================================================
# the evaluate-manifest gate
# ================================================


def test_a_run_that_evaluated_without_registering_is_offered(
    fake_fs: AbstractFileSystem, three_states: dict[str, str]
) -> None:
    """The run this monitor exists for: good tables, no completion marker."""
    assert three_states["evaluated_only"] in runs.discover_run_ids(TRAIN_ROOT, fake_fs)


def test_a_run_with_no_evaluate_output_is_absent(
    fake_fs: AbstractFileSystem, three_states: dict[str, str]
) -> None:
    """Selecting it would make every loader below raise on a missing object."""
    assert three_states["no_evaluate"] not in runs.discover_run_ids(TRAIN_ROOT, fake_fs)


def test_the_gate_reads_the_step_marker_not_the_pipeline_marker(
    fake_fs: AbstractFileSystem, three_states: dict[str, str]
) -> None:
    """Both markers present and only one absent must give the same answer."""
    offered = runs.discover_run_ids(TRAIN_ROOT, fake_fs)
    assert set(offered) == {three_states["registered"], three_states["evaluated_only"]}


def test_an_environment_with_no_evaluated_run_lists_none(
    fake_fs: AbstractFileSystem,
) -> None:
    """Legitimate for a fresh environment, so it returns empty rather than raising."""
    assert runs.discover_run_ids(TRAIN_ROOT, fake_fs) == ()


# ================================================
# ordering
# ================================================


def test_generated_ids_order_chronologically_despite_differing_case(
    fake_fs: AbstractFileSystem,
) -> None:
    """The same-date pair an ASCII sort inverts, uppercase carrying the later time."""
    pair = (EARLIER_LOWERCASE, LATER_UPPERCASE)
    # Self-check: the pair must distinguish the two sorts, or a case-sensitive
    # sort passes this test too and the fixture has quietly stopped working.
    assert sorted(pair, reverse=True) != sorted(pair, key=str.lower, reverse=True)

    for run_id in (EARLIER_LOWERCASE, LATER_UPPERCASE):
        _write_run(fake_fs, run_id, evaluated=True, registered=False)

    assert runs.discover_run_ids(TRAIN_ROOT, fake_fs) == (
        LATER_UPPERCASE,
        EARLIER_LOWERCASE,
    )


def test_a_hand_supplied_run_id_is_offered_without_a_guaranteed_position(
    fake_fs: AbstractFileSystem,
) -> None:
    """A --run-id override is only min_length checked, so it sorts somewhere; the
    guarantee is that it stays selectable."""
    for run_id in (EARLIER_LOWERCASE, LATER_UPPERCASE, HAND_SUPPLIED):
        _write_run(fake_fs, run_id, evaluated=True, registered=False)

    offered = runs.discover_run_ids(TRAIN_ROOT, fake_fs)
    assert HAND_SUPPLIED in offered
    assert offered.index(LATER_UPPERCASE) < offered.index(EARLIER_LOWERCASE)


# ================================================
# the pointer and the default selection
# ================================================


def test_the_pointer_names_the_run_the_page_opens(
    fake_fs: AbstractFileSystem, three_states: dict[str, str]
) -> None:
    """Authoritative for "the current run", and one object read."""
    fake_fs.pipe(POINTER_URI, _pointer_json(three_states["evaluated_only"]))

    pointed = runs.read_pointer_run_id(POINTER_URI, fake_fs)
    offered = runs.discover_run_ids(TRAIN_ROOT, fake_fs)
    assert runs.default_run_id(offered, pointed) == three_states["evaluated_only"]


def test_an_unwritten_pointer_slice_reads_as_no_run(
    fake_fs: AbstractFileSystem,
) -> None:
    """`"unset"` is a string where a record is expected, so `.get` is not enough."""
    fake_fs.pipe(POINTER_URI, _pointer_json(None))
    assert runs.read_pointer_run_id(POINTER_URI, fake_fs) is None


def test_a_missing_pointer_reads_as_no_run(fake_fs: AbstractFileSystem) -> None:
    """An environment whose pointer was never written must not raise here."""
    assert runs.read_pointer_run_id(POINTER_URI, fake_fs) is None


def test_the_default_falls_back_when_the_pointer_names_an_unlisted_run(
    fake_fs: AbstractFileSystem, three_states: dict[str, str]
) -> None:
    """The pointer can name a run whose evaluate output is gone, which the
    dropdown cannot offer."""
    fake_fs.pipe(POINTER_URI, _pointer_json(three_states["no_evaluate"]))

    pointed = runs.read_pointer_run_id(POINTER_URI, fake_fs)
    offered = runs.discover_run_ids(TRAIN_ROOT, fake_fs)
    assert pointed == three_states["no_evaluate"]
    assert runs.default_run_id(offered, pointed) == offered[0]


def test_no_offered_runs_yields_no_default() -> None:
    """Pure, so the empty environment needs no filesystem at all."""
    assert runs.default_run_id((), None) is None


# ================================================
# expiry of the cached reads
# ================================================


def test_a_new_run_and_its_pointer_are_seen_together_once_the_cache_expires(
    fake_fs: AbstractFileSystem, cache_clock: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Otherwise a long-running server never offers a run evaluated after its first
    read, and a pointer expiring alone names a run the listing lacks."""
    monkeypatch.setattr(runs.gcsfs, "GCSFileSystem", MemoryFileSystem)
    first, second = generate_run_id(), generate_run_id()
    assert first != second, "generate_run_id collided"
    _write_run(fake_fs, first, evaluated=True, registered=True)
    fake_fs.pipe(POINTER_URI, _pointer_json(first))
    assert runs.list_run_ids(TRAIN_ROOT) == (first,)
    assert runs.pointer_run_id(POINTER_URI) == first

    _write_run(fake_fs, second, evaluated=True, registered=True)
    fake_fs.pipe(POINTER_URI, _pointer_json(second))
    # Short of expiry both still read stale: the cache is in play, and neither
    # read expires ahead of the other.
    cache_clock.return_value = load.MUTABLE_TTL_SECONDS - 1
    assert second not in runs.list_run_ids(TRAIN_ROOT)
    assert runs.pointer_run_id(POINTER_URI) == first

    cache_clock.return_value = load.MUTABLE_TTL_SECONDS + 1
    assert second in runs.list_run_ids(TRAIN_ROOT)
    assert runs.pointer_run_id(POINTER_URI) == second
