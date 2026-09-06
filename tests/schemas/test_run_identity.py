from __future__ import annotations

import pytest
from pydantic import ValidationError

from fcstnyctaxi.schemas.run_identity import FeatureRunIdentity, TrainRunIdentity


@pytest.fixture
def valid_train_identity() -> dict:
    """A complete, valid TrainRunIdentity dict. Each test gets a fresh copy."""
    return {
        "git_hash": "a1b2c3d",
        "feature_run_id": "UTC20260905T120000000000Z",
        "training_run_id": "UTC20260905T130000000000Z",
        "panel_uri": "gs://b/dev/feature/f1/data_prep/time_series.parquet",
        "calendar_uri": "gs://b/dev/feature/f1/data_prep/fiscal_calendar.parquet",
    }


def test_valid_dict_constructs_train_run_identity(valid_train_identity: dict) -> None:
    """A complete dict constructs TrainRunIdentity."""
    identity = TrainRunIdentity(**valid_train_identity)

    assert identity.feature_run_id == "UTC20260905T120000000000Z"
    assert identity.panel_uri.startswith("gs://")


def test_null_git_hash_raises(valid_train_identity: dict) -> None:
    """get_git_hash() returns None on failure; this is where that must stop.

    A dataclass would accept None, serialise null, and stamp a null column on
    every downstream table — the failure that already produced two registered
    benchmarks whose git_hash cannot reproduce their artifacts.
    """
    valid_train_identity["git_hash"] = None

    with pytest.raises(ValidationError, match="git_hash"):
        TrainRunIdentity(**valid_train_identity)


def test_empty_git_hash_raises(valid_train_identity: dict) -> None:
    """An empty string is as useless as null and is rejected the same way."""
    valid_train_identity["git_hash"] = ""

    with pytest.raises(ValidationError, match="git_hash"):
        TrainRunIdentity(**valid_train_identity)


def test_local_path_as_panel_uri_raises(valid_train_identity: dict) -> None:
    """The runner holds both the URI and the staged local Path.

    Passing the wrong one would stamp /tmp/... as provenance with nothing
    raising, so the gs:// pattern is what closes that slip.
    """
    valid_train_identity["panel_uri"] = "/tmp/scratch/time_series.parquet"

    with pytest.raises(ValidationError, match="panel_uri"):
        TrainRunIdentity(**valid_train_identity)


def test_local_path_as_calendar_uri_raises(valid_train_identity: dict) -> None:
    """Same slip, other frame."""
    valid_train_identity["calendar_uri"] = "/tmp/scratch/fiscal_calendar.parquet"

    with pytest.raises(ValidationError, match="calendar_uri"):
        TrainRunIdentity(**valid_train_identity)


def test_train_run_identity_is_frozen(valid_train_identity: dict) -> None:
    """Recorded provenance is immutable once constructed."""
    identity = TrainRunIdentity(**valid_train_identity)

    with pytest.raises(ValidationError):
        identity.git_hash = "deadbee"  # type: ignore[misc]


def test_feature_run_identity_carries_no_uris() -> None:
    """Feature produces the artifacts, so it cannot carry pointers to them.

    This is why the three models stay flat rather than sharing a base class.
    """
    identity = FeatureRunIdentity(
        git_hash="a1b2c3d",
        feature_run_id="UTC20260905T120000000000Z",
        sql_sha256="0" * 64,
    )

    assert set(FeatureRunIdentity.model_fields) == {
        "git_hash",
        "feature_run_id",
        "sql_sha256",
    }
    assert identity.feature_run_id == "UTC20260905T120000000000Z"
