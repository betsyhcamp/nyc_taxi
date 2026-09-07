from __future__ import annotations

import pytest
from pydantic import ValidationError

from fcstnyctaxi.schemas.run_identity import FeatureRunIdentity, TrainRunIdentity

_VALID_SHA256 = "deadbeef" * 8  # 64 lowercase hex, the shape hexdigest() returns


@pytest.fixture
def valid_train_identity() -> dict:
    """A complete, valid TrainRunIdentity dict. Each test gets a fresh copy."""
    return {
        "git_hash": "a1b2c3d",
        "feature_run_id": "UTC20260905T120000000000Z",
        "train_run_id": "UTC20260905T130000000000Z",
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
        sql_sha256=_VALID_SHA256,
    )

    assert set(FeatureRunIdentity.model_fields) == {
        "git_hash",
        "feature_run_id",
        "sql_sha256",
    }
    assert identity.feature_run_id == "UTC20260905T120000000000Z"


def test_sql_text_as_sql_sha256_raises() -> None:
    """PreparedSql carries sql_text and sha256 as adjacent str fields.

    Passing the wrong one would stamp an entire query as provenance, and a
    min_length constraint would not notice. This is the same shape of slip the
    gs:// patterns close for panel_uri and calendar_uri.
    """
    with pytest.raises(ValidationError, match="sql_sha256"):
        FeatureRunIdentity(
            git_hash="a1b2c3d",
            feature_run_id="UTC20260905T120000000000Z",
            sql_sha256="SELECT ds, unique_id, y FROM `nyc-taxi-ehc.curated.fact`",
        )


@pytest.mark.parametrize(
    "bad_digest",
    [
        pytest.param(_VALID_SHA256[:-1], id="truncated"),
        pytest.param(_VALID_SHA256 + "0", id="too_long"),
        pytest.param(_VALID_SHA256.upper(), id="uppercase"),
        pytest.param(f"sha256:{_VALID_SHA256}", id="manifest_prefixed"),
    ],
)
def test_malformed_sql_sha256_raises(bad_digest: str) -> None:
    """One field, one spelling.

    Uppercase is rejected although it names the same digest, because two
    spellings would compare unequal as strings in a field whose whole job is
    identity. The `sha256:` prefixed form is the convention used for config-file
    hashes in each run's emitted manifest.json, and is deliberately not a second
    accepted form here.
    """
    with pytest.raises(ValidationError, match="sql_sha256"):
        FeatureRunIdentity(
            git_hash="a1b2c3d",
            feature_run_id="UTC20260905T120000000000Z",
            sql_sha256=bad_digest,
        )
