import pytest

from fcstnyctaxi.lib.container_images import (
    artifact_registry_prefix,
    require_digest_ref,
)

REPO = "us-central1-docker.pkg.dev/nyc-taxi-ehc/fcst-ml-containers"
FLAT = f"{REPO}/train@sha256:" + "b" * 64
# Nested names are legal in Artifact Registry; the prefix is still three segments.
NESTED = f"{REPO}/team/train@sha256:" + "c" * 64
SOURCE = "FCST_TRAIN_IMAGE"


@pytest.mark.parametrize("image_ref", [FLAT, NESTED], ids=["flat", "nested"])
def test_require_digest_ref_accepts_a_digest_pinned_reference(image_ref: str) -> None:
    """Test that a well-formed reference is returned unchanged, nesting included."""
    assert require_digest_ref(image_ref, SOURCE) == image_ref


@pytest.mark.parametrize(
    ("image_ref", "expected_message"),
    [
        (None, "not set"),
        ("", "not set"),
        (REPO + "/train:abc1234", "64 hex"),
        (REPO + "/train@sha256:zz", "64 hex"),
        (REPO + "/train@sha256:", "64 hex"),
        (REPO + "/@sha256:" + "c" * 64, "64 hex"),
        ("a/b/c@sha256:" + "c" * 64, "64 hex"),
    ],
    ids=[
        "unset",
        "empty",
        "tag",
        "digest-not-hex",
        "digest-empty",
        "no-name",
        "no-host",
    ],
)
def test_require_digest_ref_rejects(
    image_ref: str | None, expected_message: str
) -> None:
    """Test that anything but a digest-pinned reference is refused: a tag is the
    likely mistake, the rest an "@sha256:" substring test would allow."""
    with pytest.raises(ValueError, match=expected_message):
        require_digest_ref(image_ref, SOURCE)


@pytest.mark.parametrize(
    "image_ref", [None, REPO + "/train:abc1234"], ids=["unset", "tag"]
)
def test_require_digest_ref_names_the_caller_source(image_ref: str | None) -> None:
    """Test that the caller's label reaches every message, so a hardcoded name
    cannot put back the environment this module was moved out of."""
    with pytest.raises(ValueError, match="MY_OWN_IMAGE"):
        require_digest_ref(image_ref, "MY_OWN_IMAGE")


@pytest.mark.parametrize("image_ref", [FLAT, NESTED], ids=["flat", "nested"])
def test_artifact_registry_prefix_stops_at_the_repository(image_ref: str) -> None:
    """Test the prefix is three segments at any depth: a last-segment parse passes on
    `flat`, deepens on `nested`, and a sibling stops matching, so the check is inert."""
    assert artifact_registry_prefix(image_ref) == REPO


def test_artifact_registry_prefix_rejects_a_malformed_reference() -> None:
    """Test that a bad argument raises ValueError naming `image_ref`: routing through
    `require_digest_ref` would name a variable this caller was never given."""
    with pytest.raises(ValueError, match="image_ref"):
        artifact_registry_prefix("not-an-image")
