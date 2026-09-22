import google.auth
import pytest
from google.auth.compute_engine import _metadata
from google.auth.exceptions import DefaultCredentialsError


def test_a_test_finds_no_gcp_credentials() -> None:
    """A test missing a mock fails as it would in CI, not by reaching the project."""
    with pytest.raises(DefaultCredentialsError):
        google.auth.default()


def test_google_auth_took_the_metadata_settings() -> None:
    """Private, but read once at import: an earlier importer voids both silently."""
    assert _metadata._NO_GCE_CHECK
    assert _metadata._GCE_METADATA_HOST == "127.0.0.1:1"
