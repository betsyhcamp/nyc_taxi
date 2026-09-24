import json
import logging

import pytest
from pydantic import ValidationError
from pytest_mock import MockerFixture

from fcstnyctaxi.lib import run_outputs
from fcstnyctaxi.lib.run_outputs import (
    _EXPECTED_SCHEMA_VERSION,
    read_feature_run_outputs,
    resolve_feature_artifacts,
)
from fcstnyctaxi.lib.utils import get_project_root_dir

CONFIG_DIR = get_project_root_dir() / "config"
ENV = "dev"
FEATURE_RUN_ID = "f-20260914T000000000000Z"
PANEL_URI = f"gs://BUCKET/{ENV}/feature/{FEATURE_RUN_ID}/step/time_series.parquet"
CALENDAR_URI = (
    f"gs://BUCKET/{ENV}/feature/{FEATURE_RUN_ID}/step/fiscal_calendar.parquet"
)
EXOG_URI = f"gs://BUCKET/{ENV}/feature/{FEATURE_RUN_ID}/step/exogenous_features.parquet"


def _manifest(**overrides: object) -> str:
    """A manifest as the producer writes it, with any key replaced; None removes it."""
    fields: dict[str, object] = {
        "schema_version": _EXPECTED_SCHEMA_VERSION,
        "feature_run_id": FEATURE_RUN_ID,
        "env": ENV,
        "published": {
            "panel_uri": PANEL_URI,
            "calendar_uri": CALENDAR_URI,
            "exogenous_uri": EXOG_URI,
        },
    }
    fields.update(overrides)
    return json.dumps({k: v for k, v in fields.items() if v is not None})


def test_a_missing_manifest_names_a_completion_failure(mocker: MockerFixture) -> None:
    """Presence is the completion signal, so a 404 on a path the operator never
    constructed is the wrong thing to report."""
    mocker.patch.object(
        run_outputs, "read_text_from_gcs", side_effect=FileNotFoundError
    )

    with pytest.raises(ValueError, match="did not complete"):
        read_feature_run_outputs(
            config_dir=CONFIG_DIR, env=ENV, feature_run_id=FEATURE_RUN_ID
        )


def test_a_declared_id_mismatch_is_refused(mocker: MockerFixture) -> None:
    """A manifest copied between run directories otherwise reads as authoritative."""
    mocker.patch.object(
        run_outputs,
        "read_text_from_gcs",
        return_value=_manifest(feature_run_id="f-someone-elses-run"),
    )

    with pytest.raises(ValueError, match="copied between runs"):
        read_feature_run_outputs(
            config_dir=CONFIG_DIR, env=ENV, feature_run_id=FEATURE_RUN_ID
        )


def test_an_env_mismatch_is_refused_but_an_absent_env_is_not(
    mocker: MockerFixture,
) -> None:
    """Both halves of the optional guard: checked when present, never required."""
    transport = mocker.patch.object(
        run_outputs, "read_text_from_gcs", return_value=_manifest(env="prod")
    )

    with pytest.raises(ValueError, match="declares env"):
        read_feature_run_outputs(
            config_dir=CONFIG_DIR, env=ENV, feature_run_id=FEATURE_RUN_ID
        )

    transport.return_value = _manifest(env=None)
    assert (
        read_feature_run_outputs(
            config_dir=CONFIG_DIR, env=ENV, feature_run_id=FEATURE_RUN_ID
        ).env
        is None
    )


_OVERRIDE_URIS = {
    "panel_uri": PANEL_URI,
    "calendar_uri": CALENDAR_URI,
    "additional_exog_uri": EXOG_URI,
}


@pytest.mark.parametrize(
    "supplied",
    [
        ("panel_uri",),
        ("calendar_uri",),
        ("additional_exog_uri",),
        ("panel_uri", "calendar_uri"),
        ("panel_uri", "additional_exog_uri"),
        ("calendar_uri", "additional_exog_uri"),
    ],
    ids="+".join,
)
def test_a_partial_set_of_uri_flags_is_refused(
    supplied: tuple[str, ...], mocker: MockerFixture
) -> None:
    """Artifacts from different sources would be recorded nowhere."""
    # So a partial set the guard lets through fails the assertion, not on credentials.
    mocker.patch.object(run_outputs, "read_text_from_gcs", return_value=_manifest())
    uris = {
        flag: _OVERRIDE_URIS[flag] if flag in supplied else None
        for flag in _OVERRIDE_URIS
    }

    with pytest.raises(ValueError, match="must be given together"):
        resolve_feature_artifacts(
            config_dir=CONFIG_DIR, env=ENV, feature_run_id=FEATURE_RUN_ID, **uris
        )


def test_a_non_gcs_override_uri_is_refused() -> None:
    """Nothing else checks a hand-typed override until download_from_gcs receives it."""
    with pytest.raises(ValidationError):
        resolve_feature_artifacts(
            config_dir=CONFIG_DIR,
            env=ENV,
            feature_run_id=FEATURE_RUN_ID,
            panel_uri="/tmp/scratch/panel.parquet",
            calendar_uri=CALENDAR_URI,
            additional_exog_uri=EXOG_URI,
        )


def test_the_override_path_reports_its_source(caplog: pytest.LogCaptureFixture) -> None:
    """The log line is the evidence the override flags are to be retired on."""
    with caplog.at_level(logging.INFO):
        artifacts = resolve_feature_artifacts(
            config_dir=CONFIG_DIR,
            env=ENV,
            feature_run_id=FEATURE_RUN_ID,
            **_OVERRIDE_URIS,
        )

    assert artifacts.panel_uri == PANEL_URI
    assert artifacts.exogenous_uri == EXOG_URI
    assert "source=supplied" in caplog.text
    assert EXOG_URI in caplog.text


def test_the_manifest_path_returns_and_logs_the_third_uri(
    mocker: MockerFixture, caplog: pytest.LogCaptureFixture
) -> None:
    """Nothing past the resolver reads the third URI yet, so it is pinned here."""
    mocker.patch.object(run_outputs, "read_text_from_gcs", return_value=_manifest())

    with caplog.at_level(logging.INFO):
        artifacts = resolve_feature_artifacts(
            config_dir=CONFIG_DIR,
            env=ENV,
            feature_run_id=FEATURE_RUN_ID,
            panel_uri=None,
            calendar_uri=None,
            additional_exog_uri=None,
        )

    assert artifacts.exogenous_uri == EXOG_URI
    assert "source=resolved" in caplog.text
    assert EXOG_URI in caplog.text


def test_a_two_uri_manifest_is_refused(mocker: MockerFixture) -> None:
    """A run published before the third artifact existed cannot resolve without it."""
    two_uris = {"panel_uri": PANEL_URI, "calendar_uri": CALENDAR_URI}
    mocker.patch.object(
        run_outputs, "read_text_from_gcs", return_value=_manifest(published=two_uris)
    )

    with pytest.raises(ValidationError, match="exogenous_uri"):
        resolve_feature_artifacts(
            config_dir=CONFIG_DIR,
            env=ENV,
            feature_run_id=FEATURE_RUN_ID,
            panel_uri=None,
            calendar_uri=None,
            additional_exog_uri=None,
        )
