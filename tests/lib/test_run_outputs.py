import json
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError
from pytest_mock import MockerFixture

from fcstnyctaxi.lib import run_outputs
from fcstnyctaxi.lib.run_outputs import (
    _EXPECTED_SCHEMA_VERSION,
    read_feature_run_outputs,
    resolve_feature_artifacts,
)
from fcstnyctaxi.lib.storage_layout import resolve_run_outputs_uri
from fcstnyctaxi.lib.utils import get_project_root_dir
from fcstnyctaxi.schemas.run_outputs import FeatureRunOutputs
from scripts import publish_feature_stand_in

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


@pytest.fixture
def source_frames(mocker: MockerFixture) -> None:
    """The two artifacts the stand-in copies, standing in for the real parquet."""
    panel = pd.DataFrame(
        {
            "unique_id": pd.Series(["a", "a", "b"], dtype="string"),
            # Object dtype is load-bearing: a BigQuery DATE arrives this way and
            # `.min()` then returns a date, so datetime64 here would stop
            # exercising the manifest's `pd.Timestamp` coercion.
            "ds": pd.Series(
                [date(2025, 1, 5), date(2025, 2, 2), date(2025, 1, 5)], dtype="object"
            ),
            "y": [1.0, 2.0, 3.0],
        }
    )
    calendar = pd.DataFrame({"fiscal_year_month": [202501, 202502]})
    mocker.patch.object(
        publish_feature_stand_in.pd, "read_parquet", side_effect=[panel, calendar]
    )


@pytest.mark.xfail(
    strict=True,
    raises=ValidationError,
    reason="the stand-in cannot publish until it writes the exogenous artifact",
)
def test_the_manifest_lands_at_the_run_root_after_both_parquet_writes(
    source_frames: None, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Written first it would be a promise; under the step, it needs the step name."""
    # One manager, so the assertion is about order rather than each call alone.
    writes = mocker.MagicMock()
    writes.attach_mock(mocker.patch.object(pd.DataFrame, "to_parquet"), "parquet")
    writes.attach_mock(
        mocker.patch.object(publish_feature_stand_in, "write_text_to_gcs"), "manifest"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "publish_feature_stand_in",
            "--env",
            ENV,
            "--feature-run-id",
            FEATURE_RUN_ID,
        ],
    )

    publish_feature_stand_in.main()

    assert [call[0] for call in writes.mock_calls] == ["parquet", "parquet", "manifest"]

    text, uri = writes.mock_calls[-1].args
    # resolve_run_prefix is not patched, so this is real path construction.
    assert uri == resolve_run_outputs_uri(CONFIG_DIR, ENV, "feature", FEATURE_RUN_ID)
    assert f"/{publish_feature_stand_in._STEP}/" not in uri

    outputs = FeatureRunOutputs.model_validate_json(text)
    assert outputs.feature_run_id == FEATURE_RUN_ID
    assert outputs.env == ENV
    assert Path(uri).parent.name == FEATURE_RUN_ID


@pytest.mark.xfail(
    strict=True,
    raises=ValidationError,
    reason="the stand-in cannot publish until it writes the exogenous artifact",
)
def test_what_the_stand_in_writes_reads_back_with_no_version_warning(
    source_frames: None,
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Binds two constants nothing else does: the writer's _SCHEMA_VERSION and the
    reader's expectation drift apart silently, because a mismatch only warns."""
    mocker.patch.object(pd.DataFrame, "to_parquet")
    written = mocker.patch.object(publish_feature_stand_in, "write_text_to_gcs")
    monkeypatch.setattr(
        sys,
        "argv",
        ["publish_feature_stand_in", "--env", ENV, "--feature-run-id", FEATURE_RUN_ID],
    )
    publish_feature_stand_in.main()

    # Patched on the importing module, which imports at module scope. Picked wrong,
    # the failure is a credentials error in CI rather than an assertion failure.
    mocker.patch.object(
        run_outputs, "read_text_from_gcs", return_value=written.call_args.args[0]
    )
    with caplog.at_level(logging.WARNING):
        outputs = read_feature_run_outputs(
            config_dir=CONFIG_DIR, env=ENV, feature_run_id=FEATURE_RUN_ID
        )

    assert outputs.published.panel_uri.endswith("time_series.parquet")
    assert outputs.published.calendar_uri.endswith("fiscal_calendar.parquet")
    assert "schema_version" not in caplog.text


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
