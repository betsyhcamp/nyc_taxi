import logging
import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from pytest_mock import MockerFixture

from fcstnyctaxi.lib import run_outputs
from fcstnyctaxi.lib.run_outputs import read_feature_run_outputs
from fcstnyctaxi.lib.storage_layout import (
    resolve_run_outputs_uri,
    resolve_run_prefix,
)
from fcstnyctaxi.lib.utils import get_project_root_dir
from fcstnyctaxi.schemas.run_outputs import FeatureRunOutputs
from scripts import publish_feature_stand_in

CONFIG_DIR = get_project_root_dir() / "config"
ENV = "dev"
FEATURE_RUN_ID = "f-20260914T000000000000Z"


@pytest.fixture
def source_frames(mocker: MockerFixture) -> None:
    """The three artifacts the stand-in copies. Only the panel is read: the other
    two are copied whole, so neither's columns reach an assertion."""
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
    exog = pd.DataFrame({"week_sin": [0.0, 1.0]})
    mocker.patch.object(
        publish_feature_stand_in.pd,
        "read_parquet",
        side_effect=[panel, calendar, exog],
    )


@pytest.fixture
def durable_writes(
    source_frames: None, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> MagicMock:
    """Every durable write the stand-in makes, on one manager, so a test can assert
    about their order rather than about each call alone."""
    writes = mocker.MagicMock()
    writes.attach_mock(mocker.patch.object(pd.DataFrame, "to_parquet"), "parquet")
    writes.attach_mock(
        mocker.patch.object(publish_feature_stand_in, "write_text_to_gcs"), "manifest"
    )
    writes.attach_mock(
        mocker.patch.object(publish_feature_stand_in, "delete_from_gcs"), "delete"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["publish_feature_stand_in", "--env", ENV, "--feature-run-id", FEATURE_RUN_ID],
    )
    return writes


def test_the_manifest_lands_at_the_run_root_after_every_parquet_write(
    durable_writes: MagicMock,
) -> None:
    """Written first it would be a promise; under the step, it needs the step name."""
    publish_feature_stand_in.main()

    calls = durable_writes.mock_calls
    manifest_index = next(i for i, call in enumerate(calls) if call[0] == "manifest")
    written = {call.args[0] for call in calls[:manifest_index] if call[0] == "parquet"}

    text, uri = calls[manifest_index].args
    # resolve_run_prefix is not patched, so this is real path construction.
    assert uri == resolve_run_outputs_uri(CONFIG_DIR, ENV, "feature", FEATURE_RUN_ID)
    assert f"/{publish_feature_stand_in._STEP}/" not in uri

    outputs = FeatureRunOutputs.model_validate_json(text)
    assert set(outputs.published.model_dump().values()) <= written
    assert outputs.feature_run_id == FEATURE_RUN_ID
    assert outputs.env == ENV
    assert Path(uri).parent.name == FEATURE_RUN_ID


def test_the_completion_marker_is_deleted_before_any_artifact_is_rewritten(
    durable_writes: MagicMock,
) -> None:
    """A republish that fails partway would otherwise leave an earlier manifest
    marking the run complete over artifacts it half replaced."""
    publish_feature_stand_in.main()

    calls = durable_writes.mock_calls
    names = [call[0] for call in calls]
    assert "delete" in names

    deleted = names.index("delete")
    assert all(i > deleted for i, name in enumerate(names) if name != "delete")
    # The URI the manifest is written to, so deleting some other object cannot pass
    # for having cleared the marker.
    assert calls[deleted].args[0] == calls[names.index("manifest")].args[1]


def test_the_manifest_names_the_exogenous_artifact_at_its_run_scoped_path(
    durable_writes: MagicMock,
) -> None:
    """A URI under another run, or the calendar's, publishes the wrong bytes under
    the right name."""
    publish_feature_stand_in.main()

    # resolve_run_prefix is not patched, so this is real path construction.
    run_prefix = resolve_run_prefix(CONFIG_DIR, ENV, "feature", FEATURE_RUN_ID)
    outputs = FeatureRunOutputs.model_validate_json(
        durable_writes.manifest.call_args.args[0]
    )
    assert outputs.published.exogenous_uri == (
        f"{run_prefix}{publish_feature_stand_in._STEP}/"
        f"{publish_feature_stand_in._EXOG_FILENAME}"
    )
    # The assertion above builds its filename from the same constant the code does,
    # so a constant pointing at another artifact's name would satisfy it. Two roles
    # sharing a URI means one artifact was published over the other.
    published = outputs.published.model_dump()
    assert len(set(published.values())) == len(published)


def test_what_the_stand_in_writes_reads_back_with_no_version_warning(
    durable_writes: MagicMock,
    mocker: MockerFixture,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Binds two constants nothing else does: the writer's _SCHEMA_VERSION and the
    reader's expectation drift apart silently, because a mismatch only warns."""
    publish_feature_stand_in.main()

    # Patched on the importing module, which imports at module scope. Picked wrong,
    # the failure is a credentials error in CI rather than an assertion failure.
    mocker.patch.object(
        run_outputs,
        "read_text_from_gcs",
        return_value=durable_writes.manifest.call_args.args[0],
    )
    with caplog.at_level(logging.WARNING):
        outputs = read_feature_run_outputs(
            config_dir=CONFIG_DIR, env=ENV, feature_run_id=FEATURE_RUN_ID
        )

    assert outputs.published.panel_uri.endswith("time_series.parquet")
    assert outputs.published.calendar_uri.endswith("fiscal_calendar.parquet")
    assert "schema_version" not in caplog.text
