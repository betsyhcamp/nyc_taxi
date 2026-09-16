import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from pytest_mock import MockerFixture

from fcstnyctaxi.lib.storage_layout import resolve_run_outputs_uri
from fcstnyctaxi.lib.utils import get_project_root_dir
from fcstnyctaxi.schemas.run_outputs import FeatureRunOutputs
from scripts import publish_feature_stand_in

CONFIG_DIR = get_project_root_dir() / "config"
ENV = "dev"
FEATURE_RUN_ID = "f-20260914T000000000000Z"


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
