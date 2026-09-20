import os

os.environ.setdefault(
    "FCST_TRAIN_IMAGE",
    "us-central1-docker.pkg.dev/nyc-taxi-ehc/fcst-ml-containers/train@sha256:"
    + "a" * 64,
)

from pathlib import Path
from typing import Any, cast

import pytest
from kfp.dsl import Artifact, Dataset
from kfp.dsl.python_component import PythonComponent
from pytest_mock import MockerFixture

from fcstnyctaxi.components.train.backtest_component import backtest
from fcstnyctaxi.core.train.backtest_impl import BacktestSummary

# dsl.component has no return annotation, so a checker sees the undecorated function
# and .execute() reads as unknown. Cast once rather than at each call site.
COMPONENT = cast(PythonComponent, backtest)

RUN_PREFIX = "gs://sentinel-bucket/dev/train/t-sentinel/"
MODEL_NAME = "model_a"
GIT_HASH = "abc1234-dirty"

# Distinct, so the pairing assertions also prove no two inputs were swapped.
COMPOSED_CONFIGS_URI = f"{RUN_PREFIX}compose_configs/"
PANEL_URI = "gs://sentinel-bucket/dev/feature/f-sentinel/panel.parquet"
CALENDAR_URI = "gs://sentinel-bucket/dev/feature/f-sentinel/calendar.parquet"

# Derived from run_prefix, never a literal: the convention is the wrapper's, and a
# hardcoded string here would restate it rather than check it.
SIDECAR_URI = f"{RUN_PREFIX}backtest/{MODEL_NAME}/"

# Real, not a Mock: the wrapper feeds as_dict() to metadata.update(), which raises
# on a Mock for a reason unrelated to anything under test.
SUMMARY = BacktestSummary(
    n_origins=3,
    first_origin="2025-04-20",
    last_origin="2025-06-15",
    n_series=7,
    feature_run_id="f-sentinel",
    output_rows={"metrics.parquet": 12, "monthly_series.parquet": 84},
)


@pytest.fixture
def mock_impl(mocker: MockerFixture) -> Any:
    """backtest_impl replaced at its source module, as the wrapper imports it."""
    mock = mocker.patch("fcstnyctaxi.core.train.backtest_impl.backtest_impl")
    mock.return_value = SUMMARY
    return mock


@pytest.fixture
def baked_git_hash(monkeypatch: pytest.MonkeyPatch) -> str:
    """The value Dockerfile.train bakes as ENV FCST_GIT_HASH."""
    monkeypatch.setenv("FCST_GIT_HASH", GIT_HASH)
    return GIT_HASH


def _artifacts() -> tuple[Artifact, Dataset, Dataset, Artifact]:
    """Fresh inputs and output; sidecar starts at uri="" so .path reads ""."""
    return (
        Artifact(name="composed_configs", uri=COMPOSED_CONFIGS_URI),
        Dataset(name="panel", uri=PANEL_URI),
        Dataset(name="calendar", uri=CALENDAR_URI),
        Artifact(name="sidecar", uri=""),
    )


def test_wrapper_places_its_sidecar_pairs_every_input_and_stamps_it(
    mock_impl: Any, baked_git_hash: str
) -> None:
    """Test that the wrapper places its sidecar, pairs each input, and stamps it."""
    composed_configs, panel, calendar, sidecar = _artifacts()
    assert sidecar.path == ""  # baseline: demonstrably wrong until assigned

    COMPONENT.execute(
        run_prefix=RUN_PREFIX,
        model_name=MODEL_NAME,
        composed_configs=composed_configs,
        panel=panel,
        calendar=calendar,
        sidecar=sidecar,
    )

    assert sidecar.uri == SIDECAR_URI

    kwargs = mock_impl.call_args.kwargs
    # Through KFP rather than hardcoding /gcs/: tests our ordering, not KFP's mount.
    assert kwargs["out_dir"] == Path(Artifact(uri=SIDECAR_URI).path)
    assert kwargs["panel_path"] == Path(panel.path)
    assert kwargs["calendar_path"] == Path(calendar.path)
    assert kwargs["compose_configs_dir"] == Path(composed_configs.path)
    assert kwargs["model_name"] == MODEL_NAME

    # Every summary field reaches metadata, without pinning which fields exist.
    for key, value in SUMMARY.as_dict().items():
        assert sidecar.metadata[key] == value
    assert sidecar.metadata["model_name"] == MODEL_NAME
    assert sidecar.metadata["git_hash"] == GIT_HASH


@pytest.mark.parametrize("baked_value", [None, ""])
def test_missing_git_hash_raises_naming_the_variable(
    mock_impl: Any, monkeypatch: pytest.MonkeyPatch, baked_value: str | None
) -> None:
    """Test that absent or empty FCST_GIT_HASH is refused by name, before any work."""
    if baked_value is None:
        monkeypatch.delenv("FCST_GIT_HASH", raising=False)
    else:
        monkeypatch.setenv("FCST_GIT_HASH", baked_value)

    composed_configs, panel, calendar, sidecar = _artifacts()

    with pytest.raises(RuntimeError, match="FCST_GIT_HASH"):
        COMPONENT.execute(
            run_prefix=RUN_PREFIX,
            model_name=MODEL_NAME,
            composed_configs=composed_configs,
            panel=panel,
            calendar=calendar,
            sidecar=sidecar,
        )

    mock_impl.assert_not_called()


def test_impl_failure_propagates_and_leaves_metadata_unstamped(
    mock_impl: Any, baked_git_hash: str
) -> None:
    """Test that an impl failure reaches the caller and stamps no metadata."""
    mock_impl.side_effect = ValueError("out_dir must be named for its model")
    composed_configs, panel, calendar, sidecar = _artifacts()

    with pytest.raises(ValueError, match="named for its model"):
        COMPONENT.execute(
            run_prefix=RUN_PREFIX,
            model_name=MODEL_NAME,
            composed_configs=composed_configs,
            panel=panel,
            calendar=calendar,
            sidecar=sidecar,
        )

    assert sidecar.metadata == {}
    # Not "untouched": .uri is assigned before the impl call, so asserting a pristine
    # artifact would assert the ordering bug back in.
    assert sidecar.uri == SIDECAR_URI
