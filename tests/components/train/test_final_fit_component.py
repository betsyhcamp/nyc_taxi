import os

# Before the imports, which bind base_image; unset aborts the whole suite.
os.environ.setdefault(
    "FCST_TRAIN_IMAGE",
    "us-central1-docker.pkg.dev/test-project/fcst-ml-containers/train@sha256:"
    + "a" * 64,
)

from pathlib import Path
from typing import Any, cast

import pytest
from kfp.dsl import Artifact, Dataset, Model
from kfp.dsl.python_component import PythonComponent
from pytest_mock import MockerFixture

from fcstnyctaxi.components.train.final_fit_component import final_fit
from fcstnyctaxi.core.train.final_fit_impl import FinalFitSummary

# For the type checker: dsl.component has no return annotation.
COMPONENT = cast(PythonComponent, final_fit)

RUN_PREFIX = "gs://sentinel-bucket/dev/train/t-sentinel/"
MODEL_NAME = "model_a"
GIT_HASH = "abc1234-dirty"

# Distinct, so a panel/calendar swap fails the pairing assertions.
COMPOSED_CONFIGS_URI = f"{RUN_PREFIX}compose_configs/"
PANEL_URI = "gs://sentinel-bucket/dev/feature/f-sentinel/panel.parquet"
CALENDAR_URI = "gs://sentinel-bucket/dev/feature/f-sentinel/calendar.parquet"
EXOG_URI = "gs://sentinel-bucket/dev/feature/f-sentinel/exogenous.parquet"

# Derived: a literal would restate the convention, not check it.
BUNDLE_URI = f"{RUN_PREFIX}final_fit/{MODEL_NAME}/"

# Real, not a Mock: a MagicMock unpacks to {}, so the metadata loop checks nothing.
SUMMARY = FinalFitSummary(
    train_end_ds="2025-06-29",
    n_series=3,
    n_obs=156,
    train_run_id="t-sentinel",
    feature_run_id="f-sentinel",
)


@pytest.fixture
def mock_impl(mocker: MockerFixture) -> Any:
    """final_fit_impl replaced at its source module, as the wrapper imports it."""
    mock = mocker.patch("fcstnyctaxi.core.train.final_fit_impl.final_fit_impl")
    mock.return_value = SUMMARY
    return mock


@pytest.fixture
def baked_git_hash(monkeypatch: pytest.MonkeyPatch) -> str:
    """The value Dockerfile.train bakes as ENV FCST_GIT_HASH."""
    monkeypatch.setenv("FCST_GIT_HASH", GIT_HASH)
    return GIT_HASH


def _artifacts() -> tuple[Artifact, Dataset, Dataset, Dataset, Model]:
    """Fresh inputs and output; bundle starts at uri="" so .path reads ""."""
    return (
        Artifact(name="composed_configs", uri=COMPOSED_CONFIGS_URI),
        Dataset(name="panel", uri=PANEL_URI),
        Dataset(name="calendar", uri=CALENDAR_URI),
        Dataset(name="additional_exog", uri=EXOG_URI),
        Model(name="bundle", uri=""),
    )


def test_wrapper_places_its_bundle_pairs_every_input_and_stamps_it(
    mock_impl: Any, baked_git_hash: str
) -> None:
    """Test that the wrapper places its bundle, pairs each input, and stamps it."""
    composed_configs, panel, calendar, additional_exog, bundle = _artifacts()
    assert bundle.path == ""  # baseline: wrong until assigned

    COMPONENT.execute(
        run_prefix=RUN_PREFIX,
        model_name=MODEL_NAME,
        composed_configs=composed_configs,
        panel=panel,
        calendar=calendar,
        additional_exog=additional_exog,
        bundle=bundle,
    )

    # Only this module guards the step segment: the impl skips it and no DAG test sees
    # a runtime URI, so backtest/ for final_fit/ writes into the sidecar silently.
    assert bundle.uri == BUNDLE_URI

    kwargs = mock_impl.call_args.kwargs
    # Via KFP, not /gcs/: tests our ordering, not KFP's mount.
    assert kwargs["out_dir"] == Path(Model(uri=BUNDLE_URI).path)
    assert kwargs["panel_path"] == Path(panel.path)
    assert kwargs["calendar_path"] == Path(calendar.path)
    assert kwargs["additional_exog_path"] == Path(additional_exog.path)
    assert kwargs["compose_configs_dir"] == Path(composed_configs.path)
    assert kwargs["model_name"] == MODEL_NAME

    # Every summary field, without pinning which exist.
    for key, value in SUMMARY.as_dict().items():
        assert bundle.metadata[key] == value
    assert bundle.metadata["model_name"] == MODEL_NAME
    assert bundle.metadata["git_hash"] == GIT_HASH


@pytest.mark.parametrize("baked_value", [None, ""])
def test_missing_git_hash_raises_naming_the_variable(
    mock_impl: Any, monkeypatch: pytest.MonkeyPatch, baked_value: str | None
) -> None:
    """Test that absent or empty FCST_GIT_HASH is refused by name, before any work."""
    if baked_value is None:
        monkeypatch.delenv("FCST_GIT_HASH", raising=False)
    else:
        monkeypatch.setenv("FCST_GIT_HASH", baked_value)

    composed_configs, panel, calendar, additional_exog, bundle = _artifacts()

    with pytest.raises(RuntimeError, match="FCST_GIT_HASH"):
        COMPONENT.execute(
            run_prefix=RUN_PREFIX,
            model_name=MODEL_NAME,
            composed_configs=composed_configs,
            panel=panel,
            calendar=calendar,
            additional_exog=additional_exog,
            bundle=bundle,
        )

    mock_impl.assert_not_called()


def test_impl_failure_propagates_and_leaves_metadata_unstamped(
    mock_impl: Any, baked_git_hash: str
) -> None:
    """Test that an impl failure reaches the caller and stamps no metadata."""
    mock_impl.side_effect = ValueError("out_dir is not named for model 'model_a'.")
    composed_configs, panel, calendar, additional_exog, bundle = _artifacts()

    with pytest.raises(ValueError, match="not named for model"):
        COMPONENT.execute(
            run_prefix=RUN_PREFIX,
            model_name=MODEL_NAME,
            composed_configs=composed_configs,
            panel=panel,
            calendar=calendar,
            additional_exog=additional_exog,
            bundle=bundle,
        )

    assert bundle.metadata == {}
    # Set before the impl call; a pristine uri would be the ordering bug.
    assert bundle.uri == BUNDLE_URI
