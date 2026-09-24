import os

os.environ.setdefault(
    "FCST_TRAIN_IMAGE",
    "us-central1-docker.pkg.dev/test-project/fcst-ml-containers/train@sha256:"
    + "a" * 64,
)

from pathlib import Path
from typing import Any, cast

import pytest
from kfp.dsl import Artifact
from kfp.dsl.python_component import PythonComponent
from pytest_mock import MockerFixture

from fcstnyctaxi.components.train.evaluate_component import evaluate
from fcstnyctaxi.core.train.evaluate_impl import EvaluateSummary

# dsl.component has no return annotation, so a checker reads .execute() as unknown.
COMPONENT = cast(PythonComponent, evaluate)

RUN_PREFIX = "gs://sentinel-bucket/dev/train/t-sentinel/"
GIT_HASH = "abc1234-dirty"

BENCHMARK_MODEL = "model_a"
CHALLENGER_MODEL = "model_b"

# Distinct, so a transposed pair fails the pairing assertions; equal URIs would be
# blind to it by construction.
COMPOSED_CONFIGS_URI = f"{RUN_PREFIX}compose_configs/"
CHALLENGER_SIDECAR_URI = f"{RUN_PREFIX}backtest/{CHALLENGER_MODEL}/"
BENCHMARK_SIDECAR_URI = f"{RUN_PREFIX}backtest/{BENCHMARK_MODEL}/"

# Derived, never a literal: a literal would restate the convention, not check it.
SCORES_URI = f"{RUN_PREFIX}evaluate/"

# Real, not a Mock: a MagicMock unpacks to {}, so the metadata loop checks nothing.
# The model names reach the wrapper only on this summary.
SUMMARY = EvaluateSummary(
    train_run_id="t-sentinel",
    challenger_model=CHALLENGER_MODEL,
    benchmark_model=BENCHMARK_MODEL,
    feature_run_id="f-sentinel",
    n_origins=1,
    first_origin="2025-05-18",
    last_origin="2025-05-18",
    n_series=3,
    n_folds_total=2,
    hero_metric_name="wrmae_pooled",
    hero_metric_values={"horizon_1": 0.92, "horizon_2": 0.88},
    output_rows={
        "per_series_comparison.parquet": 6,
        "fold_metrics.parquet": 4,
        "period_metrics.parquet": 2,
        "summary_metrics.parquet": 2,
    },
)


@pytest.fixture
def mock_impl(mocker: MockerFixture) -> Any:
    """evaluate_impl replaced at its source module, as the wrapper imports it."""
    mock = mocker.patch("fcstnyctaxi.core.train.evaluate_impl.evaluate_impl")
    mock.return_value = SUMMARY
    return mock


@pytest.fixture
def baked_git_hash(monkeypatch: pytest.MonkeyPatch) -> str:
    """The value Dockerfile.train bakes as ENV FCST_GIT_HASH."""
    monkeypatch.setenv("FCST_GIT_HASH", GIT_HASH)
    return GIT_HASH


def _artifacts() -> tuple[Artifact, Artifact, Artifact, Artifact]:
    """Fresh inputs and output; scores starts at uri="" so .path reads ""."""
    return (
        Artifact(name="composed_configs", uri=COMPOSED_CONFIGS_URI),
        Artifact(name="challenger_sidecar", uri=CHALLENGER_SIDECAR_URI),
        Artifact(name="benchmark_sidecar", uri=BENCHMARK_SIDECAR_URI),
        Artifact(name="scores", uri=""),
    )


def test_wrapper_places_its_scores_pairs_every_input_and_stamps_it(
    mock_impl: Any, baked_git_hash: str
) -> None:
    """Test that the wrapper places its scores, pairs each input, and stamps it."""
    composed_configs, challenger, benchmark, scores = _artifacts()
    assert scores.path == ""  # baseline: demonstrably wrong until assigned

    COMPONENT.execute(
        run_prefix=RUN_PREFIX,
        composed_configs=composed_configs,
        challenger_sidecar=challenger,
        benchmark_sidecar=benchmark,
        scores=scores,
    )

    assert scores.uri == SCORES_URI

    kwargs = mock_impl.call_args.kwargs
    # Through KFP rather than hardcoding /gcs/: tests our ordering, not KFP's mount.
    assert kwargs["out_dir"] == Path(Artifact(uri=SCORES_URI).path)
    # The role pairing, which is this PR's hazard: a transposition inverts every
    # skill ratio and reports a complete, plausible table.
    assert kwargs["challenger_dir"] == Path(challenger.path)
    assert kwargs["benchmark_dir"] == Path(benchmark.path)
    assert kwargs["compose_configs_dir"] == Path(composed_configs.path)

    # Every summary field reaches metadata, without pinning which fields exist.
    for key, value in SUMMARY.as_dict().items():
        assert scores.metadata[key] == value
    assert scores.metadata["git_hash"] == GIT_HASH


@pytest.mark.parametrize("baked_value", [None, ""])
def test_missing_git_hash_raises_naming_the_variable(
    mock_impl: Any, monkeypatch: pytest.MonkeyPatch, baked_value: str | None
) -> None:
    """Test that absent or empty FCST_GIT_HASH is refused by name, before any work."""
    if baked_value is None:
        monkeypatch.delenv("FCST_GIT_HASH", raising=False)
    else:
        monkeypatch.setenv("FCST_GIT_HASH", baked_value)

    composed_configs, challenger, benchmark, scores = _artifacts()

    with pytest.raises(RuntimeError, match="FCST_GIT_HASH"):
        COMPONENT.execute(
            run_prefix=RUN_PREFIX,
            composed_configs=composed_configs,
            challenger_sidecar=challenger,
            benchmark_sidecar=benchmark,
            scores=scores,
        )

    mock_impl.assert_not_called()


def test_impl_failure_propagates_and_leaves_metadata_unstamped(
    mock_impl: Any, baked_git_hash: str
) -> None:
    """Test that an impl failure reaches the caller and stamps no metadata."""
    mock_impl.side_effect = ValueError("out_dir must be named 'evaluate'")
    composed_configs, challenger, benchmark, scores = _artifacts()

    with pytest.raises(ValueError, match="must be named"):
        COMPONENT.execute(
            run_prefix=RUN_PREFIX,
            composed_configs=composed_configs,
            challenger_sidecar=challenger,
            benchmark_sidecar=benchmark,
            scores=scores,
        )

    assert scores.metadata == {}
    # Not "untouched": .uri is assigned before the impl call, so asserting a pristine
    # artifact would assert the ordering bug back in.
    assert scores.uri == SCORES_URI
