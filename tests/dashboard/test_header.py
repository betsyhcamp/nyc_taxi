from typing import Any

import pytest
import yaml

from dashboard.train import header
from fcstnyctaxi.core.train.evaluate_impl import (
    EvaluateSummary,
    _build_manifest,
)
from fcstnyctaxi.lib.utils import get_project_root_dir
from fcstnyctaxi.schemas.config.train import TrainModelingConfig
from fcstnyctaxi.schemas.run_identity import TrainRunIdentity

RUN_ID = "20260929t045735959418z"
FEATURE_RUN_ID = "20260923t011300349443z"
MODEL_TAG = "projects/1/locations/us-central1/models/model-a@11"


@pytest.fixture
def identity() -> TrainRunIdentity:
    """One run's provenance, as `compose_configs` writes it."""
    base = f"gs://BUCKET/dev/feature/{FEATURE_RUN_ID}/data_prep"
    return TrainRunIdentity(
        git_hash="0" * 40,
        feature_run_id=FEATURE_RUN_ID,
        train_run_id=RUN_ID,
        panel_uri=f"{base}/time_series.parquet",
        calendar_uri=f"{base}/fiscal_calendar.parquet",
        additional_exog_uri=f"{base}/exogenous_features.parquet",
    )


@pytest.fixture
def manifest(identity: TrainRunIdentity) -> dict[str, Any]:
    """`evaluate_manifest.json` as the evaluate step emits it."""
    modeling = TrainModelingConfig.model_validate(
        yaml.safe_load(
            (get_project_root_dir() / "config/train/modeling.yaml").read_text()
        )
    )
    summary = EvaluateSummary(
        train_run_id=RUN_ID,
        challenger_model=modeling.model_roles.challenger,
        benchmark_model=modeling.model_roles.benchmark,
        feature_run_id=FEATURE_RUN_ID,
        n_origins=30,
        first_origin="2025-03-23",
        last_origin="2025-10-12",
        n_series=68,
        n_folds_total=56,
        hero_metric_name="wrmae_pooled",
        hero_metric_values={"horizon_1": 0.83, "horizon_2": 0.81},
        output_rows={"summary_metrics.parquet": 144},
    )
    built = _build_manifest(summary, identity, modeling)
    # Self-check: the producer really nests these two, so a flat read below would
    # be the strip's bug and not the fixture's.
    assert "train_run_id" in built["lineage"]
    assert "n_origins" in built["origins"]
    # And the two roles must differ, or a transposed pair reads identically and
    # the role test passes vacuously. `ModelRoles` permits one model in both as a
    # smoke test, and this reads the committed config, so that is reachable.
    assert built["challenger_model"] != built["benchmark_model"]
    return built


# ================================================
# the identity fields
# ================================================


def test_the_strip_reads_a_real_manifest_without_a_gap(
    manifest: dict[str, Any],
) -> None:
    """Built by the producer, so a key path the manifest lacks raises here rather
    than on the page."""
    assert header.identity_fields(manifest)


def test_the_strip_draws_every_field_it_reads(manifest: dict[str, Any]) -> None:
    """A field read and not drawn is one the page silently omits."""
    drawn = {header._HEAD, *(label for row in header._ROWS for label in row)}
    assert set(header.identity_fields(manifest)) == drawn


def test_the_nested_fields_are_read_from_their_nesting(
    manifest: dict[str, Any],
) -> None:
    """`feature_run_id` sits under `lineage` and the origins under `origins`,
    where a top-level read would silently find nothing."""
    fields = header.identity_fields(manifest)
    assert fields["feature run"] == FEATURE_RUN_ID
    assert fields["first origin"] == manifest["origins"]["first_origin"]
    assert fields["origins"] == str(manifest["origins"]["n_origins"])


def test_the_two_model_roles_are_not_crossed(manifest: dict[str, Any]) -> None:
    """A transposed pair would invert every skill number silently, and the
    benchmark is named at all because the one in code is not the charter's."""
    fields = header.identity_fields(manifest)
    assert fields["challenger"] == manifest["challenger_model"]
    assert fields["benchmark"] == manifest["benchmark_model"]


def test_a_manifest_missing_a_gated_field_raises(manifest: dict[str, Any]) -> None:
    """The strip gates every number below it, so a gap must stop the page."""
    del manifest["n_series"]
    with pytest.raises(ValueError, match="series"):
        header.identity_fields(manifest)


def test_the_run_over_run_fields_stay_out(manifest: dict[str, Any]) -> None:
    """`git_hash` and the input URIs are in the manifest and deliberately unread:
    their value was comparison, which this version does not do."""
    printed = set(header.identity_fields(manifest).values())
    assert manifest["lineage"]["git_hash"] not in printed
    assert manifest["lineage"]["panel_uri"] not in printed


# ================================================
# the registration badge
# ================================================


def test_the_badge_labels_cannot_be_read_as_each_other() -> None:
    """Swapped, every reader learns the opposite of whether a model shipped, and
    a test comparing against the constants moves with them. Phrasing is free; the
    negative saying so is not."""
    assert "not" in header.NOT_REGISTERED
    assert "not" not in header.REGISTERED


def test_a_run_with_no_completion_marker_badges_as_evaluated() -> None:
    """The run this monitor exists for. Absence is information, not an error."""
    status = header.registration(None)
    assert status.label == header.NOT_REGISTERED
    assert status.model_tag is None


def test_a_registered_run_badges_with_its_model_tag() -> None:
    """What makes the badge worth more than a boolean."""
    status = header.registration(
        {"train_run_id": RUN_ID, "published": {"model_tag": MODEL_TAG}}
    )
    assert status.label == header.REGISTERED
    assert status.model_tag == MODEL_TAG


def test_a_marker_without_a_published_block_still_badges_registered() -> None:
    """Presence is the signal; the tag is a detail that may be absent."""
    status = header.registration({"train_run_id": RUN_ID})
    assert status.label == header.REGISTERED
    assert status.model_tag is None


# ================================================
# the renderer, smoke only
# ================================================


@pytest.mark.parametrize("marker", [None, {"published": {"model_tag": MODEL_TAG}}])
def test_the_strip_renders_in_both_registration_states(
    manifest: dict[str, Any], marker: dict[str, Any] | None
) -> None:
    """No pixel test; this catches a malformed column spec or a missing label."""
    header.render_identity_strip(manifest, marker)
