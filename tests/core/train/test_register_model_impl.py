import json
import re
import shutil
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, call

import pandas as pd
import pytest
import yaml
from google.api_core.exceptions import AlreadyExists, NotFound
from google.cloud.aiplatform.models import VersionInfo
from pytest_mock import MockerFixture
from stand_in_model import STAND_IN_MODEL_NAME, stand_in_config_dir

from fcstnyctaxi.core.train import register_model_impl as impl_module
from fcstnyctaxi.core.train.compose_configs_impl import (
    compose_configs_impl,
    compose_train_static_configs,
)
from fcstnyctaxi.core.train.final_fit_impl import final_fit_impl
from fcstnyctaxi.core.train.register_model_impl import (
    RegisterModelSummary,
    register_model_impl,
)
from fcstnyctaxi.lib.registry_ids import compose_display_name, compose_model_id
from fcstnyctaxi.lib.storage_layout import (
    LATEST_POINTER_FILENAME,
    RUN_OUTPUTS_FILENAME,
    SourcedPath,
    latest_pointer_path,
)
from fcstnyctaxi.lib.utils import get_project_root_dir
from fcstnyctaxi.schemas.config.environment import EnvironmentConfig
from fcstnyctaxi.schemas.config.train import ModelRegistry, TrainInfraConfig
from fcstnyctaxi.schemas.run_identity import TrainRunIdentity
from fcstnyctaxi.schemas.run_outputs import (
    FeatureRunOutputs,
    LatestRunPointer,
    TrainRunOutputs,
)

CONFIG_DIR = get_project_root_dir() / "config"
ENV = "dev"
TRAIN_RUN_ID = "t-20260921t000000000000z"
OTHER_RUN_ID = "t-20260922t000000000000z"
# Feature's format is not ours, so its uppercase stays legal throughout.
FEATURE_RUN_ID = "f-20260920T000000000000Z"
GIT_HASH = "a" * 40
OTHER_GIT_HASH = "b" * 40
SERVING_IMAGE = "us-central1-docker.pkg.dev/p/r/train@sha256:" + "c" * 64
MODEL_NAME = STAND_IN_MODEL_NAME
# Vertex names resources by project number, never by the project id init was given.
PROJECT_NUMBER = "123456789"


class FakeRegistry:
    """The registry surface the impl calls, keeping its versions across calls.

    Each rule is the SDK's or the service's: a missing Model raises NotFound when
    constructed, list_versions takes only `labels.key=value`, VersionInfo carries no
    labels, and get_model(version=...) returns that version's.
    """

    def __init__(self) -> None:
        self.location = ""
        self.labels: dict[str, list[dict[str, str]]] = {}
        self.uploads: list[dict[str, Any]] = []

    def resource_name(self, model_id: str) -> str:
        """The unversioned name, as `Model.resource_name` gives it."""
        return f"projects/{PROJECT_NUMBER}/locations/{self.location}/models/{model_id}"

    def init(self, *, project: str, location: str) -> None:
        """Record the location resource names are built in."""
        self.location = location

    def model(self, model_name: str) -> SimpleNamespace:
        """`aiplatform.Model(model_name=...)`, which fetches eagerly."""
        if model_name not in self.labels:
            raise NotFound(f"Model {model_name} is not found.")
        return SimpleNamespace(resource_name=self.resource_name(model_name))

    def model_registry(self, model: SimpleNamespace) -> SimpleNamespace:
        """`aiplatform.ModelRegistry(model)`, scoped to that one resource."""
        model_id = model.resource_name.rsplit("/", 1)[-1]
        return SimpleNamespace(
            list_versions=partial(self._list_versions, model_id),
            get_model=partial(self._version, model_id),
        )

    def upload(self, **kwargs: Any) -> SimpleNamespace:
        """`Model.upload`, refusing the pair `Model.copy` refuses."""
        requested, parent_model = kwargs.get("model_id"), kwargs.get("parent_model")
        if requested is not None and parent_model is not None:
            raise ValueError("model_id and parent_model can not be set together.")
        if parent_model is None:
            model_id = cast(str, requested)
            if model_id in self.labels:
                raise AlreadyExists(f"Model {model_id} already exists.")
            self.labels[model_id] = []
        else:
            model_id = cast(str, parent_model).rsplit("/", 1)[-1]
            self.model(model_id)
        self.uploads.append(kwargs)
        self.labels[model_id].append(dict(kwargs["labels"]))
        return self._version(model_id, str(len(self.labels[model_id])))

    def _list_versions(self, model_id: str, filter: str | None = None) -> list:
        """Only the documented equality; anything else is refused, as the RPC would."""
        match = re.fullmatch(r'labels\.([\w-]+)=(?:"([^"]*)"|([^"\s]+))', filter or "")
        if match is None:
            raise ValueError(f"filter {filter!r} is outside the documented grammar.")
        key, value = match.group(1), match.group(2) or match.group(3)
        return [
            VersionInfo(
                version_id=str(number),
                version_create_time=None,
                version_update_time=None,
                model_display_name=model_id,
                model_resource_name=self.resource_name(model_id),
                version_aliases=[],
                version_description="",
            )
            for number, labels in enumerate(self.labels[model_id], start=1)
            if labels.get(key) == value
        ]

    def _version(self, model_id: str, version: str) -> SimpleNamespace:
        """One version as `get_model` returns it, carrying its own labels."""
        return SimpleNamespace(
            version_id=version,
            versioned_resource_name=f"{self.resource_name(model_id)}@{version}",
            labels=dict(self.labels[model_id][int(version) - 1]),
        )


@pytest.fixture
def registry(mocker: MockerFixture) -> FakeRegistry:
    """A fresh registry behind the impl's `aiplatform`, the call log on `.sdk`."""
    fake = FakeRegistry()
    sdk = mocker.patch.object(impl_module, "aiplatform")
    sdk.init.side_effect = fake.init
    sdk.Model.side_effect = fake.model
    sdk.Model.upload.side_effect = fake.upload
    sdk.ModelRegistry.side_effect = fake.model_registry
    fake.sdk = sdk  # type: ignore[attr-defined]
    return fake


def _sdk(registry: FakeRegistry) -> MagicMock:
    """The patched module, typed for the call assertions."""
    return cast(MagicMock, registry.sdk)  # type: ignore[attr-defined]


@pytest.fixture
def inputs(
    tmp_path: Path, full_panel: pd.DataFrame, full_calendar: pd.DataFrame
) -> tuple[SourcedPath, SourcedPath, SourcedPath]:
    """The three frames as Feature publishes them, the panel stamped with its run id.

    The exogenous file is written although no step opens it yet, so compose records
    a URI naming bytes that exist.
    """
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir()
    panel_path = inputs_dir / "time_series.parquet"
    calendar_path = inputs_dir / "fiscal_calendar.parquet"
    additional_exog_path = inputs_dir / "exogenous_features.parquet"
    full_panel.assign(feature_run_id=FEATURE_RUN_ID).to_parquet(panel_path)
    full_calendar.to_parquet(calendar_path)
    full_panel[["unique_id", "ds"]].assign(
        holiday_days_in_week=0, week_sin=0.0, week_cos=1.0
    ).to_parquet(additional_exog_path)
    return (
        SourcedPath(path=panel_path, uri="gs://bucket/time_series.parquet"),
        SourcedPath(path=calendar_path, uri="gs://bucket/fiscal_calendar.parquet"),
        SourcedPath(
            path=additional_exog_path, uri="gs://bucket/exogenous_features.parquet"
        ),
    )


def _seed_environment(root: Path, env: str = ENV) -> Path:
    """The environment root, its pointer seeded as the writer leaves the real one.

    Seeded only when absent, like the hand-placed file: a second run in one
    environment must find its predecessor's record, not a fresh seed. The
    `audit` key is one this repo does not model, and sorts ahead of the two it
    does, so a reordering writer moves it.
    """
    env_root = root / env
    env_root.mkdir(parents=True, exist_ok=True)
    pointer_path = env_root / LATEST_POINTER_FILENAME
    if not pointer_path.is_file():
        seed = {
            "feature": {
                "feature_run_id": FEATURE_RUN_ID,
                "published": {
                    "panel_uri": "gs://bucket/time_series.parquet",
                    "calendar_uri": "gs://bucket/fiscal_calendar.parquet",
                    "exogenous_uri": "gs://bucket/exogenous_features.parquet",
                },
                "env": env,
                "schema_version": "0.1.0",
                "git_hash": "d" * 40,
                "completed_at": "2026-09-20T00:00:00+00:00",
                "panel": {"rows": 60, "series": 3},
            },
            "inference": "unset",
            "audit": {"written_by": "a slice this repo does not model"},
        }
        pointer_path.write_text(json.dumps(seed, indent=2) + "\n")
    # A broken fixture reads as a broken check.
    on_disk = json.loads(pointer_path.read_text())
    LatestRunPointer.model_validate(on_disk)
    FeatureRunOutputs.model_validate(on_disk["feature"])
    return env_root


def _compose(
    inputs: tuple[SourcedPath, SourcedPath, SourcedPath],
    root: Path,
    run_id: str,
    git_hash: str,
    *,
    config_dir: Path | None = None,
    env: str = ENV,
) -> Path:
    """Run compose_configs for one run, in the layout the resolvers really produce.

    The run root is <root>/<env>/train/<run_id>, so the environment root above it
    is inside tmp_path and carries a pointer, as it does in both execution modes.
    """
    panel, calendar, additional_exog = inputs
    run_dir = _seed_environment(root, env) / "train" / run_id
    compose_configs_impl(
        config_dir=config_dir or stand_in_config_dir(root),
        env=env,
        panel=panel,
        calendar=calendar,
        additional_exog=additional_exog,
        expected_feature_run_id=FEATURE_RUN_ID,
        train_run_id=run_id,
        git_hash=git_hash,
        out_dir=run_dir / "compose_configs",
    )
    return run_dir


def _fit(inputs: tuple[SourcedPath, SourcedPath, SourcedPath], run_dir: Path) -> None:
    """Run final_fit on the challenger, leaving its bundle under run_dir."""
    panel, calendar, additional_exog = inputs
    final_fit_impl(
        panel_path=panel.path,
        calendar_path=calendar.path,
        additional_exog_path=additional_exog.path,
        compose_configs_dir=run_dir / "compose_configs",
        model_name=MODEL_NAME,
        out_dir=run_dir / "final_fit" / MODEL_NAME,
    )


def _copied_config_dir(tmp_path: Path) -> Path:
    """The stand-in model's tree under tmp_path, for a test to edit one fragment of."""
    return stand_in_config_dir(tmp_path)


def _stage(
    inputs: tuple[SourcedPath, SourcedPath, SourcedPath],
    root: Path,
    run_id: str = TRAIN_RUN_ID,
    git_hash: str = GIT_HASH,
) -> Path:
    """A run root exactly as compose_configs and final_fit leave it."""
    run_dir = _compose(inputs, root, run_id, git_hash)
    _fit(inputs, run_dir)
    return run_dir


@pytest.fixture
def run_dir(
    inputs: tuple[SourcedPath, SourcedPath, SourcedPath], tmp_path: Path
) -> Path:
    """This test's run root, self-checking both upstream steps and the pointer."""
    run_dir = _stage(inputs, tmp_path)
    assert (run_dir / "compose_configs" / "manifest.json").is_file()
    assert (run_dir / "final_fit" / MODEL_NAME / "final_fit_manifest.json").is_file()
    assert latest_pointer_path(run_dir).is_file()
    return run_dir


def _bundle(run_dir: Path) -> SourcedPath:
    """The bundle as its wrapper hands it over: a mount path and a gs:// URI."""
    return SourcedPath(
        path=run_dir / "final_fit" / MODEL_NAME,
        uri=f"gs://bucket/{ENV}/train/{run_dir.name}/final_fit/{MODEL_NAME}/",
    )


def _register(root: Path, **overrides: Any) -> RegisterModelSummary:
    """Call the impl as its wrapper will, every path derived from the run root."""
    arguments: dict[str, Any] = {
        "bundle": _bundle(root),
        "compose_configs_dir": root / "compose_configs",
        "serving_container_image_uri": SERVING_IMAGE,
        "run_dir": root,
    }
    arguments.update(overrides)
    return register_model_impl(**arguments)


def _outputs(run_dir: Path) -> TrainRunOutputs:
    """The run record, read back through the schema Inference reads it with."""
    return TrainRunOutputs.model_validate_json(
        (run_dir / RUN_OUTPUTS_FILENAME).read_text()
    )


def _identity(run_dir: Path) -> TrainRunIdentity:
    """The identity compose_configs wrote for this run."""
    return TrainRunIdentity.model_validate_json(
        (run_dir / "run_identity.json").read_text()
    )


def _registry_config() -> ModelRegistry:
    """From the committed tree, so an infra.yaml edit cannot fail these."""
    _, infra, _ = compose_train_static_configs(CONFIG_DIR, ENV)
    return cast(TrainInfraConfig, infra.config).model_registry


def _expected_model_id() -> str:
    """The id the impl must compose for the challenger."""
    return compose_model_id(_registry_config().model_id_prefix, MODEL_NAME)


# ================================================
# The four registry states
# ================================================


def test_a_first_registration_creates_the_resource_by_id(
    run_dir: Path, registry: FakeRegistry
) -> None:
    """model_id names a resource being created; a parent_model would name none."""
    summary = _register(run_dir)

    [upload] = registry.uploads
    assert upload["model_id"] == _expected_model_id()
    assert upload["parent_model"] is None
    assert summary.uploaded
    assert summary.model_tag.endswith(f"/models/{_expected_model_id()}@1")


def test_the_upload_is_located_labeled_and_pointed_from_the_runs_own_record(
    run_dir: Path, registry: FakeRegistry
) -> None:
    """Project and location from environment.yaml before any fetch, labels from the
    identity, and the bundle's URI rather than the mount path the impl reads."""
    _register(run_dir)

    environment, _, _ = compose_train_static_configs(CONFIG_DIR, ENV)
    compute = cast(EnvironmentConfig, environment.config).compute
    assert _sdk(registry).mock_calls[0] == call.init(
        project=compute.project_id, location=compute.location
    )
    [upload] = registry.uploads
    identity = _identity(run_dir)
    assert upload["labels"] == {
        "train_run_id": identity.train_run_id,
        "git_hash": identity.git_hash,
    }
    assert upload["artifact_uri"] == _bundle(run_dir).uri
    assert upload["serving_container_image_uri"] == SERVING_IMAGE
    # The one place a registry query can match on the Feature run.
    assert f"feature_run_id={identity.feature_run_id}" in upload["version_description"]


def test_run_outputs_records_what_the_registry_returned(
    run_dir: Path, registry: FakeRegistry
) -> None:
    """The tag is read back, never assembled, and training_data is the bundle's own."""
    summary = _register(run_dir)

    outputs = _outputs(run_dir)
    manifest = json.loads(
        (_bundle(run_dir).path / "final_fit_manifest.json").read_text()
    )
    assert outputs.published.model_tag == summary.model_tag
    assert outputs.published.bundle_uri == _bundle(run_dir).uri
    assert outputs.training_data.model_dump(mode="json") == manifest["training_data"]


def test_a_reworded_label_moves_the_display_name_and_never_the_id(
    inputs: tuple[SourcedPath, SourcedPath, SourcedPath],
    registry: FakeRegistry,
    tmp_path: Path,
) -> None:
    """The two prefixes are identical in the committed tree, so only a reworded label
    shows which one each name was composed from."""
    config_dir = _copied_config_dir(tmp_path)
    infra_path = config_dir / "train" / "infra.yaml"
    infra = yaml.safe_load(infra_path.read_text())
    infra["model_registry"]["display_name_prefix"] = "reworded-label"
    infra_path.write_text(yaml.safe_dump(infra))
    reworded_run_dir = _compose(
        inputs, tmp_path, TRAIN_RUN_ID, GIT_HASH, config_dir=config_dir
    )
    _fit(inputs, reworded_run_dir)

    _register(reworded_run_dir)

    [upload] = registry.uploads
    assert upload["display_name"] == compose_display_name("reworded-label", MODEL_NAME)
    assert upload["model_id"] == _expected_model_id()


def test_env_is_read_from_the_runs_own_record(
    inputs: tuple[SourcedPath, SourcedPath, SourcedPath],
    registry: FakeRegistry,
    tmp_path: Path,
) -> None:
    """EnvironmentConfig declares no env, so a literal would stamp every run dev."""
    config_dir = _copied_config_dir(tmp_path)
    environments = config_dir / "environments"
    shutil.copy(environments / "dev.yaml", environments / "prod.yaml")
    prod_run_dir = _compose(
        inputs, tmp_path, TRAIN_RUN_ID, GIT_HASH, config_dir=config_dir, env="prod"
    )
    _fit(inputs, prod_run_dir)

    _register(prod_run_dir)

    assert _outputs(prod_run_dir).env == "prod"


def test_a_second_run_adds_a_version_to_the_same_resource(
    inputs: tuple[SourcedPath, SourcedPath, SourcedPath],
    run_dir: Path,
    registry: FakeRegistry,
    tmp_path: Path,
) -> None:
    """A second resource would fork the registry: the next run must name a parent."""
    other_run_dir = _stage(inputs, tmp_path, run_id=OTHER_RUN_ID)

    _register(run_dir)
    second = _register(other_run_dir)

    assert list(registry.labels) == [_expected_model_id()]
    assert registry.uploads[1]["model_id"] is None
    assert registry.uploads[1]["parent_model"] == registry.resource_name(
        _expected_model_id()
    )
    assert second.model_tag.endswith("@2")


def test_a_restart_uploads_nothing_and_writes_the_same_record(
    run_dir: Path, registry: FakeRegistry
) -> None:
    """A filter naming the wrong label finds nothing, so the rerun would upload again.
    Only the completion time may differ between the two records."""
    first = _register(run_dir)
    first_outputs = _outputs(run_dir)

    second = _register(run_dir)

    assert len(registry.uploads) == 1
    assert not second.uploaded
    assert second.model_tag == first.model_tag
    assert _outputs(run_dir).model_dump(exclude={"completed_at"}) == (
        first_outputs.model_dump(exclude={"completed_at"})
    )


def test_a_rerun_on_another_commit_raises_and_leaves_no_record(
    inputs: tuple[SourcedPath, SourcedPath, SourcedPath],
    run_dir: Path,
    registry: FakeRegistry,
    tmp_path: Path,
) -> None:
    """Skipping would leave a version whose git_hash label names a commit that did not
    produce the bytes now at its URI; the message names both."""
    _register(run_dir)
    _stage(inputs, tmp_path, git_hash=OTHER_GIT_HASH)

    with pytest.raises(ValueError, match=f"{GIT_HASH}.*{OTHER_GIT_HASH}"):
        _register(run_dir)

    assert len(registry.uploads) == 1
    assert not (run_dir / RUN_OUTPUTS_FILENAME).exists()


def test_two_versions_under_one_run_id_raise_naming_both(
    run_dir: Path, registry: FakeRegistry
) -> None:
    """Corruption, reachable by a manual upload: no rule picks the one to record."""
    labels = {"train_run_id": TRAIN_RUN_ID, "git_hash": GIT_HASH}
    registry.init(project="p", location="us-central1")
    registry.upload(model_id=_expected_model_id(), labels=labels)
    registry.upload(parent_model=_expected_model_id(), labels=labels)

    with pytest.raises(RuntimeError, match=r"\['1', '2'\]"):
        _register(run_dir)

    assert len(registry.uploads) == 2
    assert not (run_dir / RUN_OUTPUTS_FILENAME).exists()


# ================================================
# Guards that fire before the registry is touched
# ================================================


def test_a_run_dir_that_is_not_this_runs_root_is_refused(
    inputs: tuple[SourcedPath, SourcedPath, SourcedPath],
    run_dir: Path,
    registry: FakeRegistry,
    tmp_path: Path,
) -> None:
    """A miswired run root would put this run's record in another run's directory."""
    other_run_dir = _compose(inputs, tmp_path, OTHER_RUN_ID, GIT_HASH)

    with pytest.raises(ValueError, match="run_dir"):
        _register(run_dir, run_dir=other_run_dir)

    assert _sdk(registry).mock_calls == []


def test_a_train_run_id_that_is_not_label_safe_is_refused(
    inputs: tuple[SourcedPath, SourcedPath, SourcedPath],
    registry: FakeRegistry,
    tmp_path: Path,
) -> None:
    """A run minted before ids were lowercased; the service would refuse its label."""
    legacy_run_dir = _stage(inputs, tmp_path, run_id="t-20260913T000000000000Z")

    with pytest.raises(ValueError, match="train_run_id"):
        _register(legacy_run_dir)

    assert _sdk(registry).mock_calls == []


def test_a_bundle_uri_naming_another_model_is_refused(
    run_dir: Path, registry: FakeRegistry
) -> None:
    """Two independent sides: the name final_fit recorded, and the URI wired in."""
    bundle = SourcedPath(
        path=_bundle(run_dir).path, uri=f"gs://bucket/{ENV}/train/x/final_fit/other/"
    )

    with pytest.raises(ValueError, match="does not end in the model"):
        _register(run_dir, bundle=bundle)

    assert _sdk(registry).mock_calls == []


def test_a_bundle_fitted_under_another_identity_is_refused(
    inputs: tuple[SourcedPath, SourcedPath, SourcedPath],
    run_dir: Path,
    registry: FakeRegistry,
    tmp_path: Path,
) -> None:
    """A rerun whose compose_configs finished and final_fit did not: labeling the old
    bytes with the new identity is the one outcome registration must never leave."""
    _compose(inputs, tmp_path, TRAIN_RUN_ID, OTHER_GIT_HASH)

    with pytest.raises(ValueError, match="fitted under"):
        _register(run_dir)

    assert _sdk(registry).mock_calls == []


def test_a_bundle_with_no_completion_marker_is_refused(
    run_dir: Path, registry: FakeRegistry
) -> None:
    """final_fit writes its manifest last, so without it the bundle is partial."""
    (_bundle(run_dir).path / "final_fit_manifest.json").unlink()

    with pytest.raises(ValueError, match="did not complete"):
        _register(run_dir)

    assert _sdk(registry).mock_calls == []


# ================================================
# The run pointer
# ================================================


def _pointer(run_dir: Path) -> dict:
    """The environment's pointer as it stands on disk."""
    return json.loads(latest_pointer_path(run_dir).read_text())


def test_the_seeded_pointer_is_where_the_impl_reads_it(
    run_dir: Path, tmp_path: Path
) -> None:
    """The fixture places it by the layout; the impl derives it from run_dir alone."""
    assert latest_pointer_path(run_dir) == tmp_path / ENV / LATEST_POINTER_FILENAME


def test_a_missing_pointer_raises_before_the_marker_is_deleted(
    run_dir: Path, registry: FakeRegistry
) -> None:
    """Seeded once per environment, so absence means the wrong root, not a lost file."""
    _register(run_dir)
    earlier = (run_dir / RUN_OUTPUTS_FILENAME).read_text()
    latest_pointer_path(run_dir).unlink()

    with pytest.raises(ValueError, match=LATEST_POINTER_FILENAME):
        _register(run_dir)

    assert (run_dir / RUN_OUTPUTS_FILENAME).read_text() == earlier


def test_a_missing_pointer_never_reaches_the_registry(
    run_dir: Path, registry: FakeRegistry
) -> None:
    """A version in the shared registry is the cost of checking this too late."""
    latest_pointer_path(run_dir).unlink()

    with pytest.raises(ValueError, match=LATEST_POINTER_FILENAME):
        _register(run_dir)

    assert _sdk(registry).mock_calls == []


def test_the_pointer_is_written_after_the_completion_marker(
    run_dir: Path, registry: FakeRegistry, mocker: MockerFixture
) -> None:
    """Written first, it would name a run whose record was not at its root yet."""
    written: list[Path] = []
    real_write_text = Path.write_text

    def recording_write_text(self: Path, *args: Any, **kwargs: Any) -> int:
        written.append(self)
        return real_write_text(self, *args, **kwargs)

    mocker.patch.object(Path, "write_text", recording_write_text)

    _register(run_dir)

    assert written.index(run_dir / RUN_OUTPUTS_FILENAME) < written.index(
        latest_pointer_path(run_dir)
    )


def test_the_train_block_drops_only_the_cross_reference(
    run_dir: Path, registry: FakeRegistry
) -> None:
    """`feature` already names the newest Feature run; this names the one trained on."""
    _register(run_dir)

    record = json.loads((run_dir / RUN_OUTPUTS_FILENAME).read_text())
    block = _pointer(run_dir)["train"]

    assert "feature_run_id" in record
    assert block == {
        key: value for key, value in record.items() if key != "feature_run_id"
    }


def test_every_key_train_does_not_own_survives_the_rewrite(
    run_dir: Path, registry: FakeRegistry
) -> None:
    """Round-tripping through the model would drop every key it does not declare."""
    before = {key: value for key, value in _pointer(run_dir).items() if key != "train"}

    _register(run_dir)

    after = {key: value for key, value in _pointer(run_dir).items() if key != "train"}
    assert after == before


def test_the_order_of_keys_train_does_not_own_survives(
    run_dir: Path, registry: FakeRegistry
) -> None:
    """A sorting writer passes the value comparison above and still reorders."""
    before = _pointer(run_dir)
    top_level = [key for key in before if key != "train"]
    feature_keys = list(before["feature"])

    _register(run_dir)

    after = _pointer(run_dir)
    assert [key for key in after if key != "train"] == top_level
    assert list(after["feature"]) == feature_keys


def test_the_rewritten_pointer_is_still_parseable_json(
    run_dir: Path, registry: FakeRegistry
) -> None:
    """The writer dumps the whole object; splicing a value into the text would not."""
    _register(run_dir)

    reread = LatestRunPointer.model_validate_json(
        latest_pointer_path(run_dir).read_text()
    )

    assert reread.train is not None
