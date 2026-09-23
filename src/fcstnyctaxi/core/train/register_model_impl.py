import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from google.api_core.exceptions import NotFound
from google.cloud import aiplatform

from fcstnyctaxi.lib.registry_ids import compose_display_name, compose_model_id
from fcstnyctaxi.lib.storage_layout import (
    LATEST_POINTER_FILENAME,
    RUN_OUTPUTS_FILENAME,
    SourcedPath,
    latest_pointer_path,
)
from fcstnyctaxi.lib.utils import require_label_safe_run_id
from fcstnyctaxi.schemas.config.environment import EnvironmentConfig
from fcstnyctaxi.schemas.config.train import TrainInfraConfig
from fcstnyctaxi.schemas.run_identity import TrainRunIdentity
from fcstnyctaxi.schemas.run_outputs import (
    LatestRunPointer,
    RegisteredModel,
    TrainingData,
    TrainRunOutputs,
)

_log = logging.getLogger(__name__)

_RUN_ID_LABEL = "train_run_id"
_GIT_HASH_LABEL = "git_hash"
_SCHEMA_VERSION = "0.1.0"


@dataclass(frozen=True)
class RegisterModelSummary:
    """What the registry call produced, for a wrapper that never opens the bundle."""

    model_tag: str
    model_id: str
    version_id: str
    uploaded: bool
    model_name: str
    train_run_id: str
    git_hash: str

    def as_dict(self) -> dict[str, Any]:
        """For KFP's `artifact.metadata`; every field is already a str or a bool."""
        return asdict(self)


def _this_runs_version(model: Any, identity: TrainRunIdentity) -> Any | None:
    """The version this run registered earlier, or None if there is none.

    One label filtered, the other read back: the RPC documents no `AND`, and
    `VersionInfo` drops the labels the response carried.
    """
    registry = aiplatform.ModelRegistry(model)
    hits = registry.list_versions(
        filter=f'labels.{_RUN_ID_LABEL}="{identity.train_run_id}"'
    )
    if not hits:
        return None
    if len(hits) > 1:
        raise RuntimeError(
            f"versions {[hit.version_id for hit in hits]} of {model.resource_name} all "
            f"carry train_run_id {identity.train_run_id!r}. Delete the surplus, rerun."
        )

    version = registry.get_model(version=hits[0].version_id)
    stored = version.labels.get(_GIT_HASH_LABEL)
    # A reused --run-id after a code change overwrote the bundle this version names.
    if stored != identity.git_hash:
        raise ValueError(
            f"version {hits[0].version_id} of {model.resource_name} was registered "
            f"for train_run_id {identity.train_run_id!r} at git_hash {stored!r}, "
            f"but this run is at {identity.git_hash!r}."
        )
    return version


def register_model_impl(
    *,
    bundle: SourcedPath,
    compose_configs_dir: Path,
    serving_container_image_uri: str,
    run_dir: Path,
) -> RegisterModelSummary:
    """Upload the bundle as a version of its model, or find it already registered.

    Keyword-only: two `Path` parameters transpose without a type error. Provenance is
    read, never passed, the run pointer included: its path is derived from `run_dir`.
    `run_output.json` is deleted after the file checks and written second to last,
    ahead of the pointer. A documented exception to core/'s no-GCP-clients rule; an
    injected client would put the filter strings beyond the tests.

    Args:
        bundle (SourcedPath): The final fit bundle; path read, URI uploaded.
        compose_configs_dir (Path): The compose step's output, with
            `run_identity.json` beside it.
        serving_container_image_uri (str): Required by `Model.upload`; serves nothing.
        run_dir (Path): The run root, where `run_output.json` is written.

    Raises:
        ValueError: If the run id, `run_dir` or the bundle fails its check, a composed
            name exceeds its cap, this run id is registered at another git_hash, or
            the environment root holds no run pointer.
        RuntimeError: If more than one version carries this run id.
        ValidationError: If a file fails to revalidate on read.

    Returns:
        RegisterModelSummary: The registered version, and whether this call uploaded it.
    """
    identity = TrainRunIdentity.model_validate_json(
        (compose_configs_dir.parent / "run_identity.json").read_text()
    )
    # Before any label is built: the schema holds the id to min_length=1 alone.
    require_label_safe_run_id(identity.train_run_id, "run_identity.json train_run_id")
    if run_dir.name != identity.train_run_id:
        raise ValueError(
            f"run_dir {run_dir} is not the root of run {identity.train_run_id!r}."
        )

    manifest_path = bundle.path / "final_fit_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(
            f"No final_fit_manifest.json in {bundle.path}, so final_fit "
            "did not complete this bundle."
        )
    manifest = json.loads(manifest_path.read_text())
    model_name = manifest["model_name"]
    if bundle.uri.rstrip("/").rsplit("/", 1)[-1] != model_name:
        raise ValueError(
            f"bundle {bundle.uri} does not end in the model its manifest names, "
            f"{model_name!r}."
        )
    # The labels come from identity, so they must describe the bytes being uploaded.
    lineage = TrainRunIdentity.model_validate(manifest["lineage"])
    if lineage != identity:
        raise ValueError(
            f"bundle {bundle.uri} was fitted under {lineage}, not {identity}."
        )
    training_data = TrainingData.model_validate(manifest["training_data"])

    infra = TrainInfraConfig.model_validate(
        yaml.safe_load((compose_configs_dir / "infra.yaml").read_text())
    )
    environment = EnvironmentConfig.model_validate(
        yaml.safe_load((compose_configs_dir / "environment.yaml").read_text())
    )
    # EnvironmentConfig has no env field by design; compose_configs records it here.
    env = json.loads((compose_configs_dir / "manifest.json").read_text())["env"]
    registry = infra.model_registry
    model_id = compose_model_id(registry.model_id_prefix, model_name)
    display_name = compose_display_name(registry.display_name_prefix, model_name)

    # Read before anything irreversible. The file is seeded once per environment, so
    # absence means this run resolved the wrong environment root, not that a file is
    # missing.
    pointer_path = latest_pointer_path(run_dir)
    if not pointer_path.is_file():
        raise ValueError(
            f"No {LATEST_POINTER_FILENAME} at {pointer_path}, so run "
            f"{identity.train_run_id!r} resolved an environment root that has none."
        )
    pointer = json.loads(pointer_path.read_text())
    # Validated, then discarded: pydantic drops keys it does not declare, so the write
    # below edits this raw object instead of round-tripping the model.
    LatestRunPointer.model_validate(pointer)

    # Otherwise a failed rerun leaves the old record claiming the pipeline finished.
    (run_dir / RUN_OUTPUTS_FILENAME).unlink(missing_ok=True)

    aiplatform.init(
        project=environment.compute.project_id, location=environment.compute.location
    )
    try:
        model = aiplatform.Model(model_name=model_id)
    except NotFound:
        model = None
    version = None if model is None else _this_runs_version(model, identity)

    uploaded = version is None
    if version is None:
        version = aiplatform.Model.upload(
            # model_id creates a resource, parent_model selects one; never both.
            model_id=model_id if model is None else None,
            parent_model=None if model is None else model.resource_name,
            display_name=display_name,
            artifact_uri=bundle.uri,
            serving_container_image_uri=serving_container_image_uri,
            labels={
                _RUN_ID_LABEL: identity.train_run_id,
                _GIT_HASH_LABEL: identity.git_hash,
            },
            version_description=(
                f"feature_run_id={identity.feature_run_id} "
                f"train_end_ds={training_data.train_end_ds} "
                f"n_series={training_data.n_series} n_obs={training_data.n_obs}"
            ),
            # `default` means the newest registration, never an approved one.
            is_default_version=True,
            # The response is read for the versioned name before the record is written.
            sync=True,
        )

    outputs = TrainRunOutputs(
        train_run_id=identity.train_run_id,
        published=RegisteredModel(
            model_tag=version.versioned_resource_name, bundle_uri=bundle.uri
        ),
        feature_run_id=identity.feature_run_id,
        env=env,
        schema_version=_SCHEMA_VERSION,
        git_hash=identity.git_hash,
        completed_at=datetime.now(UTC),
        training_data=training_data,
    )
    # run_output.json written is the pipeline's completion marker. Keep this last
    # before the pointer.
    (run_dir / RUN_OUTPUTS_FILENAME).write_text(
        outputs.model_dump_json(indent=2, exclude_none=True) + "\n"
    )

    # A rerun of the run the pointer already names leaves it briefly naming a run
    # whose marker was deleted above. Accepted; logged so it is visible afterward.
    prior = pointer.get("train")
    if isinstance(prior, dict) and prior.get("train_run_id") == identity.train_run_id:
        _log.info(
            "run pointer already names %s; replacing its record.",
            identity.train_run_id,
        )
    # feature_run_id alone is dropped: the pointer's own `feature` key answers a
    # different question with a different value, and two answers cannot disagree if
    # only one is written.
    pointer["train"] = json.loads(
        outputs.model_dump_json(exclude_none=True, exclude={"feature_run_id"})
    )
    # After the marker, so the pointer never names a run that has none.
    pointer_path.write_text(json.dumps(pointer, indent=2) + "\n")

    _log.info(
        "register_model complete: model_tag=%s uploaded=%s train_run_id=%s",
        outputs.published.model_tag,
        uploaded,
        identity.train_run_id,
    )
    return RegisterModelSummary(
        model_tag=outputs.published.model_tag,
        model_id=model_id,
        version_id=version.version_id,
        uploaded=uploaded,
        model_name=model_name,
        train_run_id=identity.train_run_id,
        git_hash=identity.git_hash,
    )
