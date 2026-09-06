"""EnvironmentConfig — the one destination every slice composes.

Selected by the ``env`` parameter, which picks ``config/environments/<env>.yaml``.
There is deliberately **no** ``env`` field: the file path is the binding, and
declaring it inside would be a second source of a derived fact — copy
``dev.yaml`` to ``prod.yaml``, forget the field, and a file at the prod path
declares itself dev. ``env`` is recorded in the emitted ``manifest.json``
instead (spec §4.11).

``service_accounts`` is deliberately absent, pending an infosec determination
on whether identifiers may live in source control. Values are supplied at
submit time via ``FCST_{FEATURE,TRAIN,INFERENCE}_SERVICE_ACCOUNT``; see
``.env.example`` and spec §4.12.

Note: ``VertexSettings`` here is **not**
``fcstnyctaxi.schemas.config_schemas.VertexSettings``, which serves the ingress
``PipelineConfig`` and additionally carries ``pipeline_service_account`` and
``display_name_prefix``. That module is slated to become
``FeaturePipelineConfig`` (monthly_revenue_training_pipeline §14); this is the
forward-looking definition. ``display_name_prefix`` is per-slice here and lives
on ``<Slice>InfraConfig``.
"""

from pydantic import BaseModel, ConfigDict, Field


class GcpSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    project_id: str = Field(..., min_length=1)
    location: str = Field(..., min_length=1)
    bucket_name: str = Field(..., min_length=1)


class VertexSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pipeline_root: str = Field(..., min_length=1, pattern=r"^gs://")


class ImageRef(BaseModel):
    """One image's stable identity — no tag, no digest.

    The tag is the git hash, resolved at build and submit time
    (configs_schemas_images §5), so committing one here would put a second
    source of truth for a value the release process owns into a file that is
    itself baked into the image.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    repository: str = Field(..., min_length=1)
    image: str = Field(..., min_length=1)


class SliceImages(BaseModel):
    """One ImageRef per slice.

    A named class with three required fields rather than ``dict[str, ImageRef]``,
    so a missing slice is a validation error at composition rather than a
    KeyError at submit time. Same reasoning as ModelRoles.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    feature: ImageRef
    train: ImageRef
    inference: ImageRef


class ArtifactRegistrySettings(BaseModel):
    """Registry location, deliberately independent of ``gcp.*``.

    A shared registry commonly lives in a separate project, and registry region
    diverges from compute region more often than project does. The two are
    coincident today; that is not the same fact.

    Stored in parts rather than as a full URI, because a URI would duplicate
    ``location`` and ``project_id`` from this same file, so a prod file with a
    different project would need the same change made twice. The host is
    composed where used::

        f"{ar.location}-docker.pkg.dev/{ar.project_id}/{img.repository}/{img.image}"
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    project_id: str = Field(..., min_length=1)
    location: str = Field(..., min_length=1)
    images: SliceImages


class EnvironmentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    gcp: GcpSettings
    vertex: VertexSettings
    artifact_registry: ArtifactRegistrySettings
