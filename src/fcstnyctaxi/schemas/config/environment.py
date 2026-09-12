# TODO: Once conventions are understood by the team, trim down comments

"""EnvironmentConfig is the one destination every slice (F/T/I) composes.

The rules this tree follows are in ``config/README.md``.

Selected by the ``env`` parameter, which picks ``config/environments/<env>.yaml``.
There is deliberately **no** ``env`` field: the file path is the binding, and
declaring it inside would be a second source of a derived fact since copy
``dev.yaml`` to ``prod.yaml``, forget the field, and a file at the prod path
declares itself dev. ``env`` is recorded in the emitted ``manifest.json``
instead.

**This is not "the GCP block" — it is everything that varies by environment,
for all three slices.** The parity rule makes each slice's own ``infra.yaml`` and
``modeling.yaml`` environment-*independent*, so ``environments/<env>.yaml`` is
the only file in the tree that can hold a value differing between dev and prod.
That is why ``artifact_registry.images`` carries Feature's and Inference's image
refs in a file Training composes, and why ``source_data`` lives here rather than
in ``config/feature/``.

The blocks are **operational planes**, each with its own project and location
because each is separately assignable in GCP:

    compute            where Vertex AI pipeline jobs run and bill
    storage            where this system writes its artifacts
    vertex             Vertex-specific settings
    artifact_registry  where the Vertex AI pipeline code image is pulled from
    source_data        where the source tables upstream of all piplines live

``service_accounts`` is deliberately absent since putting this identifier in source
control could be a security issue by some standards. Values are supplied at submit
time via ``FCST_{FEATURE,TRAIN,INFERENCE}_SERVICE_ACCOUNT``; see ``.env.example``.

Note: ``VertexSettings`` here is **not**
``<project_repo_name>.schemas.config_schemas.VertexSettings``, which serves the ingress
``PipelineConfig`` and additionally carries ``pipeline_service_account`` and
``display_name_prefix``. That module is slated to become
``FeaturePipelineConfig`` when the Feature pipeline is rebuilt; this is the
forward-looking definition. ``display_name_prefix`` is per-slice here and lives
on ``<Slice>InfraConfig``.
"""

from pydantic import BaseModel, ConfigDict, Field


class ComputeSettings(BaseModel):
    """Where pipeline jobs are created, run, and billed.

    Consumed by ``aiplatform.init(project=, location=)``, and by Feature as the
    BigQuery **job** project — the one that pays for the query, which is a
    different thing from the project holding the tables (see SourceDataSettings).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    project_id: str = Field(..., min_length=1)
    location: str = Field(..., min_length=1)


class StorageSettings(BaseModel):
    """Where this system writes its own artifacts, addressed by bucket name alone.

    No ``project_id`` or ``location``, unlike every other plane here — bucket names
    are globally unique, so ``build_run_prefix`` and the GCS helpers take nothing else.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    bucket_name: str = Field(..., min_length=1)


class VertexSettings(BaseModel):
    """Vertex-specific settings — ``pipeline_root`` is KFP's own scratch.

    Its own block rather than a field under storage, because the block name says who
    consumes it; not derived from ``bucket_name``, so scratch keeps its own lifecycle.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    pipeline_root: str = Field(..., min_length=1, pattern=r"^gs://")


class ImageRef(BaseModel):
    """One image's stable identity: repository plus one flat image segment.

    No tag or digest: the release process owns those. The flat segment matters —
    the compile step derives the repository by stripping the last segment, so a
    nested name quietly narrows what counts as project-owned instead of failing.
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
    """Where the code image is pulled from — independent of compute.

    Prod must pull the same bytes dev ran, and the config tree is baked in under the
    git-hash tag, so deriving this from compute would force a rebuild per environment
    and give one git hash two images. Parts rather than a URI, composed where used as
    f"{location}-docker.pkg.dev/{project_id}/{repository}/{image}".
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    project_id: str = Field(..., min_length=1)
    location: str = Field(..., min_length=1)
    images: SliceImages


class SourceDataSettings(BaseModel):
    """Where the BigQuery source tables Feature queries live.

    Separate from compute because the two are separately assignable. ``location``
    is not a free choice: a BigQuery job runs in its dataset's location, and may be
    multi-regional (``US``) — which is why every plane says ``location``, not region.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    project_id: str = Field(..., min_length=1)
    location: str = Field(..., min_length=1)


class EnvironmentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    compute: ComputeSettings
    storage: StorageSettings
    vertex: VertexSettings
    artifact_registry: ArtifactRegistrySettings
    source_data: SourceDataSettings
