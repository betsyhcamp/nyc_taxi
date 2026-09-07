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
    """Where this system writes its own artifacts.

    **Deliberately carries no project_id or location, unlike every other plane
    in this file.** GCS bucket names are globally unique, so a ``gs://`` URI
    addresses an object completely: ``build_run_prefix``, ``build_feature_uri``,
    ``upload_to_gcs`` and ``download_from_gcs`` all take the bucket name and
    nothing else. No API call in this codebase, present or planned, accepts a
    project or a location for storage.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    bucket_name: str = Field(..., min_length=1)


class VertexSettings(BaseModel):
    """Vertex-specific settings.

    ``pipeline_root`` is where KFP writes component outputs, caches, and
    execution metadata. It stays here rather than under storage because the
    block name says who consumes it, which ``pipeline_root`` alone does not
     and it is deliberately not derived from ``bucket_name``, so orchestration
    scratch can be given its own bucket and lifecycle policy without a schema
    change.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    pipeline_root: str = Field(..., min_length=1, pattern=r"^gs://")


class ImageRef(BaseModel):
    """One image's stable identity with no tag, no digest.

    The tag is the git hash, resolved at build and submit time, so committing one
    here would put a second source of truth for a value the release process owns
    into a file that is itself baked into the image.
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

    A shared registry might live in a separate project, and registry region
    could then diverge from compute region more often than project does. The decisive
    case is image promotion: the whole config tree is baked into the image and the tag
    is the git hash, so one tag identifies code and config together and prod
    should pull the **same bytes** dev ran. Deriving this project from compute's
    would force a rebuild per environment, which produces a different image for
    one git hash. See ``config/README.md``.

    Stored in parts rather than as a full URI, because a URI would duplicate
    ``location`` and ``project_id`` from this same file. The host is composed
    where used:

        f"{ar.location}-docker.pkg.dev/{ar.project_id}/{img.repository}/{img.image}"
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    project_id: str = Field(..., min_length=1)
    location: str = Field(..., min_length=1)
    images: SliceImages


class SourceDataSettings(BaseModel):
    """Where the BigQuery source tables live which are upstream of Feature which are
    initially queried by Feature.

    Separate from compute because the two are separately assignable: source
    tables commonly live in data-platform or domain-owned projects with their
    own governance, and ML compute reads them cross-project.

    ``location`` is **not a free choice** — a BigQuery job must run in its
    dataset's location, so this value is determined by where the data is. It may
    be regional (``us-central1``) or multi-regional (``US``), which is why the
    field is spelled ``location`` rather than ``region`` across every plane here.

    Unread right now: Training and Inference never touch BigQuery. It lives in
    EnvironmentConfig rather than ``config/feature/`` because the parity rule
    makes Feature's own fragments environment-independent, and this varies by
    environment when dev and prod read different datasets.
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
