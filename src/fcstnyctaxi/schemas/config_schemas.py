from pydantic import BaseModel, ConfigDict, Field


class ProjectSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    project_id: str
    location: str
    env: str
    bucket_name: str


class DockerSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    extract_db_to_bucket: str = Field(..., min_length=1)


class VertexSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pipeline_service_account: str = Field(
        ..., min_length=1, pattern=r".+@.+\.iam\.gserviceaccount\.com$"
    )
    pipeline_root: str = Field(..., min_length=1, pattern=r"^gs://.+")
    display_name_prefix: str = Field(..., min_length=1)


class ExtractDbToBucketConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sql_filename: str
    sql_params: dict[str, object] = {}
    gcs_prefix: str
    output_filename: str = Field(..., min_length=1)


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    project_settings: ProjectSettings
    extract_db_to_bucket: ExtractDbToBucketConfig
    docker: DockerSettings
    vertex: VertexSettings
