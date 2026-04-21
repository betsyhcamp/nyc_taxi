from pydantic import BaseModel, ConfigDict


class ProjectSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    project_id: str
    location: str
    env: str
    bucket_name: str


class ExtractDbToBucketConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sql_filename: str
    sql_params: dict[str, object] = {}
    gcs_prefix: str


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    project_settings: ProjectSettings
    extract_db_to_bucket: ExtractDbToBucketConfig
