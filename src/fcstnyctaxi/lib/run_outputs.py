"""Read an upstream slice's run-root outputs manifest, and resolve its artifacts.

Not `lib/io.py`, which is transport only: this knows the run layout and the schema.
"""

import logging

from fcstnyctaxi.lib.io import read_text_from_gcs
from fcstnyctaxi.lib.storage_layout import build_object_uri, build_run_root
from fcstnyctaxi.schemas.run_outputs import FeatureArtifacts, FeatureRunOutputs
from fcstnyctaxi.schemas.storage.common import RUN_OUTPUT_FILENAME

_log = logging.getLogger(__name__)

# One value, not a set: a set would imply several simultaneously acceptable
# versions, which is a case not in scope.
_EXPECTED_SCHEMA_VERSION = "0.1.0"


def read_feature_run_outputs(
    *, bucket: str, env: str, feature_run_id: str
) -> FeatureRunOutputs:
    """One Feature run's manifest, checked against the id and env it was asked for.

    Learns no step name and no artifact filename: it builds one run-root path and
    takes the URIs by role.

    Raises:
        ValueError: the manifest is absent, which means the run never completed;
            or its `feature_run_id` or `env` disagrees with the caller's.
        ValidationError: a load-bearing key is missing or malformed.
    """
    run_root = build_run_root(bucket, env, "feature", feature_run_id)
    uri = build_object_uri(run_root, RUN_OUTPUT_FILENAME)
    try:
        text = read_text_from_gcs(uri)
    # Presence is the completion signal, so absence is a domain fact about the run,
    # not a missing object on a path the operator never constructed.
    except FileNotFoundError as err:
        raise ValueError(
            f"No outputs manifest at {uri}, so Feature run {feature_run_id!r} did "
            "not complete. Its presence is what marks a run finished."
        ) from err

    outputs = FeatureRunOutputs.model_validate_json(text)
    if outputs.feature_run_id != feature_run_id:
        raise ValueError(
            f"{uri} declares feature_run_id {outputs.feature_run_id!r}, not the "
            f"requested {feature_run_id!r}; it was likely copied between runs."
        )
    if outputs.env is not None and outputs.env != env:
        raise ValueError(
            f"{uri} declares env {outputs.env!r}, not the requested {env!r}."
        )
    # A diagnostic, never a gate: correctness rests on the keys above, not on the
    # version. Absent means the producer has not adopted versioning yet.
    if (
        outputs.schema_version is not None
        and outputs.schema_version != _EXPECTED_SCHEMA_VERSION
    ):
        _log.warning(
            "%s declares schema_version %s, which this reader has not seen; "
            "proceeding on the keys it needs.",
            uri,
            outputs.schema_version,
        )
    return outputs


def resolve_feature_artifacts(
    *,
    bucket: str,
    env: str,
    feature_run_id: str,
    panel_uri: str | None,
    calendar_uri: str | None,
    additional_exog_uri: str | None,
) -> FeatureArtifacts:
    """The three input URIs, from the caller's flags or from the Feature run's manifest.

    All three flags override, none resolves, anything between is refused: artifacts
    from different sources would be recorded nowhere.

    Raises:
        ValueError: one or two URIs were supplied, or resolution failed.
        ValidationError: a supplied override URI is not a gs:// URI, or a
            load-bearing key in the manifest is missing or malformed.
    """
    supplied = [
        uri is not None for uri in (panel_uri, calendar_uri, additional_exog_uri)
    ]
    if any(supplied) and not all(supplied):
        raise ValueError(
            "--panel-uri, --calendar-uri and --additional-exog-uri must be given "
            "together: pass all three to override resolution, or none to resolve "
            "from --feature-run-id."
        )
    if panel_uri is None:
        artifacts = read_feature_run_outputs(
            bucket=bucket, env=env, feature_run_id=feature_run_id
        ).published
        source = "resolved"
    else:
        artifacts = FeatureArtifacts(
            panel_uri=panel_uri,
            calendar_uri=calendar_uri,
            exogenous_uri=additional_exog_uri,
        )
        source = "supplied"

    # One definition, because the override flags are to be retired on this evidence
    # and two callers computing it separately could disagree.
    _log.info(
        "feature artifacts: source=%s panel=%s calendar=%s additional_exog=%s",
        source,
        artifacts.panel_uri,
        artifacts.calendar_uri,
        artifacts.exogenous_uri,
    )
    return artifacts
