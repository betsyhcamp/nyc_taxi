import argparse
import hashlib
import logging
import os
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import yaml
from dotenv import load_dotenv
from google.cloud import aiplatform

from fcstnyctaxi.lib.config.bindings import environment_bindings, train_infra_bindings
from fcstnyctaxi.lib.config.composition import compose_config
from fcstnyctaxi.lib.storage_layout import resolve_run_prefix
from fcstnyctaxi.lib.utils import (
    generate_run_id,
    get_project_root_dir,
    require_path_safe_run_id,
)
from fcstnyctaxi.schemas.config.environment import EnvironmentConfig
from fcstnyctaxi.schemas.config.train import TrainInfraConfig

logger = logging.getLogger(__name__)

# EnvironmentConfig carries no identities, so the account arrives from the
# environment and its absence is this script's own error to raise.
_SERVICE_ACCOUNT_VAR = "FCST_TRAIN_SERVICE_ACCOUNT"

# A constant so a test can redirect it. Shares the local runner's scratch root,
# which `task scratch-clean` already removes.
_LAST_RUN_ID_PATH = Path(tempfile.gettempdir()) / "fcstnyctaxi" / ".last_run_id"


def _parse_args() -> argparse.Namespace:
    """The template to submit, plus the domain flags the local runner also takes.

    `--env` carries no `choices`: `require_known_environment`, via
    `resolve_run_prefix`, is the whole guard.
    """
    parser = argparse.ArgumentParser(
        description="Submit a compiled Training template to Vertex AI Pipelines."
    )
    parser.add_argument(
        "--template-path",
        required=True,
        # A str, not a Path: this flag also takes a published template URI,
        # which Path would collapse to "https:/...".
        help="Compiled template to submit, as compile-train writes it.",
    )
    parser.add_argument(
        "--env",
        required=True,
        help="Environment selector; needs a config/environments/<env>.yaml.",
    )
    parser.add_argument(
        "--feature-run-id",
        required=True,
        help="The Feature run that produced the two artifacts, checked against "
        "the feature_run_id column in both.",
    )
    parser.add_argument(
        "--panel-uri",
        required=True,
        help="gs:// URI of the actuals, as publish_feature_stand_in.py prints it.",
    )
    parser.add_argument(
        "--calendar-uri",
        required=True,
        help="gs:// URI of the fiscal calendar, from the same Feature run.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Reuse an existing id, overwriting that run's artifacts; one is "
        "generated when absent.",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Block until the run finishes. Fire and forget by default.",
    )
    parser.add_argument(
        "--no-caching",
        action="store_true",
        help="Re-execute every task; caching is on by default.",
    )
    return parser.parse_args()


def _executor_images(spec: dict[str, Any]) -> set[str]:
    """Every Docker image the compiled pipeline will actually run.Importer steps run no
    image and are skipped.
    """
    executors = spec["deploymentSpec"]["executors"]
    return {ex["container"]["image"] for ex in executors.values() if "container" in ex}


def _console_url(resource_name: str, project_id: str, location: str) -> str:
    """The Vertex UI page for one submitted run.

    Built from the public resource name rather than the private `_dashboard_uri`,
    which `submit` has already called by the time this runs.
    """
    job_id = resource_name.rsplit("/", 1)[-1]
    return (
        f"https://console.cloud.google.com/vertex-ai/locations/{location}"
        f"/pipelines/runs/{job_id}?project={project_id}"
    )


def main() -> None:
    """Submit one compiled template as a Training run and log what it submitted.

    Raises:
        RuntimeError: If FCST_TRAIN_SERVICE_ACCOUNT is unset or blank.
        ValueError: If either run id is not path-safe, if `--env` has no
            `environments/<env>.yaml`, or on any composition failure.
        FileNotFoundError: If `--template-path` names no file.
    """
    # Never override=True: the task environment sets FCST_TRAIN_IMAGE before the
    # interpreter starts, and a .env must not become a second source.
    load_dotenv()
    logging.Formatter.converter = time.gmtime
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03dZ %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    # Parsed before the account is read, so --help works with no .env present.
    args = _parse_args()

    service_account = os.environ.get(_SERVICE_ACCOUNT_VAR, "").strip()
    if not service_account:
        raise RuntimeError(
            f"{_SERVICE_ACCOUNT_VAR} is unset, so this run has no identity to "
            "submit as. Set it in .env or export it; see .env.example."
        )

    # Both are argv, so both are checked. Only --run-id becomes a path; the
    # other is checked so the two flags accept one vocabulary.
    run_id = generate_run_id() if args.run_id is None else args.run_id
    require_path_safe_run_id(args.feature_run_id, "--feature-run-id")
    require_path_safe_run_id(run_id, "--run-id")

    config_dir = get_project_root_dir() / "config"
    # First, because it carries the --env guard: an unknown selector names the
    # available ones rather than surfacing as a missing fragment below.
    run_prefix = resolve_run_prefix(config_dir, args.env, "train", run_id)
    # EnvironmentConfig composed a second time, deliberately: the alternative is
    # exposing the private prefix builder and losing one authority on layout.
    environment = cast(
        EnvironmentConfig,
        compose_config(config_dir, environment_bindings(args.env)).config,
    )
    infra = cast(
        TrainInfraConfig, compose_config(config_dir, train_infra_bindings()).config
    )

    # Also the missing-template guard: this raises FileNotFoundError, and
    # PipelineJob refuses the path too. No third check.
    template = Path(args.template_path)
    template_bytes = template.read_bytes()
    images = _executor_images(yaml.safe_load(template_bytes))
    enable_caching = not args.no_caching

    logger.info(
        "submitting: template=%s sha256=%s mtime=%s images=%s run_prefix=%s "
        "run_id=%s feature_run_id=%s caching=%s",
        template,
        hashlib.sha256(template_bytes).hexdigest(),
        datetime.fromtimestamp(template.stat().st_mtime, UTC).isoformat(),
        sorted(images),
        run_prefix,
        run_id,
        args.feature_run_id,
        enable_caching,
    )
    if not images:
        logger.warning(
            "template %s pins zero container executors, so the graph about to "
            "run executes no project code.",
            template,
        )

    aiplatform.init(
        project=environment.compute.project_id,
        location=environment.compute.location,
    )
    job = aiplatform.PipelineJob(
        display_name=f"{infra.display_name_prefix}-{run_id}",
        template_path=str(template),
        pipeline_root=environment.vertex.pipeline_root,
        parameter_values={
            "env": args.env,
            "train_run_id": run_id,
            "feature_run_id": args.feature_run_id,
            "panel_uri": args.panel_uri,
            "calendar_uri": args.calendar_uri,
        },
        enable_caching=enable_caching,
    )
    job.submit(service_account=service_account)

    # After the submit, so a failed one records nothing; before the wait, so an
    # interrupted --wait still leaves the id a restart needs.
    _LAST_RUN_ID_PATH.parent.mkdir(parents=True, exist_ok=True)
    _LAST_RUN_ID_PATH.write_text(f"{run_id}\n")

    logger.info(
        "submitted: display_name=%s resource=%s console=%s last_run_id=%s",
        job.display_name,
        job.resource_name,
        _console_url(
            job.resource_name,
            environment.compute.project_id,
            environment.compute.location,
        ),
        _LAST_RUN_ID_PATH,
    )

    if args.wait:
        job.wait()


if __name__ == "__main__":
    main()
