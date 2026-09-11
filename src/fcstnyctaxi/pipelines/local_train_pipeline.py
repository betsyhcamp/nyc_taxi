"""Run Training's `compose_configs` step locally against real GCS artifacts.

The local half of the two execution modes: it resolves placement, stages the
Feature artifacts to a scratch directory, calls `compose_configs_impl` with the
arguments a KFP wrapper will pass, and publishes the step directory. On Vertex
the mount removes the staging; the impl call is the same either way.

Permanent, not a prototype. It coexists with the Vertex pipeline as the mode that
runs without an image, and it is the mode in which the emitted configs are read.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from fcstnyctaxi.core.train.compose_configs_impl import (
    SourcedPath,
    compose_configs_impl,
    compose_train_static_configs,
)
from fcstnyctaxi.lib.io import build_run_prefix, download_from_gcs, sync_to_gcs
from fcstnyctaxi.lib.utils import (
    generate_run_id,
    get_project_root_dir,
    require_path_safe_run_id,
)

logger = logging.getLogger(__name__)

# This runner drives one step, and the step names its own directory under the run
# root. No step accepts a full output path, so this is not a parameter.
_STEP = "compose_configs"


def _parse_args() -> argparse.Namespace:
    """The environment, the upstream Feature run and its artifacts, and placement.

    `--env` carries no `choices`: unlike the stand-in this reads no hardcoded
    location, so `require_known_environment` is the whole guard.
    """
    parser = argparse.ArgumentParser(
        description="Compose and publish every Training config for one run."
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
        "--scratch-dir",
        # A TemporaryDirectory deletes on failure, which is when the staged
        # inputs and the half-written output matter most.
        default=Path(tempfile.gettempdir()) / "fcstnyctaxi",
        type=Path,
        help="Root of the local mirror of the bucket namespace. Defaults to "
        "<tmpdir>/fcstnyctaxi. Never deleted automatically.",
    )
    return parser.parse_args()


def _mirror_path(gcs_uri: str, root: Path) -> Path:
    """The local stand-in for one GCS object or prefix, bucket segment included.

    Derived from the URI, so two objects sharing a basename cannot alias locally.
    """
    return root / gcs_uri.removeprefix("gs://")


def _require_git_hash(repo_dir: Path) -> str:
    """The commit this run reproduces from, refused rather than stamped as null.

    Both commands run at repo_dir, since the process CWD can be a sibling repo.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    # Wrapped because git's own exit status names the command that failed, not
    # what this run needed it for.
    except (OSError, subprocess.CalledProcessError) as err:
        raise RuntimeError(
            "git rev-parse HEAD failed, so this run cannot record the commit that "
            "produced it. Run from inside the repository, with git installed."
        ) from err

    # Tracked-only: notes/, memory/ and notebooks/eda_figs/ are untracked and not
    # ignored, so git status --porcelain would read dirty on every run.
    completed = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--"], cwd=repo_dir, check=False
    )
    # 0 clean, 1 dirty, 128 git failed — and 128 is truthy, so an unguarded check
    # stamps -dirty on a failure.
    if completed.returncode not in (0, 1):
        raise RuntimeError(
            f"git diff --quiet HEAD -- exited {completed.returncode}, so a clean "
            "tree cannot be told from a modified one."
        )
    return f"{sha}-dirty" if completed.returncode else sha


def main() -> None:
    """Compose one Training run's configs and publish them to its run prefix.

    Raises:
        ValueError: If either run id is not path-safe, if the two input URIs are
            identical, if `--env` has no `environments/<env>.yaml`, on a failed
            lineage check, or on any composition failure.
        RuntimeError: If the git hash cannot be determined, since a run whose
            commit is unknown cannot be reproduced from its own record.
        ValidationError: If an identity field is malformed.
    """
    logging.Formatter.converter = time.gmtime
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03dZ %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    args = _parse_args()

    # Both ids are checked here because both are argv. Only --run-id becomes a
    # path; --feature-run-id is checked so the two flags accept one vocabulary.
    run_id = generate_run_id() if args.run_id is None else args.run_id
    require_path_safe_run_id(args.feature_run_id, "--feature-run-id")
    require_path_safe_run_id(run_id, "--run-id")
    if args.panel_uri == args.calendar_uri:
        raise ValueError(
            "--panel-uri and --calendar-uri name the same object; the lineage "
            "check and the origin guard both pass when the frames are one file."
        )

    project_root = get_project_root_dir()
    config_dir = project_root / "config"
    git_hash = _require_git_hash(project_root)

    environment, _, _ = compose_train_static_configs(config_dir, args.env)
    # Held as a value rather than folded into step_uri: backtest, evaluate, and
    # final_fit each append their own step name to this same prefix next PR.
    run_prefix = build_run_prefix(
        bucket=environment.config.storage.bucket_name,
        env=args.env,
        slice_name="train",
        run_id=run_id,
    )
    step_uri = f"{run_prefix}{_STEP}/"

    mirror_root = args.scratch_dir
    out_dir = _mirror_path(step_uri, mirror_root)
    logger.info(
        "compose_configs starting: run_id=%s feature_run_id=%s out_dir=%s uri=%s",
        run_id,
        args.feature_run_id,
        out_dir,
        step_uri,
    )

    # Scratch persists, so a retry under the same --run-id finds the previous
    # attempt's files. Clearing gives out_dir exactly this run's output and
    # sync_to_gcs gives the prefix exactly out_dir; neither implies the other.
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    panel = SourcedPath(
        path=download_from_gcs(
            args.panel_uri, _mirror_path(args.panel_uri, mirror_root).parent
        ),
        uri=args.panel_uri,
    )
    calendar = SourcedPath(
        path=download_from_gcs(
            args.calendar_uri, _mirror_path(args.calendar_uri, mirror_root).parent
        ),
        uri=args.calendar_uri,
    )

    summary = compose_configs_impl(
        config_dir=config_dir,
        env=args.env,
        panel=panel,
        calendar=calendar,
        expected_feature_run_id=args.feature_run_id,
        train_run_id=run_id,
        git_hash=git_hash,
        out_dir=out_dir,
    )
    # The impl writes run_identity.json last, so publishing it alone and last
    # makes its presence at the prefix mean complete rather than started.
    uploaded, removed = sync_to_gcs(
        out_dir, step_uri, completion_marker="run_identity.json"
    )

    logger.info(
        "compose_configs complete: uploaded=%d removed=%d uri=%s run_id=%s "
        "models=%s n_origins=%d origins=%s..%s last_complete_actual_month=%d "
        "start_months=%s out_dir=%s",
        uploaded,
        removed,
        step_uri,
        run_id,
        summary.model_names,
        summary.n_origins,
        summary.first_origin,
        summary.last_origin,
        summary.last_complete_actual_month,
        summary.start_months,
        out_dir,
    )


if __name__ == "__main__":
    main()
