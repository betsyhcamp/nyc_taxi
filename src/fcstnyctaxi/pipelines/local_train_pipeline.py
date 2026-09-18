"""
The local execution mode: calls each Training impl with the arguments its KFP wrapper
passes, and publishes each step directory. Vertex's mount removes the staging.

Permanent, not a prototype: the mode that runs without an image, and where the
emitted configs are read and the sidecars produced.
"""

import argparse
import logging
import shutil
import tempfile
import time
from pathlib import Path

from fcstnyctaxi.core.train.backtest_impl import backtest_impl
from fcstnyctaxi.core.train.compose_configs_impl import (
    SourcedPath,
    compose_configs_impl,
    compose_train_static_configs,
)
from fcstnyctaxi.lib.io import (
    download_from_gcs,
    require_gcs_uri,
    sync_to_gcs,
    upload_to_gcs,
)
from fcstnyctaxi.lib.run_outputs import resolve_feature_artifacts
from fcstnyctaxi.lib.storage_layout import resolve_run_prefix
from fcstnyctaxi.lib.utils import (
    generate_run_id,
    get_project_root_dir,
    require_git_hash,
    require_path_safe_run_id,
)

logger = logging.getLogger(__name__)

# Each step names its own directory under the run root. No step accepts a full
# output path, so these are not parameters. Backtest appends the model name too,
# which only the loop knows.
_COMPOSE_STEP = "compose_configs"
_BACKTEST_STEP = "backtest"


def _parse_args() -> argparse.Namespace:
    """The environment, the upstream Feature run and its artifacts, and placement.

    `--env` carries no `choices`: unlike the stand-in this reads no hardcoded
    location, so `require_known_environment` is the whole guard.
    """
    parser = argparse.ArgumentParser(
        description="Compose every Training config for one run, then back each model."
    )
    parser.add_argument(
        "--env",
        required=True,
        help="Environment selector; needs a config/environments/<env>.yaml.",
    )
    parser.add_argument(
        "--feature-run-id",
        required=True,
        help="The Feature run both artifact URIs resolve from, checked against "
        "the feature_run_id column in both.",
    )
    parser.add_argument(
        "--panel-uri",
        default=None,
        help="Override the actuals URI --feature-run-id resolves to; requires "
        "--calendar-uri.",
    )
    parser.add_argument(
        "--calendar-uri",
        default=None,
        help="Override the fiscal calendar URI --feature-run-id resolves to; "
        "requires --panel-uri.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Reuse an existing id, overwriting that run's artifacts; one is "
        "generated when absent.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Back only this model instead of every model in model_roles; must "
        "name one this run composed.",
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
    require_gcs_uri(gcs_uri)
    key = gcs_uri.removeprefix("gs://")
    # "", "." and ".." are legal GCS key segments that the filesystem collapses
    # or resolves away, so two distinct objects would mirror to one path.
    if any(segment in ("", ".", "..") for segment in key.rstrip("/").split("/")):
        raise ValueError(
            f"gcs_uri {gcs_uri!r} has an empty, '.' or '..' segment, which the "
            "local mirror cannot represent as a distinct path."
        )
    return root / key


def _select_models(model_names: list[str], selected: str | None) -> list[str]:
    """The models to back, narrowed to `selected` when one was asked for.

    Raises rather than filtering silently: a name outside the set matches nothing,
    so the loop would back no model, write nothing, and exit clean.
    """
    if selected is None:
        return list(model_names)
    if selected not in model_names:
        raise ValueError(
            f"--model {selected!r} is not in this run's model set {list(model_names)}; "
            "backing nothing would look like a successful run."
        )
    return [selected]


def main() -> None:
    """Compose one Training run's configs, then back each model, publishing both.

    A failing model stops the run rather than being collected: the publish is per
    model and inside the loop, so an earlier model's sidecar is already complete.

    Raises:
        ValueError: If either run id is not path-safe, if exactly one URI override
            was given, if the Feature run published no manifest, if an input URI
            cannot be mirrored to a distinct local path, if `--env` has no
            `environments/<env>.yaml`, if `--model` names no model this run
            composed, on a failed lineage check, or on any composition or
            backtest failure.
        RuntimeError: If the git hash cannot be determined, since a run whose
            commit is unknown cannot be reproduced from its own record.
        ValidationError: If an identity field or an override URI is malformed.
    """
    logging.Formatter.converter = time.gmtime
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03dZ %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    args = _parse_args()

    # Only --run-id becomes a path; the other is checked for one vocabulary.
    run_id = generate_run_id() if args.run_id is None else args.run_id
    require_path_safe_run_id(args.feature_run_id, "--feature-run-id")
    require_path_safe_run_id(run_id, "--run-id")

    project_root = get_project_root_dir()
    config_dir = project_root / "config"
    git_hash = require_git_hash(project_root)

    # Held as a value rather than folded into step_uri: backtest, evaluate, and
    # final_fit will each append their own step name to this same prefix.
    run_prefix = resolve_run_prefix(config_dir, args.env, "train", run_id)
    compose_uri = f"{run_prefix}{_COMPOSE_STEP}/"

    mirror_root = args.scratch_dir
    compose_dir = _mirror_path(compose_uri, mirror_root)

    # Called for the raise. Above the rmtree, and above the resolve's network read.
    compose_train_static_configs(config_dir, args.env)

    artifacts = resolve_feature_artifacts(
        config_dir=config_dir,
        env=args.env,
        feature_run_id=args.feature_run_id,
        panel_uri=args.panel_uri,
        calendar_uri=args.calendar_uri,
    )
    # Before the clear below, so a rejected URI cannot cost the previous output.
    panel_path = _mirror_path(artifacts.panel_uri, mirror_root)
    calendar_path = _mirror_path(artifacts.calendar_uri, mirror_root)

    logger.info(
        "compose_configs starting: run_id=%s feature_run_id=%s out_dir=%s uri=%s",
        run_id,
        args.feature_run_id,
        compose_dir,
        compose_uri,
    )

    # Scratch persists, so a retry under the same --run-id finds stale files.
    # sync_to_gcs matches the prefix to compose_dir; it does not clean it. The
    # backtest step needs no equivalent: its eight filenames are fixed in code,
    # while this step emits one config per model_roles entry.
    if compose_dir.exists():
        shutil.rmtree(compose_dir)
    compose_dir.mkdir(parents=True)

    panel = SourcedPath(
        path=download_from_gcs(artifacts.panel_uri, panel_path.parent),
        uri=artifacts.panel_uri,
    )
    calendar = SourcedPath(
        path=download_from_gcs(artifacts.calendar_uri, calendar_path.parent),
        uri=artifacts.calendar_uri,
    )

    summary = compose_configs_impl(
        config_dir=config_dir,
        env=args.env,
        panel=panel,
        calendar=calendar,
        expected_feature_run_id=args.feature_run_id,
        train_run_id=run_id,
        git_hash=git_hash,
        out_dir=compose_dir,
    )
    # Above both publishes, so a rejected --model leaves the run prefix untouched.
    model_names = _select_models(summary.model_names, args.model)

    # First, so a failed step publish still leaves a record of what the run read.
    # Not sync_to_gcs here: at the run prefix it deletes every sibling step's output.
    upload_to_gcs(compose_dir.parent / "run_identity.json", run_prefix)

    # The impl writes manifest.json last, so publishing it alone and last makes its
    # presence at the prefix mean complete rather than started.
    uploaded, removed = sync_to_gcs(
        compose_dir, compose_uri, completion_marker="manifest.json"
    )

    logger.info(
        "compose_configs complete: uploaded=%d removed=%d uri=%s identity=%s "
        "run_id=%s models=%s n_origins=%d origins=%s..%s "
        "last_complete_actual_month=%d start_months=%s out_dir=%s",
        uploaded,
        removed,
        compose_uri,
        f"{run_prefix}run_identity.json",
        run_id,
        summary.model_names,
        summary.n_origins,
        summary.first_origin,
        summary.last_origin,
        summary.last_complete_actual_month,
        summary.start_months,
        compose_dir,
    )

    for model_name in model_names:
        model_uri = f"{run_prefix}{_BACKTEST_STEP}/{model_name}/"
        # Derived from the URI, like the compose step above, so the published
        # location and the local one cannot disagree.
        model_dir = _mirror_path(model_uri, mirror_root)

        # The staged inputs, not a second download: the lineage check the impl
        # runs must see the frames compose_configs checked.
        backtest_summary = backtest_impl(
            panel_path=panel.path,
            calendar_path=calendar.path,
            compose_configs_dir=compose_dir,
            model_name=model_name,
            out_dir=model_dir,
        )
        # Inside the loop, so a later model failing leaves this sidecar complete.
        uploaded, removed = sync_to_gcs(
            model_dir, model_uri, completion_marker="backtest_manifest.json"
        )

        # Row counts are the one summary field backtest_impl does not log itself.
        logger.info(
            "backtest published: model=%s uploaded=%d removed=%d uri=%s rows=%s",
            model_name,
            uploaded,
            removed,
            model_uri,
            backtest_summary.output_rows,
        )


if __name__ == "__main__":
    main()
