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
from typing import cast

from fcstnyctaxi.core.train.backtest_impl import backtest_impl
from fcstnyctaxi.core.train.compose_configs_impl import (
    compose_configs_impl,
    compose_train_static_configs,
)
from fcstnyctaxi.core.train.evaluate_impl import evaluate_impl
from fcstnyctaxi.core.train.final_fit_impl import final_fit_impl
from fcstnyctaxi.core.train.register_model_impl import register_model_impl
from fcstnyctaxi.lib.container_images import require_digest_ref
from fcstnyctaxi.lib.io import (
    delete_from_gcs,
    download_from_gcs,
    require_gcs_uri,
    sync_to_gcs,
    upload_to_gcs,
)
from fcstnyctaxi.lib.run_outputs import resolve_feature_artifacts
from fcstnyctaxi.lib.storage_layout import (
    RUN_OUTPUTS_FILENAME,
    SourcedPath,
    resolve_run_prefix,
)
from fcstnyctaxi.lib.utils import (
    generate_run_id,
    get_project_root_dir,
    require_git_hash,
    require_label_safe_run_id,
    require_path_safe_run_id,
)
from fcstnyctaxi.schemas.config.train import TrainModelingConfig

logger = logging.getLogger(__name__)

# Each step names its own directory under the run root. No step accepts a full
# output path, so these are not parameters. Backtest and final_fit append the model
# name too.
_COMPOSE_STEP = "compose_configs"
_BACKTEST_STEP = "backtest"
_EVALUATE_STEP = "evaluate"
_FINAL_FIT_STEP = "final_fit"

# Shared with the readiness check below, which would otherwise skip every run.
_BACKTEST_MARKER = "backtest_manifest.json"


def _parse_args() -> argparse.Namespace:
    """The environment, the upstream Feature run and its artifacts, and placement.

    `--env` carries no `choices`: unlike the stand-in this reads no hardcoded
    location, so `require_known_environment` is the whole guard.
    """
    parser = argparse.ArgumentParser(
        description="Compose every Training config for one run, back each model, "
        "score the challenger against the benchmark, fit the challenger on the "
        "full panel, then, given --serving-image, register it."
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
        "the panel's feature_run_id column.",
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
        "name one this run composed. Scoring and the final fit run only if both "
        "sidecars are complete.",
    )
    parser.add_argument(
        "--serving-image",
        default=None,
        help="Register the fitted challenger, recording this digest-pinned image as "
        "the runtime that reads its bundle. Absent, the run stops after the final "
        "fit, registers nothing and writes no run_outputs.json.",
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


def _backtest_uri(run_prefix: str, model_name: str) -> str:
    """One model's sidecar prefix, built here alone because the scoring step
    resolves two of them by role, outside the loop that publishes them."""
    return f"{run_prefix}{_BACKTEST_STEP}/{model_name}/"


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
    """Compose one Training run, back each model, score the pair, fit the challenger
    and, given `--serving-image`, register it.

    A failing model stops the run; each sidecar publishes inside the loop, so earlier
    ones are already complete. Scoring reads every sidecar the run id holds, so a
    narrowed rerun that completes the pair scores it without backing the other model
    again. The fit runs only when scoring did, and registration only after the fit.

    Raises:
        ValueError: If a run id or `--serving-image` fails its format check, only one
            URI override is given, the Feature run published no manifest, an input
            URI cannot be mirrored, `--env` has no config file, `--model` names no
            composed model, or any step fails.
        RuntimeError: If the git hash cannot be determined, or more than one
            registered version carries this run id.
        ValidationError: If an identity field or an override URI is malformed.
    """
    logging.Formatter.converter = time.gmtime
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03dZ %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    args = _parse_args()

    # Only --run-id becomes a path and a registry label. The other is Feature's, so
    # it is held to the path vocabulary alone.
    run_id = generate_run_id() if args.run_id is None else args.run_id
    require_path_safe_run_id(args.feature_run_id, "--feature-run-id")
    require_label_safe_run_id(run_id, "--run-id")
    # Here, so a tag rather than a digest is refused before any step spends a thing.
    if args.serving_image is not None:
        require_digest_ref(args.serving_image, "--serving-image")

    project_root = get_project_root_dir()
    config_dir = project_root / "config"
    git_hash = require_git_hash(project_root)

    # Held as a value rather than folded into step_uri: backtest, evaluate, and
    # final_fit each append their own step name to this same prefix.
    run_prefix = resolve_run_prefix(config_dir, args.env, "train", run_id)
    compose_uri = f"{run_prefix}{_COMPOSE_STEP}/"

    mirror_root = args.scratch_dir
    compose_dir = _mirror_path(compose_uri, mirror_root)

    # Above the rmtree and the resolve, so a bad config raises as itself. Roles
    # from here, not evaluate_impl's modeling.yaml, so its role check has two sides.
    _, _, modeling = compose_train_static_configs(config_dir, args.env)
    roles = cast(TrainModelingConfig, modeling.config).model_roles

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
        model_uri = _backtest_uri(run_prefix, model_name)
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
            model_dir, model_uri, completion_marker=_BACKTEST_MARKER
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

    # The run id is the unit of work, not the invocation. backtest_impl deletes its
    # marker first and writes it last, so its presence means a finished sidecar.
    challenger_dir = _mirror_path(
        _backtest_uri(run_prefix, roles.challenger), mirror_root
    )
    benchmark_dir = _mirror_path(
        _backtest_uri(run_prefix, roles.benchmark), mirror_root
    )
    unscored = [
        f"{role} ({name})"
        for role, name, sidecar_dir in (
            ("challenger", roles.challenger, challenger_dir),
            ("benchmark", roles.benchmark, benchmark_dir),
        )
        if not (sidecar_dir / _BACKTEST_MARKER).is_file()
    ]
    if unscored:
        logger.warning(
            "evaluate and final_fit skipped, no complete sidecar for %s under %s. "
            "Any scores or bundle already in that run predate the sidecars it now "
            "holds; rerun without --model to refresh them.",
            unscored,
            run_prefix,
        )
        return

    evaluate_uri = f"{run_prefix}{_EVALUATE_STEP}/"
    evaluate_dir = _mirror_path(evaluate_uri, mirror_root)

    # No model names passed: the impl reads model_roles itself, which is what makes
    # its check that the two directories match the two roles independent of here.
    evaluate_summary = evaluate_impl(
        challenger_dir=challenger_dir,
        benchmark_dir=benchmark_dir,
        compose_configs_dir=compose_dir,
        out_dir=evaluate_dir,
    )
    uploaded, removed = sync_to_gcs(
        evaluate_dir, evaluate_uri, completion_marker="evaluate_manifest.json"
    )

    # Row counts are the one summary field evaluate_impl does not log itself.
    logger.info(
        "evaluate published: uploaded=%d removed=%d uri=%s rows=%s",
        uploaded,
        removed,
        evaluate_uri,
        evaluate_summary.output_rows,
    )

    # After scoring, as in the DAG: a crashed evaluate must not leave a bundle with no
    # scores. The challenger is the registration target whatever evaluate reported.
    final_fit_uri = f"{run_prefix}{_FINAL_FIT_STEP}/{roles.challenger}/"
    final_fit_dir = _mirror_path(final_fit_uri, mirror_root)

    # No rmtree, unlike compose: the impl recreates model/ and fixes its other names.
    final_fit_impl(
        panel_path=panel.path,
        calendar_path=calendar.path,
        compose_configs_dir=compose_dir,
        model_name=roles.challenger,
        out_dir=final_fit_dir,
    )
    uploaded, removed = sync_to_gcs(
        final_fit_dir, final_fit_uri, completion_marker="final_fit_manifest.json"
    )

    # No summary field: final_fit_impl logs the measurements, and the run ids are
    # logged above at compose.
    logger.info(
        "final_fit published: model=%s uploaded=%d removed=%d uri=%s",
        roles.challenger,
        uploaded,
        removed,
        final_fit_uri,
    )

    # What the impl's unlink does through the mount on Vertex, at the same point: a
    # rerun that registers nothing, or fails to, must not leave the earlier run's
    # published record pointing at a version whose bundle was just replaced.
    delete_from_gcs(f"{run_prefix}{RUN_OUTPUTS_FILENAME}")

    # Opt-in: every local run holds live credentials, and each registration is a
    # version in the shared registry.
    if args.serving_image is None:
        logger.info(
            "register_model skipped: no --serving-image, so nothing was registered "
            "and no %s is published under %s.",
            RUN_OUTPUTS_FILENAME,
            run_prefix,
        )
        return

    # The wrapper's arguments: the bundle's mirror paired with the URI it uploads.
    register_summary = register_model_impl(
        bundle=SourcedPath(path=final_fit_dir, uri=final_fit_uri),
        compose_configs_dir=compose_dir,
        serving_container_image_uri=args.serving_image,
        run_dir=compose_dir.parent,
    )
    # Last of every publish, so its presence at the run root means the run finished.
    upload_to_gcs(compose_dir.parent / RUN_OUTPUTS_FILENAME, run_prefix)

    logger.info(
        "register_model published: model_tag=%s uploaded=%s uri=%s",
        register_summary.model_tag,
        register_summary.uploaded,
        f"{run_prefix}{RUN_OUTPUTS_FILENAME}",
    )


if __name__ == "__main__":
    main()
