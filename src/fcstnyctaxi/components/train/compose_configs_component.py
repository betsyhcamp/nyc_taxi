# No `__future__.annotations` import KFP reads annotations as objects at decoration time
# and PEP 563 makes them strings leading to an error in this KFP component at compile.

import os
from typing import NamedTuple

from kfp import dsl
from kfp.dsl import Artifact, Dataset, Input, Output

from fcstnyctaxi.lib.container_images import require_digest_ref

# Module level: @dsl.component binds base_image at decoration time.
_IMAGE = require_digest_ref(os.environ.get("FCST_TRAIN_IMAGE"), "FCST_TRAIN_IMAGE")


# Inline, not a module-level NamedTuple: KFP copies only the body plus a fixed
# preamble into the container, so a module-level binding is absent there and the
# annotation raises NameError at def time, with CI still green.
@dsl.component(base_image=_IMAGE, packages_to_install=[])
def compose_configs(
    env: str,
    train_run_id: str,
    feature_run_id: str,
    declared_model_names: list[str],
    panel: Input[Dataset],
    calendar: Input[Dataset],
    additional_exog: Input[Dataset],
    composed_configs: Output[Artifact],
) -> NamedTuple("Outputs", [("run_prefix", str)]):  # type: ignore[valid-type]
    """Compose and emit every Training destination for one run.

    Resolves run_prefix itself, before the impl runs, because Artifact.path
    recomputes from .uri and the impl cannot supply it. Acts as a wrapper for impl func
    which composes configs.

    Args:
        env (str): Environment selector.
        train_run_id (str): This run's identifier.
        feature_run_id (str): A claim, checked against the panel.
        declared_model_names (list[str]): The compiler's claim, checked against this
            image's baked tree then discarded.
        panel (Input[Dataset]): The actuals, from Feature.
        calendar (Input[Dataset]): The fiscal calendar, same source.
        additional_exog (Input[Dataset]): The exogenous features, same source;
            recorded and compared here, never opened.
        composed_configs (Output[Artifact]): The step directory this run writes.

    Returns:
        Outputs: run_prefix, the run root each Training step appends its name to.

    Raises:
        RuntimeError: If the image carries no FCST_GIT_HASH.
        ValueError: If declared_model_names disagrees with the image's baked set,
            if env has no config file, on a failed lineage check, or on any
            composition failure.
    """
    import os
    from pathlib import Path

    from fcstnyctaxi.core.train.compose_configs_impl import compose_configs_impl
    from fcstnyctaxi.lib.config.bindings import resolve_model_names
    from fcstnyctaxi.lib.storage_layout import SourcedPath, resolve_run_prefix
    from fcstnyctaxi.runtime_paths import CONFIG_DIR  # noqa: TID251

    # The container has no .git and no git binary, so get_git_hash() returns None
    git_hash = os.environ.get("FCST_GIT_HASH")
    if not git_hash:
        raise RuntimeError(
            "FCST_GIT_HASH is unset; the image was built without --build-arg GIT_HASH, "
            "so this run cannot record the commit that produced it."
        )

    # A template from another tree silently backs fewer models than it composes.
    # Ordered: a reorder means two trees. Before the impl writes its marker.
    baked_model_names = resolve_model_names(CONFIG_DIR)
    if tuple(declared_model_names) != baked_model_names:
        raise ValueError(
            f"template declares {tuple(declared_model_names)}, image composes "
            f"{baked_model_names}: different trees. Rebuild and recompile."
        )

    run_prefix = resolve_run_prefix(CONFIG_DIR, env, "train", train_run_id)
    # Artifact.path recomputes from self.uri on every access, so this MUST precede
    # the .path read below. Reversed, the run succeeds at KFP's own prefix.
    composed_configs.uri = f"{run_prefix}compose_configs/"

    summary = compose_configs_impl(
        config_dir=CONFIG_DIR,
        env=env,
        panel=SourcedPath(path=Path(panel.path), uri=panel.uri),
        calendar=SourcedPath(path=Path(calendar.path), uri=calendar.uri),
        additional_exog=SourcedPath(
            path=Path(additional_exog.path), uri=additional_exog.uri
        ),
        expected_feature_run_id=feature_run_id,
        train_run_id=train_run_id,
        git_hash=git_hash,
        out_dir=Path(composed_configs.path),
    )

    composed_configs.metadata.update(
        {
            **summary.as_dict(),
            "train_run_id": train_run_id,
            "feature_run_id": feature_run_id,
            "git_hash": git_hash,
        }
    )
    # KFP maps bare tuple positionally onto annotation's fields so no class is needed.
    return (run_prefix,)
