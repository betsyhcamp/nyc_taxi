# No `__future__.annotations` import here, though 40 other modules carry it
# as house style. KFP reads annotations as objects at decoration time and PEP 563
# makes them strings, so the failure names artifacts rather than the import.

import os
from typing import NamedTuple

from kfp import dsl
from kfp.dsl import Artifact, Dataset, Input, Output


def _require_digest_ref(image: str | None) -> str:
    """The image reference, or a message naming what is wrong with it.

    A pure helper, so the three checks are testable without importlib.reload. It
    judges a value and not the environment that value came from, which is why
    every refusal is a ValueError; the message carries which fix applies.
    """
    if not image:
        raise ValueError(
            "FCST_TRAIN_IMAGE is not set. The compile step must set it to the "
            "pushed digest reference (repo@sha256:...) before importing this module."
        )
    if "@sha256:" not in image:
        raise ValueError(
            f"FCST_TRAIN_IMAGE must be digest-pinned, got {image!r}. A tag can be "
            "repointed, so one compiled spec would execute different code over time."
        )
    if image.count("/") != 3:
        raise ValueError(
            "FCST_TRAIN_IMAGE must use a flat image name "
            f"(<host>/<project>/<repository>/<image>@sha256:...), got {image!r}."
        )
    return image


# No fallback, unlike the extract component. A derived default compiles cleanly and
# then pulls whatever the tag resolves to that day, which is a false success.
_IMAGE = _require_digest_ref(os.environ.get("FCST_TRAIN_IMAGE"))


# The return type is declared inline rather than as a module-level NamedTuple:
# KFP copies only this function's body plus a fixed import preamble into the
# container, so a module-level binding is absent there and the annotation raises
# NameError at def time, before any component code runs and with CI still green.
@dsl.component(base_image=_IMAGE, packages_to_install=[])
def compose_configs(
    env: str,
    train_run_id: str,
    feature_run_id: str,
    panel: Input[Dataset],
    calendar: Input[Dataset],
    composed_configs: Output[Artifact],
) -> NamedTuple("Outputs", [("run_prefix", str)]):  # type: ignore[valid-type]
    """Compose and emit every Training destination for one run.

    Composes EnvironmentConfig itself rather than staying thin: run_prefix needs
    bucket_name, and it has to exist before the impl runs. The rule is the local
    runner's rule: the caller places the work, the impl records it, so both modes
    build the URI through build_run_prefix from the same inputs.

    Returns:
        Outputs: run_prefix, the run root each PR 4 step appends its name to.

    Raises:
        RuntimeError: If the image carries no FCST_GIT_HASH.
        ValueError: If env has no config file, on a failed lineage check, or on
            any composition failure.
    """
    import os
    from pathlib import Path
    from typing import cast

    from fcstnyctaxi.core.train.compose_configs_impl import (
        SourcedPath,
        compose_configs_impl,
        compose_train_static_configs,
    )
    from fcstnyctaxi.lib.io import build_run_prefix

    # The one sanctioned reader of the banned module. Scoped to this line rather
    # than to components/ because per-file-ignores keys on the rule code, and a
    # directory-wide TID251 waiver would also waive the __future__ ban here.
    # tests/ needs no waiver: a wrapper test patches the constant by string
    # target, which ruff never sees as an import.
    from fcstnyctaxi.runtime_paths import CONFIG_DIR  # noqa: TID251
    from fcstnyctaxi.schemas.config.environment import EnvironmentConfig

    # The container has no .git and no git binary, so get_git_hash() would return
    # None and pydantic would report a bad field rather than the missing build arg.
    git_hash = os.environ.get("FCST_GIT_HASH")
    if not git_hash:
        raise RuntimeError(
            "FCST_GIT_HASH is unset; the image was built without "
            "--build-arg GIT_HASH, so this run cannot record the commit that "
            "produced it."
        )

    environment, _, _ = compose_train_static_configs(CONFIG_DIR, env)
    environment_config = cast(EnvironmentConfig, environment.config)
    run_prefix = build_run_prefix(
        bucket=environment_config.storage.bucket_name,
        env=env,
        slice_name="train",
        run_id=train_run_id,
    )
    # Artifact.path is a property recomputed from self.uri on every access, so this
    # assignment MUST precede the .path read below. Reversed, the impl is handed
    # KFP's own location and the run succeeds at the wrong prefix.
    composed_configs.uri = f"{run_prefix}compose_configs/"

    summary = compose_configs_impl(
        config_dir=CONFIG_DIR,
        env=env,
        panel=SourcedPath(path=Path(panel.path), uri=panel.uri),
        calendar=SourcedPath(path=Path(calendar.path), uri=calendar.uri),
        expected_feature_run_id=feature_run_id,
        train_run_id=train_run_id,
        git_hash=git_hash,
        out_dir=Path(composed_configs.path),
    )
    # Every value here is one the wrapper already holds, so stamping reads no
    # artifact payload. panel_uri and calendar_uri are omitted: dsl.importer
    # already records them as this task's inputs in ML Metadata.
    composed_configs.metadata.update(
        {
            **summary.as_dict(),
            "train_run_id": train_run_id,
            "feature_run_id": feature_run_id,
            "git_hash": git_hash,
        }
    )
    # A bare tuple, not a NamedTuple instance: KFP maps it positionally onto the
    # annotation's fields (kfp/dsl/executor.py:271), so no class is needed here.
    return (run_prefix,)
