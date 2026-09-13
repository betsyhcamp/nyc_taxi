# No `__future__.annotations` import KFP reads annotations as objects at decoration time
# and PEP 563 makes them strings leading to an error in this KFP component at compile.

import os
import re
from typing import NamedTuple

from kfp import dsl
from kfp.dsl import Artifact, Dataset, Input, Output

# The Artifact Registry convention: `repository` is the first three segments whatever
# the image name's depth and a nested name is legal rather than banned.
_IMAGE_REF = re.compile(
    r"^(?P<repository>[^/@]+/[^/@]+/[^/@]+)/(?P<name>[^@]+)@sha256:(?P<digest>[0-9a-f]{64})$"
)


def _require_digest_ref(image: str | None) -> str:
    """The image reference, or a ValueError naming the fault. Not inlined at
    _IMAGE, so the checks are testable without importlib.reload."""
    if not image:
        raise ValueError(
            "FCST_TRAIN_IMAGE not set. The compile step must set it to the "
            "pushed digest reference (repo@sha256:...) before importing this module."
        )
    if not _IMAGE_REF.match(image):
        raise ValueError(
            "FCST_TRAIN_IMAGE must be "
            f"<host>/<project>/<repository>/<image>@sha256:<64 hex>, got {image!r}. "
            "A tag can be repointed, so one compiled spec would execute different "
            "code over time."
        )
    return image


_IMAGE = _require_digest_ref(os.environ.get("FCST_TRAIN_IMAGE"))


# Inline, not a module-level NamedTuple: KFP copies only the body plus a fixed
# preamble into the container, so a module-level binding is absent there and the
# annotation raises NameError at def time, with CI still green.
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

    Composes EnvironmentConfig itself: run_prefix needs bucket_name before the impl
    runs. The caller places the work, the impl records it, so both modes build the
    URI from the same inputs.

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
    from fcstnyctaxi.runtime_paths import CONFIG_DIR  # noqa: TID251
    from fcstnyctaxi.schemas.config.environment import EnvironmentConfig

    # The container has no .git and no git binary, so get_git_hash() returns None
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
    # Artifact.path recomputes from self.uri on every access, so this MUST precede
    # the .path read below. Reversed, the run succeeds at KFP's own prefix.
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
    # Every value is one the wrapper already holds, so stamping reads no artifact
    # payload. panel_uri and calendar_uri are omitted: dsl.importer records them.
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
