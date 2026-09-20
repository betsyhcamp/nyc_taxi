# No `__future__.annotations`: PEP 563 strings break KFP's annotation read at compile.

import os

from kfp import dsl
from kfp.dsl import Artifact, Input, Output

from fcstnyctaxi.lib.container_images import require_digest_ref

# @dsl.component binds base_image at decoration time.
_IMAGE = require_digest_ref(os.environ.get("FCST_TRAIN_IMAGE"), "FCST_TRAIN_IMAGE")


@dsl.component(base_image=_IMAGE, packages_to_install=[])
def evaluate(
    run_prefix: str,
    composed_configs: Input[Artifact],
    challenger_sidecar: Input[Artifact],
    benchmark_sidecar: Input[Artifact],
    scores: Output[Artifact],
) -> None:
    """Score one run's challenger against its benchmark and write its five files.

    Takes no model names: the impl re-reads model_roles, so its check has a side
    this wrapper cannot touch.

    Args:
        run_prefix (str): The run root, from `compose_configs`.
        composed_configs (Input[Artifact]): The compose step's directory:
            modeling.yaml, with run_identity.json beside it.
        challenger_sidecar (Input[Artifact]): The challenger's backtest directory.
        benchmark_sidecar (Input[Artifact]): The benchmark's, the same directory
            when one model holds both roles.
        scores (Output[Artifact]): This run's four tables and their manifest.

    Raises:
        RuntimeError: If the image carries no FCST_GIT_HASH.
        ValueError: On a misplaced out_dir, an absent run_identity.json, or a
            failed impl guard.
    """
    import os
    from pathlib import Path

    from fcstnyctaxi.core.train.evaluate_impl import evaluate_impl

    # Unreachable in the DAG: compose_configs fails first. Kept for a direct call.
    git_hash = os.environ.get("FCST_GIT_HASH")
    if not git_hash:
        raise RuntimeError(
            "FCST_GIT_HASH is unset; the image was built without --build-arg "
            "GIT_HASH, so these tables cannot record the commit that produced them."
        )

    # Artifact.path recomputes from self.uri, so this MUST precede the .path read.
    # Reversed, the run succeeds at KFP's own prefix.
    scores.uri = f"{run_prefix}evaluate/"

    summary = evaluate_impl(
        challenger_dir=Path(challenger_sidecar.path),
        benchmark_dir=Path(benchmark_sidecar.path),
        compose_configs_dir=Path(composed_configs.path),
        out_dir=Path(scores.path),
    )

    # Only what the wrapper holds and the impl does not return.
    scores.metadata.update({**summary.as_dict(), "git_hash": git_hash})
