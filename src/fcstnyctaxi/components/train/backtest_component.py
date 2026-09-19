# No `__future__.annotations`: PEP 563 strings break KFP's annotation read at compile.

import os

from kfp import dsl
from kfp.dsl import Artifact, Dataset, Input, Output

from fcstnyctaxi.lib.container_images import require_digest_ref

# @dsl.component binds base_image at decoration time.
_IMAGE = require_digest_ref(os.environ.get("FCST_TRAIN_IMAGE"), "FCST_TRAIN_IMAGE")


@dsl.component(base_image=_IMAGE, packages_to_install=[])
def backtest(
    run_prefix: str,
    model_name: str,
    composed_configs: Input[Artifact],
    panel: Input[Dataset],
    calendar: Input[Dataset],
    sidecar: Output[Artifact],
) -> None:
    """Back one model over one run's origins and write its eight-file sidecar.

    One task per model, so `model_name` is a compile-time constant.

    Args:
        run_prefix (str): The run root, from `compose_configs`; this step appends
            its own name and the model's.
        model_name (str): Selects the composed config and names the sidecar.
        composed_configs (Input[Artifact]): The compose step's directory, with
            `run_identity.json` beside it.
        panel (Input[Dataset]): The weekly actuals.
        calendar (Input[Dataset]): The fiscal calendar.
        sidecar (Output[Artifact]): This model's eight-file directory; the wrapper
            sets its URI before the impl reads the path.

    Raises:
        RuntimeError: If the image carries no FCST_GIT_HASH.
        ValueError: On a misplaced sidecar, an absent `run_identity.json`, or a
            failed lineage or column check.
    """
    import os
    from pathlib import Path

    from fcstnyctaxi.core.train.backtest_impl import backtest_impl

    # Unreachable in the DAG: compose_configs reads the same baked value from the
    # same image and fails first. Kept for a direct call.
    git_hash = os.environ.get("FCST_GIT_HASH")
    if not git_hash:
        raise RuntimeError(
            "FCST_GIT_HASH is unset; the image was built without --build-arg "
            "GIT_HASH, so this sidecar cannot record the commit that produced it."
        )

    # Artifact.path recomputes from self.uri, so this MUST precede the .path read
    # below. Reversed, the run succeeds at KFP's own prefix.
    sidecar.uri = f"{run_prefix}backtest/{model_name}/"

    summary = backtest_impl(
        panel_path=Path(panel.path),
        calendar_path=Path(calendar.path),
        compose_configs_dir=Path(composed_configs.path),
        model_name=model_name,
        out_dir=Path(sidecar.path),
    )

    # Only what the wrapper holds and the impl does not return.
    sidecar.metadata.update(
        {**summary.as_dict(), "model_name": model_name, "git_hash": git_hash}
    )
