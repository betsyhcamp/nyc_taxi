# No `__future__.annotations`: PEP 563 strings break KFP's annotation read at compile.

import os

from kfp import dsl
from kfp.dsl import Artifact, Dataset, Input, Model, Output

from fcstnyctaxi.lib.container_images import require_digest_ref

# @dsl.component binds base_image at decoration time.
_IMAGE = require_digest_ref(os.environ.get("FCST_TRAIN_IMAGE"), "FCST_TRAIN_IMAGE")


@dsl.component(base_image=_IMAGE, packages_to_install=[])
def final_fit(
    run_prefix: str,
    model_name: str,
    composed_configs: Input[Artifact],
    panel: Input[Dataset],
    calendar: Input[Dataset],
    additional_exog: Input[Dataset],
    bundle: Output[Model],
) -> None:
    """Fit one model on the whole panel and write the bundle a registry consumes.

    One task, so `model_name` is a compile-time constant.

    Args:
        run_prefix (str): The run root, from `compose_configs`; this step appends
            its own name and the model's.
        model_name (str): Selects the composed config and names the bundle.
        composed_configs (Input[Artifact]): The compose step's directory, with
            `run_identity.json` beside it.
        panel (Input[Dataset]): The weekly actuals.
        calendar (Input[Dataset]): The fiscal calendar.
        additional_exog (Input[Dataset]): The exogenous features, read, trimmed and
            joined into the frame the model is fitted on.
        bundle (Output[Model]): This model's bundle directory; the wrapper sets
            its URI before the impl reads the path. `system.Model` is lineage
            only and registers nothing.

    Raises:
        RuntimeError: If the image carries no FCST_GIT_HASH.
        ValueError: On a misplaced bundle, an absent `run_identity.json`, a
            config declaring transforms, a model declaring no callables, a failed
            lineage, column or exogenous-join check, or a save that wrote nothing
            or an empty file.
    """
    import os
    from pathlib import Path

    from fcstnyctaxi.core.train.final_fit_impl import final_fit_impl

    # Unreachable in the DAG: compose_configs reads the same baked value from the
    # same image and fails first. Kept for a direct call.
    git_hash = os.environ.get("FCST_GIT_HASH")
    if not git_hash:
        raise RuntimeError(
            "FCST_GIT_HASH is unset; the image was built without --build-arg "
            "GIT_HASH, so this bundle cannot record the commit that produced it."
        )

    # MUST precede the .path read: .path recomputes from .uri, Model included, and
    # reversed, the run succeeds at KFP's own prefix. The impl never checks the step
    # segment, so a wrong one writes into a sibling step's directory silently.
    bundle.uri = f"{run_prefix}final_fit/{model_name}/"

    summary = final_fit_impl(
        panel_path=Path(panel.path),
        calendar_path=Path(calendar.path),
        additional_exog_path=Path(additional_exog.path),
        compose_configs_dir=Path(composed_configs.path),
        model_name=model_name,
        out_dir=Path(bundle.path),
    )

    # Only what the wrapper holds and the impl does not return.
    bundle.metadata.update(
        {**summary.as_dict(), "model_name": model_name, "git_hash": git_hash}
    )
