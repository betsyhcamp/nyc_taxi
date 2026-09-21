# No `__future__.annotations`: PEP 563 strings break KFP's annotation read at compile.

import os

from kfp import dsl
from kfp.dsl import Artifact, Input, Model, Output

from fcstnyctaxi.lib.container_images import require_digest_ref

# @dsl.component binds base_image at decoration time.
_IMAGE = require_digest_ref(os.environ.get("FCST_TRAIN_IMAGE"), "FCST_TRAIN_IMAGE")


@dsl.component(base_image=_IMAGE, packages_to_install=[])
def register_model(
    serving_container_image_uri: str,
    composed_configs: Input[Artifact],
    bundle: Input[Model],
    registered: Output[Model],
) -> None:
    """Register one run's bundle as a model version, and write the run's record.

    Takes no model name (the impl reads the manifest's, checked against the URI), no
    run_prefix (the output URI is a registry name) and no FCST_GIT_HASH (the impl's
    hash is the label's, so it is the only one).

    Args:
        serving_container_image_uri (str): This task's pinned image, bound at compile
            time. KFP ships the body alone, so `_IMAGE` is absent in the container,
            and no image can bake in its own digest.
        composed_configs (Input[Artifact]): The compose step's directory.
        bundle (Input[Model]): The final fit bundle.
        registered (Output[Model]): The registered version, named by its resource
            name rather than a gs:// path.

    Raises:
        ValueError: On a failed impl check or a git_hash mismatch.
        RuntimeError: If more than one version carries this run id.
    """
    from pathlib import Path

    from fcstnyctaxi.core.train.register_model_impl import register_model_impl
    from fcstnyctaxi.lib.storage_layout import SourcedPath

    summary = register_model_impl(
        bundle=SourcedPath(path=Path(bundle.path), uri=bundle.uri),
        compose_configs_dir=Path(composed_configs.path),
        serving_container_image_uri=serving_container_image_uri,
        run_dir=Path(composed_configs.path).parent,
    )

    # After the impl, which returns it. Legal although not gs://: a non-gs:// URI
    # passes through .path unchanged, and nothing here reads .path.
    registered.uri = summary.model_tag
    registered.metadata.update(summary.as_dict())
