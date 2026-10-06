from pathlib import Path

from fcstnyctaxi.schemas.storage.common import BUNDLE_MODEL_DIR


def test_the_bundle_model_dir_is_one_segment_below_the_bundle() -> None:
    """A separator or a dot segment would put model-owned files outside the bundle."""
    assert Path(BUNDLE_MODEL_DIR).name == BUNDLE_MODEL_DIR
    assert BUNDLE_MODEL_DIR not in ("", ".", "..")
