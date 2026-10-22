import importlib


def test_train_probe_imports_resolve():
    """A renamed impl fails here rather than minutes into a Cloud Build."""
    importlib.import_module("fcstnyctaxi.image_checks.train")
