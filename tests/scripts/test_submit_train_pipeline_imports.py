import os
import subprocess
import sys
import textwrap

from fcstnyctaxi.lib.utils import get_project_root_dir

# `from google.cloud import aiplatform` resolves a sys.modules stub: CPython's
# from-list handler falls back to sys.modules when the attribute lookup fails.
# Stubbed for precision, not speed: the SDK's import cost is a separate question.
_PROBE = textwrap.dedent(
    """
    import sys, types
    sys.modules["google.cloud.aiplatform"] = types.ModuleType("aiplatform")

    import scripts.submit_train_pipeline  # noqa: F401

    # The import already raises if it reaches the component's digest check.
    # This outlives that check: a DSL import fails here even if the raise moves.
    reached = sorted(
        name
        for name in sys.modules
        if name == "kfp"
        or name.startswith(("kfp.", "fcstnyctaxi.components", "fcstnyctaxi.pipelines"))
    )
    if reached:
        raise SystemExit(f"the submitter imported {reached}, which needs an image")
    print("clean")
    """
)


def test_the_submitter_imports_with_no_image_reference_set() -> None:
    """Test that importing the submitter needs no FCST_TRAIN_IMAGE. In process this
    would pass against a submitter that imports the pipeline
    """
    environment = dict(os.environ)
    environment.pop("FCST_TRAIN_IMAGE", None)

    completed = subprocess.run(
        [sys.executable, "-c", _PROBE],
        # `scripts` resolves from the interpreter's path, which for -c is the cwd.
        cwd=get_project_root_dir(),
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "clean" in completed.stdout
