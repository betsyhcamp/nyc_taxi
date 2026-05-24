from pathlib import Path

import pytest
import yaml
from kfp import compiler


def test_fcst_data_ingress_pipeline_compiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "FCSTNYCTAXI_EXTRACT_IMAGE",
        "us-central1-docker.pkg.dev/test-project/test-repo/test:test",
    )

    # Import here (not at module level) so FCSTNYCTAXI_EXTRACT_IMAGE is set before
    # the wrapper's @dsl.component decorator captures _IMAGE at import time.
    from fcstnyctaxi.pipelines.fcst_data_ingress_pipeline import (
        fcst_data_ingress_pipeline,
    )

    out = tmp_path / "pipeline.yaml"
    compiler.Compiler().compile(
        pipeline_func=fcst_data_ingress_pipeline,  # type: ignore[arg-type]
        package_path=str(out),
    )

    intermediate_rep = yaml.safe_load(out.read_text())

    # exact structure of yaml depends on KFP SDK version; may need to be adjusted
    # if fails, print & examine pprint.pprint(intermediate_rep)
    tasks = intermediate_rep["root"]["dag"]["tasks"]

    assert "extract-db-to-bucket" in tasks
