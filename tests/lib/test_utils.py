from __future__ import annotations

import re
from datetime import datetime

from fcstnyctaxi.lib.utils import generate_run_id

# ================================================
# generate_run_id tests
# ================================================


def test_generate_run_id_is_chronological_and_increasing():
    "Test that run_ids are monotonic, chronologically ordered and sortable."
    ids = [generate_run_id() for _ in range(20)]
    assert ids == sorted(ids)


def test_generate_run_id_makes_correct_format():
    """Test that generate_run_id makes desired microsecond format"""
    run_id = generate_run_id()

    assert re.fullmatch(r"\d{8}T\d{12}Z", run_id)

    # strptime raises ValueError if run_id doesn't match format;
    # then pytest catches ValueError & test will fail
    datetime.strptime(run_id, "%Y%m%dT%H%M%S%fZ")
