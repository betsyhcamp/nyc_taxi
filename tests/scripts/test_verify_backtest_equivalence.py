import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts import verify_backtest_equivalence as verify

_BASE_CONFIG: dict[str, Any] = {
    "aggregation": {"calendar_source": None, "period_col": "fiscal_year_month"},
    "data": {"freq": "W-SUN", "id_col": "unique_id"},
    "model": {"fit_predict_callable": "fcstnyctaxi.models.naive.naive_weekly"},
}


def _write_config(directory: Path, config: dict[str, Any]) -> str:
    """Write one sidecar's composed config; return its prefix."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "composed_config.yaml").write_text(yaml.safe_dump(config))
    return f"{directory}/"


@pytest.fixture
def notebook_config() -> dict[str, Any]:
    """The notebook's config: identical but for the calendar URI it injects."""
    config = copy.deepcopy(_BASE_CONFIG)
    config["aggregation"]["calendar_source"] = "gs://bucket/fiscal_calendar.parquet"
    return config


def test_identical_frames_compare_identical() -> None:
    """The baseline the migration gate is trying to produce."""
    frame = pd.DataFrame({"unique_id": ["a", "b"], "y": [1.0, 2.0]})

    passed, detail = verify._compare_frames(frame, frame.copy())

    assert passed
    assert detail == "identical"


def test_a_differing_column_set_names_the_side_it_is_missing_from() -> None:
    """Naming the side is what makes a trim mismatch readable."""
    new_df = pd.DataFrame({"y": [1.0], "only_in_impl": [1]})
    ref_df = pd.DataFrame({"y": [1.0], "only_in_notebook": [1]})

    passed, detail = verify._compare_frames(new_df, ref_df)

    assert not passed
    assert "only_new=['only_in_impl']" in detail
    assert "only_ref=['only_in_notebook']" in detail


def test_identical_nan_bearing_columns_compare_identical() -> None:
    """A bare == treats NaN as unequal to itself; monthly_series carries nulls."""
    frame = pd.DataFrame({"y": [1.0, np.nan, 3.0]})

    passed, detail = verify._compare_frames(frame, frame.copy())

    assert passed, detail


def test_categories_built_in_a_different_order_compare_identical() -> None:
    """The dtype check is blind here: unordered categories compare equal while
    sort_values orders rows by their differing category order."""
    new_df = pd.DataFrame(
        {"tier": pd.Categorical(["low", "high"], categories=["low", "high"])}
    )
    ref_df = pd.DataFrame(
        {"tier": pd.Categorical(["low", "high"], categories=["high", "low"])}
    )
    assert new_df["tier"].dtype == ref_df["tier"].dtype

    passed, detail = verify._compare_frames(new_df, ref_df)

    assert passed, detail


def test_a_dtype_difference_is_reported_without_claiming_the_values_differ() -> None:
    """Series.equals would report both, which is why the value test is not it."""
    new_df = pd.DataFrame({"n": pd.Series([1, 2], dtype="int64")})
    ref_df = pd.DataFrame({"n": pd.Series([1.0, 2.0], dtype="float64")})

    passed, detail = verify._compare_frames(new_df, ref_df)

    assert not passed
    assert "dtype mismatches" in detail
    assert "values differ" not in detail


def test_a_numeric_difference_names_its_max_absolute_delta() -> None:
    """The magnitude separates a port bug from float noise."""
    new_df = pd.DataFrame({"y": [1.0, 2.0]})
    ref_df = pd.DataFrame({"y": [1.0, 2.5]})

    passed, detail = verify._compare_frames(new_df, ref_df)

    assert not passed
    assert "max abs delta 0.5" in detail


def test_differing_row_counts_stop_before_the_value_comparison() -> None:
    """Aligning unequal frames raises, so the length check must return first."""
    new_df = pd.DataFrame({"y": [1.0, 2.0]})
    ref_df = pd.DataFrame({"y": [1.0, 2.0, 3.0]})

    passed, detail = verify._compare_frames(new_df, ref_df)

    assert not passed
    assert "row counts differ" in detail


def test_a_leaf_difference_returns_a_path_the_caller_can_match_on() -> None:
    """The expected-difference filter needs no parsing."""
    new_cfg = {"aggregation": {"calendar_source": None}}
    ref_cfg = {"aggregation": {"calendar_source": "gs://bucket/calendar.parquet"}}

    diffs = verify._dict_diff(new_cfg, ref_cfg)

    assert [path for path, _ in diffs] == ["aggregation.calendar_source"]


def test_a_key_present_on_only_one_side_names_which_side_lacks_it() -> None:
    """A dropped config block is a different failure from a changed value."""
    diffs = verify._dict_diff({"a": 1}, {"a": 1, "b": 2})

    assert diffs == [("b", "missing in new")]


def test_the_expected_config_difference_alone_passes(
    tmp_path: Path, notebook_config: dict[str, Any]
) -> None:
    """The gate's prediction: exactly this difference and nothing else."""
    impl = _write_config(tmp_path / "impl", _BASE_CONFIG)
    notebook = _write_config(tmp_path / "notebook", notebook_config)

    passed, detail = verify._compare_configs(impl, notebook)

    assert passed, detail
    assert "1 of 1" in detail


def test_an_unexpected_config_difference_fails_and_names_its_path(
    tmp_path: Path, notebook_config: dict[str, Any]
) -> None:
    """A second difference means the two sides composed different models."""
    notebook_config["model"]["fit_predict_callable"] = "models.naive.naive_weekly"
    impl = _write_config(tmp_path / "impl", _BASE_CONFIG)
    notebook = _write_config(tmp_path / "notebook", notebook_config)

    passed, detail = verify._compare_configs(impl, notebook)

    assert not passed
    assert "model.fit_predict_callable" in detail


def test_an_absent_expected_difference_is_reported_without_failing(
    tmp_path: Path,
) -> None:
    """It says nothing about whether the numbers reproduce."""
    impl = _write_config(tmp_path / "impl", _BASE_CONFIG)
    notebook = _write_config(tmp_path / "notebook", _BASE_CONFIG)

    passed, detail = verify._compare_configs(impl, notebook)

    assert passed
    assert "0 of 1" in detail


def test_a_sidecar_prefix_carrying_no_completion_marker_is_refused(
    tmp_path: Path,
) -> None:
    """A marker-less prefix can hold a mix of two attempts."""
    (tmp_path / "monthly_series.parquet").write_text("not empty, and not trustworthy")

    with pytest.raises(ValueError, match="backtest_manifest.json"):
        verify._require_completed_impl_sidecar(f"{tmp_path}/")


def test_a_completed_sidecar_prefix_is_accepted(tmp_path: Path) -> None:
    """A local prefix proves it without credentials, since either scheme works."""
    (tmp_path / "backtest_manifest.json").write_text("{}")

    verify._require_completed_impl_sidecar(f"{tmp_path}/")


def test_a_prefix_without_a_trailing_slash_names_the_flag_that_carried_it() -> None:
    """Normalizing would hide it and the marker check would blame the sidecar."""
    with pytest.raises(ValueError, match="--impl-prefix"):
        verify._require_prefix("gs://bucket/run/backtest/naive", "--impl-prefix")
