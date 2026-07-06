from __future__ import annotations

import pandas as pd
import pytest
from pytest_mock import MockerFixture
from tsbricks.backtesting import BacktestResults, CVResults

from fcstnyctaxi.lib.backtest_results import build_backtest_results, build_cv_results

# Two chronologically-ordered folds, reused across multiple tests
_ORIGIN_HORIZON_PAIRS: list[tuple] = [
    (pd.Timestamp("2025-04-20"), 2),
    (pd.Timestamp("2025-04-27"), 1),
]


@pytest.fixture
def sample_forecasts_per_fold() -> dict[str, pd.DataFrame]:
    """Forecast DataFrame per fold, keyed fold_0/fold_1 (chronological order)."""
    return {
        "fold_0": pd.DataFrame(
            {"unique_id": [1], "ds": [pd.Timestamp("2025-04-27")], "ypred": [10.0]}
        ),
        "fold_1": pd.DataFrame(
            {"unique_id": [1], "ds": [pd.Timestamp("2025-05-04")], "ypred": [11.0]}
        ),
    }


@pytest.fixture
def sample_train_val_splits_per_fold() -> dict[str, dict[str, pd.DataFrame]]:
    """Train/val split per fold, keyed to sample_forecasts_per_fold."""
    return {
        "fold_0": {
            "train": pd.DataFrame(
                {"unique_id": [1], "ds": [pd.Timestamp("2025-04-20")], "y": [9.0]}
            ),
            "val": pd.DataFrame(
                {"unique_id": [1], "ds": [pd.Timestamp("2025-04-27")], "y": [10.0]}
            ),
        },
        "fold_1": {
            "train": pd.DataFrame(
                {"unique_id": [1], "ds": [pd.Timestamp("2025-04-27")], "y": [10.0]}
            ),
            "val": pd.DataFrame(
                {"unique_id": [1], "ds": [pd.Timestamp("2025-05-04")], "y": [11.0]}
            ),
        },
    }


@pytest.fixture
def sample_metrics() -> pd.DataFrame:
    """Minimal long-format metrics DataFrame; its contents aren't inspected."""
    return pd.DataFrame(
        {
            "metric_name": ["mae", "mae"],
            "unique_id": [1, 1],
            "fold": ["fold_0", "fold_1"],
            "value": [1.0, 1.0],
        }
    )


@pytest.fixture
def sample_cv_results(
    sample_forecasts_per_fold: dict[str, pd.DataFrame],
    sample_train_val_splits_per_fold: dict[str, dict[str, pd.DataFrame]],
    sample_metrics: pd.DataFrame,
) -> CVResults:
    """A valid CVResults built from the aligned sample fixtures above."""
    return build_cv_results(
        forecasts_per_fold=sample_forecasts_per_fold,
        train_val_splits_per_fold=sample_train_val_splits_per_fold,
        metrics=sample_metrics,
        origin_horizon_pairs=_ORIGIN_HORIZON_PAIRS,
    )


# ================================================
# build_cv_results — happy path
# ================================================


def test_build_cv_results_returns_cv_results_with_expected_fields(
    sample_forecasts_per_fold: dict[str, pd.DataFrame],
    sample_train_val_splits_per_fold: dict[str, dict[str, pd.DataFrame]],
    sample_metrics: pd.DataFrame,
) -> None:
    """Aligned inputs assemble into a CVResults with matching fold_id_to_origin."""
    result = build_cv_results(
        forecasts_per_fold=sample_forecasts_per_fold,
        train_val_splits_per_fold=sample_train_val_splits_per_fold,
        metrics=sample_metrics,
        origin_horizon_pairs=_ORIGIN_HORIZON_PAIRS,
    )

    assert isinstance(result, CVResults)
    assert result.forecasts_per_fold is sample_forecasts_per_fold
    assert result.train_val_splits_per_fold is sample_train_val_splits_per_fold
    assert result.fold_origins == [origin for origin, _ in _ORIGIN_HORIZON_PAIRS]
    assert result.fold_id_to_origin == {
        "fold_0": pd.Timestamp("2025-04-20"),
        "fold_1": pd.Timestamp("2025-04-27"),
    }


def test_build_cv_results_forwards_optional_fields(
    sample_forecasts_per_fold: dict[str, pd.DataFrame],
    sample_train_val_splits_per_fold: dict[str, dict[str, pd.DataFrame]],
    sample_metrics: pd.DataFrame,
) -> None:
    """Unrecognized kwargs (e.g. fitted_values) pass straight through to CVResults."""
    fitted_values = {
        "fold_0": pd.DataFrame(
            {"unique_id": [1], "ds": [pd.Timestamp("2025-04-20")], "fitted": [9.5]}
        )
    }

    result = build_cv_results(
        forecasts_per_fold=sample_forecasts_per_fold,
        train_val_splits_per_fold=sample_train_val_splits_per_fold,
        metrics=sample_metrics,
        origin_horizon_pairs=_ORIGIN_HORIZON_PAIRS,
        fitted_values=fitted_values,
    )

    assert result.fitted_values is fitted_values


# ================================================
# build_cv_results — validation / error paths
# ================================================


def test_build_cv_results_raises_on_fold_count_mismatch(
    sample_train_val_splits_per_fold: dict[str, dict[str, pd.DataFrame]],
    sample_metrics: pd.DataFrame,
) -> None:
    """origin_horizon_pairs longer than forecasts_per_fold triggers the length check."""

    forecasts_per_fold_dict = {
        "fold_0": pd.DataFrame(
            {"unique_id": [1], "ds": [pd.Timestamp("2025-04-27")], "ypred": [10.0]}
        ),
    }

    with pytest.raises(ValueError, match="lengths"):
        build_cv_results(
            forecasts_per_fold=forecasts_per_fold_dict,
            train_val_splits_per_fold=sample_train_val_splits_per_fold,
            metrics=sample_metrics,
            origin_horizon_pairs=_ORIGIN_HORIZON_PAIRS,
        )


def test_build_cv_results_raises_on_fold_id_mismatch(
    sample_train_val_splits_per_fold: dict[str, dict[str, pd.DataFrame]],
    sample_metrics: pd.DataFrame,
) -> None:
    """Same fold count but different fold_id keys triggers the identity check."""
    forecasts_per_fold_dict = {
        "fold_0": pd.DataFrame(
            {"unique_id": [1], "ds": [pd.Timestamp("2025-04-27")], "ypred": [10.0]}
        ),
        "fold_2": pd.DataFrame(
            {"unique_id": [1], "ds": [pd.Timestamp("2025-05-04")], "ypred": [11.0]}
        ),
    }

    with pytest.raises(ValueError, match="Identities"):
        build_cv_results(
            forecasts_per_fold=forecasts_per_fold_dict,
            train_val_splits_per_fold=sample_train_val_splits_per_fold,
            metrics=sample_metrics,
            origin_horizon_pairs=_ORIGIN_HORIZON_PAIRS,
        )


# ================================================
# build_backtest_results — happy path
# ================================================


def test_build_backtest_results_sets_horizon_from_origin_horizon_pairs(
    sample_cv_results: CVResults,
) -> None:
    """horizon is built from origin_horizon_pairs directly, not derived from
    cv.fold_id_to_origin (which has no horizon information to reconstruct)."""
    result = build_backtest_results(
        sample_cv_results,
        config={},
        origin_horizon_pairs=_ORIGIN_HORIZON_PAIRS,
        capture_lineage=False,
    )

    assert isinstance(result, BacktestResults)
    assert result.horizon == _ORIGIN_HORIZON_PAIRS


def test_build_backtest_results_capture_lineage_false_leaves_lineage_none(
    sample_cv_results: CVResults,
) -> None:
    """capture_lineage=False skips git_hash/uv_lock_info population entirely."""
    result = build_backtest_results(
        sample_cv_results,
        config={},
        origin_horizon_pairs=_ORIGIN_HORIZON_PAIRS,
        capture_lineage=False,
    )

    assert result.git_hash is None
    assert result.uv_lock_info is None


def test_build_backtest_results_capture_lineage_true_calls_metadata_helpers(
    sample_cv_results: CVResults,
    mocker: MockerFixture,
) -> None:
    """capture_lineage=True (the default) populates git_hash/uv_lock_info via
    the same tsbricks.blocks.metadata helpers run_backtest() uses internally."""
    mock_git_hash = mocker.patch(
        "fcstnyctaxi.lib.backtest_results.get_git_hash", return_value="abc123"
    )
    mock_uv_lock_info = mocker.patch(
        "fcstnyctaxi.lib.backtest_results.get_uv_lock_info",
        return_value={"path": "uv.lock", "sha256": "deadbeef"},
    )

    result = build_backtest_results(
        sample_cv_results, config={}, origin_horizon_pairs=_ORIGIN_HORIZON_PAIRS
    )

    mock_git_hash.assert_called_once()
    mock_uv_lock_info.assert_called_once()
    assert result.git_hash == "abc123"
    assert result.uv_lock_info == {"path": "uv.lock", "sha256": "deadbeef"}


def test_build_backtest_results_forwards_optional_fields(
    sample_cv_results: CVResults,
) -> None:
    """Unrecognized kwargs (e.g. extra) pass straight through to BacktestResults."""
    result = build_backtest_results(
        sample_cv_results,
        config={},
        origin_horizon_pairs=_ORIGIN_HORIZON_PAIRS,
        capture_lineage=False,
        extra={"note": "smoke test"},
    )

    assert result.extra == {"note": "smoke test"}
