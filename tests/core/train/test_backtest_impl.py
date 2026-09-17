import dataclasses
from typing import Any, cast

import pandas as pd
import pytest
from tsbricks.backtesting import generate_folds
from tsbricks.backtesting.schema import BacktestConfig

from fcstnyctaxi.core.train.backtest_impl import compute_backtest_outputs
from fcstnyctaxi.lib.backtest_results import build_cv_results
from fcstnyctaxi.lib.config.bindings import train_modeling_bindings
from fcstnyctaxi.lib.config.composition import compose_config, merge_configs
from fcstnyctaxi.lib.monthly_aggregation import attach_tier_and_weight
from fcstnyctaxi.lib.utils import get_project_root_dir
from fcstnyctaxi.schemas.config.train import TrainModelingConfig

CONFIG_DIR = get_project_root_dir() / "config"

# ================================================
# Fixtures
#
# 16 weeks, W-SUN, four fiscal months of four each (202501..202504). Both origins
# sit at a month end with horizon 2, so every fold forecasts into the next month.
#
# The model is a fake resolved by dotted path, as in tests/lib/test_calibration.py:
# these tests prove the assertions fire, not that any number is right.
# ================================================

_WEEKS = pd.date_range("2025-01-05", periods=16, freq="W-SUN")
_SERIES = {"a": 10.0, "b": 5.0, "c": 1.0}

_fit_calls: list[int] = []
"""One entry per fake-model call, so a test can assert no fold ran."""


@pytest.fixture(autouse=True)
def _reset_fit_calls() -> None:
    """Module state, so every test starts from empty."""
    _fit_calls.clear()


def _fake_model_callable(
    train_df: pd.DataFrame, horizon: int, future_x_df: pd.DataFrame, **kwargs: Any
) -> pd.DataFrame:
    """Repeat each series' last observed value, and record that a fold ran."""
    _fit_calls.append(horizon)

    last_ds = train_df["ds"].max()
    future = (
        future_x_df.loc[future_x_df["ds"] > last_ds, "ds"]
        .sort_values()
        .head(horizon)
        .tolist()
    )
    last_value = train_df.sort_values("ds").groupby("unique_id")["y"].last()
    ids = sorted(train_df["unique_id"].unique())

    return pd.DataFrame(
        {
            "unique_id": [uid for uid in ids for _ in future],
            "ds": future * len(ids),
            "ypred": [float(last_value[uid]) for uid in ids for _ in future],
        }
    )


@pytest.fixture
def calendar_df() -> pd.DataFrame:
    """Only the columns the fold loop reads, not the full calendar contract."""
    week_of_month = [i % 4 + 1 for i in range(len(_WEEKS))]
    return pd.DataFrame(
        {
            "ds": _WEEKS,
            "fiscal_year_month": [202501 + i // 4 for i in range(len(_WEEKS))],
            "origin_month_fraction_elapsed": [w / 4 for w in week_of_month],
        }
    )


@pytest.fixture
def ts_df() -> pd.DataFrame:
    """Three series with distinct trailing means, so qcut gets three bins."""
    return pd.DataFrame(
        {
            "unique_id": [uid for uid in _SERIES for _ in _WEEKS],
            "ds": list(_WEEKS) * len(_SERIES),
            "y": [value for value in _SERIES.values() for _ in _WEEKS],
        }
    )


@pytest.fixture
def modeling() -> TrainModelingConfig:
    """The shipped tiering and weighting, composed rather than hand-built."""
    return cast(
        TrainModelingConfig,
        compose_config(CONFIG_DIR, train_modeling_bindings()).config,
    )


def _cfg(origins: list[tuple[int, int]]) -> BacktestConfig:
    """Compose off the production tree; origins are (week index, horizon)."""
    return merge_configs(
        CONFIG_DIR / "base" / "data.yaml",
        CONFIG_DIR / "train" / "backtest.yaml",
        {
            "model": {
                "fit_predict_callable": "test_backtest_impl._fake_model_callable",
                "hyperparameters": {"freq": "W-SUN"},
            },
            "cross_validation": {
                "forecast_origins": [
                    {"origin": str(_WEEKS[index].date()), "horizon": horizon}
                    for index, horizon in origins
                ]
            },
        },
    )


_TWO_ORIGINS = [(7, 2), (11, 2)]


# ================================================
# compute_backtest_outputs: the loop runs
# ================================================


def test_the_fold_loop_produces_one_origin_per_configured_origin(
    ts_df: pd.DataFrame, calendar_df: pd.DataFrame, modeling: TrainModelingConfig
) -> None:
    """The fixture's own check: without it a broken fixture reads as a broken check."""
    outputs = compute_backtest_outputs(
        cfg=_cfg(_TWO_ORIGINS), modeling=modeling, ts_df=ts_df, calendar_df=calendar_df
    )

    assert len(_fit_calls) == len(_TWO_ORIGINS)
    assert outputs.raw_cv_forecasts["forecast_origin_date"].nunique() == len(
        _TWO_ORIGINS
    )
    assert not outputs.monthly_series.empty


# ================================================
# The four assertions
#
# Only the repeated-origins check is reachable through cfg and data. The other three
# replace a collaborator in the impl's namespace, as tests/lib/test_calibration.py
# does. All four run the real function: the defect they catch is an unwired check,
# which a direct call to the predicate would pass.
# ================================================


def test_repeated_forecast_origins_raise_before_the_loop(
    ts_df: pd.DataFrame, calendar_df: pd.DataFrame, modeling: TrainModelingConfig
) -> None:
    """BacktestConfig rejects a repeated pair, so the reachable case is two horizons."""
    with pytest.raises(ValueError, match="forecast origins repeat"):
        compute_backtest_outputs(
            cfg=_cfg([(7, 2), (7, 3)]),
            modeling=modeling,
            ts_df=ts_df,
            calendar_df=calendar_df,
        )

    assert _fit_calls == []


def test_a_fold_count_disagreeing_with_the_pairs_raises_before_the_loop(
    ts_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    modeling: TrainModelingConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """generate_folds emits one fold per pair, so only an upstream change does this."""

    def _drop_the_last_fold(
        df: pd.DataFrame, cv_config: Any, data_config: Any, test_config: Any = None
    ) -> tuple[dict, dict | None]:
        folds, test_split = generate_folds(df, cv_config, data_config, test_config)
        return dict(list(folds.items())[:-1]), test_split

    monkeypatch.setattr(
        "fcstnyctaxi.core.train.backtest_impl.generate_folds", _drop_the_last_fold
    )

    with pytest.raises(ValueError, match="origin/horizon"):
        compute_backtest_outputs(
            cfg=_cfg(_TWO_ORIGINS),
            modeling=modeling,
            ts_df=ts_df,
            calendar_df=calendar_df,
        )

    assert _fit_calls == []


def test_an_unmapped_fold_id_raises_instead_of_nulling_the_origin(
    ts_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    modeling: TrainModelingConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """.map() returns NaT for a missing key, silently nulling an output's key column."""

    def _cv_results_missing_one_origin(**kwargs: Any) -> Any:
        cv_results = build_cv_results(**kwargs)
        first, *_ = cv_results.fold_id_to_origin
        return dataclasses.replace(
            cv_results,
            fold_id_to_origin={first: cv_results.fold_id_to_origin[first]},
        )

    monkeypatch.setattr(
        "fcstnyctaxi.core.train.backtest_impl.build_cv_results",
        _cv_results_missing_one_origin,
    )

    with pytest.raises(ValueError, match="null forecast_origin_date"):
        compute_backtest_outputs(
            cfg=_cfg(_TWO_ORIGINS),
            modeling=modeling,
            ts_df=ts_df,
            calendar_df=calendar_df,
        )


def test_duplicate_monthly_series_keys_raise(
    ts_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    modeling: TrainModelingConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A duplicated key inflates every weighted sum while the run looks healthy."""

    def _attach_twice(**kwargs: Any) -> pd.DataFrame:
        rows = attach_tier_and_weight(**kwargs)
        return pd.concat([rows, rows], ignore_index=True)

    monkeypatch.setattr(
        "fcstnyctaxi.core.train.backtest_impl.attach_tier_and_weight", _attach_twice
    )

    # One origin, so the duplication is this test's and not a repeated origin.
    with pytest.raises(ValueError, match="duplicate key row"):
        compute_backtest_outputs(
            cfg=_cfg([(7, 2)]),
            modeling=modeling,
            ts_df=ts_df,
            calendar_df=calendar_df,
        )
