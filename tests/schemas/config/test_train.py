from __future__ import annotations

import pytest
from pydantic import ValidationError

from fcstnyctaxi.schemas.config.train import (
    TrainInfraConfig,
    TrainModelingConfig,
)


@pytest.fixture
def valid_infra_dict() -> dict:
    """A complete, valid TrainInfraConfig dict. Each test gets a fresh copy."""
    return {
        "display_name_prefix": "fcst-training-pipeline",
        "feature_source": {
            "panel_filename": "time_series.parquet",
            "calendar_filename": "fiscal_calendar.parquet",
            "pointer_filename": "_latest.json",
        },
        "model_registry": {"display_name": "fcst-monthly-revenue"},
    }


@pytest.fixture
def valid_modeling_dict() -> dict:
    """A complete, valid TrainModelingConfig dict. Each test gets a fresh copy."""
    return {
        "evaluation_periods": {
            "start_months": None,
            "n_start_months": 4,
            "start_month_step": 1,
            "forecast_horizon_months": 2,
        },
        "tiering": {
            "trailing_weeks": 52,
            "tier_labels": ["very_low", "low", "middle", "high", "very_high"],
        },
        "weighting": {"trailing_weeks": 26, "dampening": "cbrt"},
        "model_roles": {"benchmark": "naive", "challenger": "xgboost"},
    }


# ================================================
# TrainInfraConfig
# ================================================


def test_valid_dict_constructs_train_infra_config(valid_infra_dict: dict) -> None:
    """A complete dict constructs TrainInfraConfig and its nested models."""
    config = TrainInfraConfig(**valid_infra_dict)

    assert config.feature_source.panel_filename == "time_series.parquet"
    assert config.model_registry.display_name == "fcst-monthly-revenue"


def test_feature_source_rejects_a_root_key(valid_infra_dict: dict) -> None:
    """feature_source carries filenames only; the directory is derived.

    A `root` key would be a second source of a fact build_feature_uri already
    derives from the env selector and the feature_run_id.
    """
    valid_infra_dict["feature_source"]["root"] = "dev/feature"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TrainInfraConfig(**valid_infra_dict)


# ================================================
# TrainModelingConfig
# ================================================


def test_valid_dict_constructs_train_modeling_config(valid_modeling_dict: dict) -> None:
    """A complete dict constructs TrainModelingConfig and its nested models."""
    config = TrainModelingConfig(**valid_modeling_dict)

    assert config.evaluation_periods.forecast_horizon_months == 2
    assert config.weighting.dampening == "cbrt"
    assert config.model_roles.benchmark == "naive"


def test_start_months_defaults_to_none(valid_modeling_dict: dict) -> None:
    """start_months is the one field where null carries meaning: derive."""
    del valid_modeling_dict["evaluation_periods"]["start_months"]

    config = TrainModelingConfig(**valid_modeling_dict)

    assert config.evaluation_periods.start_months is None


def test_explicit_start_months_pin_an_experiment(valid_modeling_dict: dict) -> None:
    """An explicit list is accepted and preserved verbatim."""
    valid_modeling_dict["evaluation_periods"]["start_months"] = [202504, 202505]

    config = TrainModelingConfig(**valid_modeling_dict)

    assert config.evaluation_periods.start_months == [202504, 202505]


def test_unknown_dampening_name_raises(valid_modeling_dict: dict) -> None:
    """dampening is a closed vocabulary, so a typo fails at composition."""
    valid_modeling_dict["weighting"]["dampening"] = "cuberoot"

    with pytest.raises(ValidationError, match="dampening"):
        TrainModelingConfig(**valid_modeling_dict)


def test_single_tier_label_raises(valid_modeling_dict: dict) -> None:
    """One bin is not tiering, so tier_labels requires at least two."""
    valid_modeling_dict["tiering"]["tier_labels"] = ["only_one"]

    with pytest.raises(ValidationError, match="tier_labels"):
        TrainModelingConfig(**valid_modeling_dict)


def test_duplicate_tier_labels_raise(valid_modeling_dict: dict) -> None:
    """Duplicate labels merge two bins in every downstream groupby.

    The merged result looks correct, which is why this cannot be left to
    inspection and is a schema constraint instead.
    """
    valid_modeling_dict["tiering"]["tier_labels"] = [
        "very_low",
        "low",
        "middle",
        "low",
        "very_high",
    ]

    with pytest.raises(ValidationError, match="tier_labels"):
        TrainModelingConfig(**valid_modeling_dict)


def test_num_tiers_is_not_a_field(valid_modeling_dict: dict) -> None:
    """The bin count is derived from len(tier_labels); there is no num_tiers.

    The pair could never be usefully independent: too few labels makes pd.qcut
    raise, and too many silently mislabels via tier_labels[:effective_tiers].
    """
    valid_modeling_dict["tiering"]["num_tiers"] = 5

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TrainModelingConfig(**valid_modeling_dict)


def test_model_role_name_with_illegal_characters_raises(
    valid_modeling_dict: dict,
) -> None:
    """Model names are file stems now and KFP task-name components later."""
    valid_modeling_dict["model_roles"]["challenger"] = "XGBoost-v2"

    with pytest.raises(ValidationError, match="challenger"):
        TrainModelingConfig(**valid_modeling_dict)


def test_duplicate_model_role_values_are_permitted(valid_modeling_dict: dict) -> None:
    """benchmark == challenger yields WRMAE = 1.0, a legitimate smoke test.

    Deduplication happens at model_names expansion, not here.
    """
    valid_modeling_dict["model_roles"] = {"benchmark": "naive", "challenger": "naive"}

    config = TrainModelingConfig(**valid_modeling_dict)

    assert config.model_roles.benchmark == config.model_roles.challenger
