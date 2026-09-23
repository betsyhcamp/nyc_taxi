import pytest
from pydantic import ValidationError

from fcstnyctaxi.schemas.config.train import (
    ModelSettings,
    TrainInfraConfig,
    TrainModelingConfig,
)
from fcstnyctaxi.schemas.run_outputs import CALENDAR_ALLOWED_COLUMNS, JOIN_KEYS


@pytest.fixture
def valid_infra_dict() -> dict:
    """A complete, valid TrainInfraConfig dict. Each test gets a fresh copy."""
    return {
        "display_name_prefix": "fcst-train-pipeline",
        "model_registry": {
            "display_name_prefix": "fcst-monthly-revenue",
            "model_id_prefix": "fcst-monthly-revenue",
        },
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
        "model_settings": {
            "naive": {"exog_features": []},
            "xgboost": {
                "exog_features": ["fiscal_month"],
                "fit_callable": "a.b.fit",
                "save_callable": "a.b.save",
            },
        },
    }


# ================================================
# TrainInfraConfig
# ================================================


def test_valid_dict_constructs_train_infra_config(valid_infra_dict: dict) -> None:
    """A complete dict constructs TrainInfraConfig and its nested models."""
    config = TrainInfraConfig(**valid_infra_dict)

    assert config.display_name_prefix == "fcst-train-pipeline"
    assert config.model_registry.display_name_prefix == "fcst-monthly-revenue"
    assert config.model_registry.model_id_prefix == "fcst-monthly-revenue"


def test_train_infra_rejects_an_unknown_key(valid_infra_dict: dict) -> None:
    """extra="forbid" refuses a stray key rather than silently dropping it. Asserts
    the refusal, not the field list, so adding a real field cannot break it."""
    valid_infra_dict["output_bucket"] = "nyc-taxi-ehc--modeling"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TrainInfraConfig(**valid_infra_dict)


@pytest.mark.parametrize(
    "prefix",
    [
        "9fcst",  # a composed id may not start with a digit
        "-fcst",  # nor a hyphen
        "_fcst",  # legal first character in a model_id, but not in the prefix
        "Fcst",  # uppercase, which no SDK check stops before the service
        "fcstRevenue",
        "fcst.revenue",
        "fcst revenue",
        "fcst_revenue",
        "fcst-",  # the `--` seam at the join
        "",
    ],
)
def test_model_id_prefix_rejects_a_prefix_the_pattern_excludes(
    valid_infra_dict: dict, prefix: str
) -> None:
    """Each case breaks one clause: first character, body charset, or the seam."""
    valid_infra_dict["model_registry"]["model_id_prefix"] = prefix

    with pytest.raises(ValidationError, match="model_id_prefix"):
        TrainInfraConfig(**valid_infra_dict)


@pytest.mark.parametrize("prefix", ["a", "fcst2", "fcst-monthly-revenue"])
def test_model_id_prefix_accepts_a_prefix_that_composes_a_clean_id(
    valid_infra_dict: dict, prefix: str
) -> None:
    """The single letter is the pattern's optional tail, easiest to break."""
    valid_infra_dict["model_registry"]["model_id_prefix"] = prefix

    assert TrainInfraConfig(**valid_infra_dict).model_registry.model_id_prefix == prefix


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


def test_empty_start_months_raises(valid_modeling_dict: dict) -> None:
    """`null` is the only way to say derive.

    An empty list would mean the same thing today, since the consumer spells
    this `cfg.start_months or derive_start_months(...)`. Forbidding it makes
    that expression equivalent to `is not None`, so narrowing the idiom later
    cannot silently turn [] into "zero months" — a failure that would surface on
    forecast_origins, two layers from the edit that caused it.
    """
    valid_modeling_dict["evaluation_periods"]["start_months"] = []

    with pytest.raises(ValidationError, match="start_months"):
        TrainModelingConfig(**valid_modeling_dict)


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
    """benchmark == challenger yields WRMAE = 1.0, a legitimate smoke test."""
    valid_modeling_dict["model_roles"] = {
        "benchmark": "xgboost",
        "challenger": "xgboost",
    }

    config = TrainModelingConfig(**valid_modeling_dict)

    assert config.model_roles.benchmark == config.model_roles.challenger


# ================================================
# ModelSettings and its validators
# ================================================


def test_a_model_may_declare_features_without_callables(
    valid_modeling_dict: dict,
) -> None:
    """Only a registration candidate needs the callable pair."""
    valid_modeling_dict["model_settings"]["naive"] = {
        "exog_features": ["count_workdays"]
    }

    config = TrainModelingConfig(**valid_modeling_dict)

    assert config.model_settings["naive"].fit_callable is None


@pytest.mark.parametrize("declared", ["fit_callable", "save_callable"])
def test_half_a_callable_pair_raises(declared: str) -> None:
    """A fit with no save persists nothing, and the reverse has nothing to persist."""
    with pytest.raises(ValidationError, match="both or neither"):
        ModelSettings(**{declared: "a.b.c"})


def test_an_exog_feature_outside_both_contracts_raises() -> None:
    """The gap a closed submodel alone leaves open: its contents."""
    with pytest.raises(ValidationError, match="in neither the calendar nor"):
        ModelSettings(exog_features=["fiscal_wek_of_month"])


# Derived, not named: the point is that one of each contract's own columns is
# accepted, and `ds` heads the calendar tuple but is a join key.
_A_CALENDAR_FEATURE = next(
    column for column in CALENDAR_ALLOWED_COLUMNS if column not in JOIN_KEYS
)


@pytest.mark.parametrize(
    "feature", [_A_CALENDAR_FEATURE, "week_sin", "holiday_days_in_week"]
)
def test_a_feature_either_contract_declares_is_accepted(feature: str) -> None:
    """The join delivers both contracts, so naming from either must validate."""
    assert ModelSettings(exog_features=[feature]).exog_features == [feature]


def test_the_selectable_set_the_message_offers_excludes_the_join_keys() -> None:
    """The check above rejects a join key, so offering one here would contradict it."""
    with pytest.raises(ValidationError) as unknown:
        ModelSettings(exog_features=["fiscal_wek_of_month"])

    offered = unknown.value.errors()[0]["msg"]
    for key in JOIN_KEYS:
        assert f"'{key}'" not in offered


@pytest.mark.parametrize("key", JOIN_KEYS)
def test_an_exog_feature_naming_a_join_key_raises_as_a_key(key: str) -> None:
    """Both keys must be named as keys, not as misspellings."""
    with pytest.raises(ValidationError, match="join key"):
        ModelSettings(exog_features=[key])


def test_the_two_exog_feature_faults_do_not_share_a_message() -> None:
    """Different fixes, so one message would misdirect the reader."""
    with pytest.raises(ValidationError) as unknown:
        ModelSettings(exog_features=["fiscal_wek_of_month"])
    with pytest.raises(ValidationError) as join_key:
        ModelSettings(exog_features=["ds"])

    assert unknown.value.errors()[0]["msg"] != join_key.value.errors()[0]["msg"]


def test_a_role_model_with_no_settings_entry_raises(
    valid_modeling_dict: dict,
) -> None:
    """Every role model is backtested, so every one needs its exog_features."""
    del valid_modeling_dict["model_settings"]["naive"]

    with pytest.raises(ValidationError, match="no entry for role model"):
        TrainModelingConfig(**valid_modeling_dict)


def test_a_settings_entry_with_no_role_is_permitted(
    valid_modeling_dict: dict,
) -> None:
    """A model config can exist with no role, so a parallel entry can too."""
    valid_modeling_dict["model_settings"]["unrostered"] = {"exog_features": []}

    config = TrainModelingConfig(**valid_modeling_dict)

    assert "unrostered" in config.model_settings


def test_a_challenger_without_the_callable_pair_raises(
    valid_modeling_dict: dict,
) -> None:
    """final_fit fits the challenger, so only it carries this obligation."""
    valid_modeling_dict["model_settings"]["xgboost"] = {"exog_features": []}

    with pytest.raises(ValidationError, match="challenger"):
        TrainModelingConfig(**valid_modeling_dict)
