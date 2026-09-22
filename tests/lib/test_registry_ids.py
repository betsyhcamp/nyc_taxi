import re

import pytest
from google.cloud.aiplatform.utils import validate_display_name

from fcstnyctaxi.lib.registry_ids import compose_display_name, compose_model_id
from fcstnyctaxi.schemas.config.train import ModelRegistry, ModelRoles

# Transcribed from `Model.upload`'s docstring, not from the code under test.
MODEL_ID_RULE = re.compile(r"[a-z_][a-z0-9_-]{0,62}")


def _registry(
    *,
    display_name_prefix: str = "fcst-monthly-revenue",
    model_id_prefix: str = "fcst-monthly-revenue",
) -> ModelRegistry:
    """Through the schema, so every prefix here is one a config could hold."""
    return ModelRegistry(
        display_name_prefix=display_name_prefix, model_id_prefix=model_id_prefix
    )


def _role_name(name: str) -> str:
    """Through ModelRoles, so every name here is one a role could hold."""
    return ModelRoles(benchmark=name, challenger=name).challenger


def test_the_model_name_rides_verbatim_in_the_id() -> None:
    """Mapped to hyphens, the id would be a second spelling of the model's name."""
    model_id = compose_model_id(_registry().model_id_prefix, _role_name("naive_weekly"))

    assert model_id == "fcst-monthly-revenue-naive_weekly"


@pytest.mark.parametrize("model_name", ["9model", "_model", "0"])
def test_a_name_the_roles_accept_composes_a_legal_id(model_name: str) -> None:
    """The prefix leads, so a name's leading digit is unrepresentable, not guarded."""
    model_id = compose_model_id(
        _registry(model_id_prefix="a").model_id_prefix, _role_name(model_name)
    )

    assert MODEL_ID_RULE.fullmatch(model_id)


def test_compose_model_id_accepts_63_characters_and_rejects_64() -> None:
    """Both sides of the documented cap, which no SDK check enforces."""
    prefix = _registry(model_id_prefix="p" * 57).model_id_prefix

    assert len(compose_model_id(prefix, _role_name("abcde"))) == 63
    with pytest.raises(ValueError, match="model_id"):
        compose_model_id(prefix, _role_name("abcdef"))


def test_compose_display_name_caps_where_the_sdk_does() -> None:
    """Agreement with validate_display_name, so step 1 refuses what upload would."""
    model_name = _role_name("lightgbm")
    at_cap = compose_display_name(
        _registry(display_name_prefix="d" * 119).display_name_prefix, model_name
    )
    over_prefix = _registry(display_name_prefix="d" * 120).display_name_prefix

    assert len(at_cap) == 128
    validate_display_name(at_cap)
    # Self-check: the SDK refuses the name the next assertion expects refused.
    with pytest.raises(ValueError):
        validate_display_name(f"{over_prefix}-{model_name}")
    with pytest.raises(ValueError, match="display_name"):
        compose_display_name(over_prefix, model_name)


def test_a_long_display_name_prefix_fails_while_its_model_id_stays_legal() -> None:
    """The prefixes are decoupled, so model_id's cap bounds no display name."""
    registry = _registry(display_name_prefix="d" * 123)
    model_name = _role_name("lightgbm")

    assert MODEL_ID_RULE.fullmatch(
        compose_model_id(registry.model_id_prefix, model_name)
    )
    with pytest.raises(ValueError, match="display_name"):
        compose_display_name(registry.display_name_prefix, model_name)
