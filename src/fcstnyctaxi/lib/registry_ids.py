MODEL_ID_MAX_LENGTH = 63
"""Documented by `Model.upload`, enforced only by the service."""

DISPLAY_NAME_MAX_LENGTH = 128
"""The SDK's `validate_display_name` cap. Restated, not imported: `aiplatform` takes
~3 s to import, and `compose_configs` calls this at step 1."""


def compose_model_id(model_id_prefix: str, model_name: str) -> str:
    """The Model resource's permanent id, which every version attaches to.

    A function, so an early check cannot drift from the id the API gets. The
    prefix's and `ModelRoles`' patterns already make the charset legal.

    Args:
        model_id_prefix (str): `ModelRegistry.model_id_prefix`.
        model_name (str): The model being registered.

    Returns:
        str: `<model_id_prefix>-<model_name>`, the name verbatim.

    Raises:
        ValueError: The id exceeds 63 characters.
    """
    model_id = f"{model_id_prefix}-{model_name}"
    if len(model_id) > MODEL_ID_MAX_LENGTH:
        raise ValueError(
            f"model_id {model_id!r} exceeds {MODEL_ID_MAX_LENGTH} characters."
        )
    return model_id


def compose_display_name(display_name_prefix: str, model_name: str) -> str:
    """The Model resource's console label.

    Its own cap, not `model_id`'s: the two prefixes are decoupled, so neither
    composition bounds the other.

    Args:
        display_name_prefix (str): `ModelRegistry.display_name_prefix`.
        model_name (str): The model being registered.

    Returns:
        str: `<display_name_prefix>-<model_name>`.

    Raises:
        ValueError: The composed name is longer than 128 characters.
    """
    display_name = f"{display_name_prefix}-{model_name}"
    if len(display_name) > DISPLAY_NAME_MAX_LENGTH:
        raise ValueError(
            f"display_name {display_name!r} exceeds {DISPLAY_NAME_MAX_LENGTH} "
            "characters."
        )
    return display_name
