"mirrors tsbricks, lives here until validated."

from typing import Any

import pandas as pd
from tsbricks.runner._utils import dynamic_import


def invoke_predict(
    fitted_model: Any,
    predict_callable: str,
    horizon: int,
    future_x_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run a model's predict-only callable against an already-fitted model.

    Mirrors tsbricks.runner.model_invocation.invoke_model()'s resolve-and-
    call shape, for the predict-only half tsbricks doesn't yet provide.
    Lives here until a second real caller validates promoting it.

    Args:
        fitted_model: An already-fitted model object (e.g. an MLForecast
            instance) which is never inspected or assumed to have a particular
            shape by this function.
        predict_callable: Dotted path string naming the model-specific
            predict function to resolve and call, e.g.
            "models.lightgbm_weekly.lightgbm_weekly_predict". Must accept
            (fitted_model, horizon, future_x_df=None) and return a forecast
            DataFrame.
        horizon: Number of forecast steps.
        future_x_df: Optional future exogenous DataFrame, forwarded to the
            predict callable only when provided.

    Returns:
        The predict callable's forecast DataFrame, unchanged.
    """
    # Carefule: Reaching into tsbricks internals to get dynamic_import(); only temporary
    predict_fn = dynamic_import(predict_callable)

    kwargs: dict[str, Any] = {}

    if future_x_df is not None:
        kwargs["future_x_df"] = future_x_df

    return predict_fn(fitted_model, horizon, **kwargs)
