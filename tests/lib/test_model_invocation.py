from __future__ import annotations

import pandas as pd
import pytest

import fcstnyctaxi.lib.model_invocation as model_invocation
from fcstnyctaxi.lib.model_invocation import invoke_predict


def _fake_predict_callable(some_model, horizon, **kwargs) -> pd.DataFrame:
    """Deliberately named/shaped unlike lightgbm_weekly_predict()'s
    (mlfcst, horizon, future_x_df=None): the first parameter isn't called
    mlfcst, and future_x_df arrives via **kwargs rather than a named
    parameter. This lets the tests below distinguish "omitted" from
    "passed as None", and proves invoke_predict() never hardcodes a
    keyword name for the fitted model.
    """
    return pd.DataFrame(
        {
            "received_model": [some_model],
            "received_horizon": [horizon],
            "future_x_df_in_kwargs": ["future_x_df" in kwargs],
            "received_future_x_df": [kwargs.get("future_x_df")],
        }
    )


@pytest.fixture
def patched_dynamic_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replaces dynamic_import with one that always resolves to
    _fake_predict_callable, regardless of the dotted-path string given.
    Isolates invoke_predict()'s own call-shaping logic from
    dynamic_import()'s real import machinery — tsbricks' own tested code,
    not ours to re-test here.
    """
    monkeypatch.setattr(
        model_invocation, "dynamic_import", lambda path: _fake_predict_callable
    )


def test_invoke_predict_resolves_and_calls_positionally(
    patched_dynamic_import: None,
) -> None:
    """fitted_model/horizon are passed positionally, not via a hardcoded
    keyword — invoke_predict() must work even though _fake_predict_callable's
    first parameter isn't named mlfcst."""
    result = invoke_predict(
        fitted_model="some_fitted_model_object",
        predict_callable="irrelevant.dotted.path",  # dynamic_import is mocked
        horizon=3,
    )
    assert result["received_model"].item() == "some_fitted_model_object"
    assert result["received_horizon"].item() == 3


def test_invoke_predict_future_x_df_only_forwarded_when_given(
    patched_dynamic_import: None,
) -> None:
    """future_x_df is only forwarded when given — omitted from kwargs
    entirely when None (not passed explicitly as None), and included,
    unchanged, when a real DataFrame is provided."""
    result = invoke_predict(
        fitted_model="some_fitted_model_object",
        predict_callable="irrelevant.dotted.path",  # dynamic_import is mocked
        horizon=3,
    )
    assert result["future_x_df_in_kwargs"].item() is False

    dummy_df = pd.DataFrame({"col_1": [1, 2], "col_b": ["a", "b"]})

    result = invoke_predict(
        fitted_model="some_fitted_model_object",
        predict_callable="irrelevant.dotted.path",  # dynamic_import is mocked
        horizon=3,
        future_x_df=dummy_df,
    )
    assert result["future_x_df_in_kwargs"].item() is True
    assert result["received_future_x_df"].item().equals(dummy_df)
