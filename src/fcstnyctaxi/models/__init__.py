"""The Training pipeline's model callables, named in config by dotted path.

Notebook callables in ``experimental_models/`` differ: their ``future_x_df`` is the
raw ``ds``-keyed calendar.

``fit_predict_callable(train_df, horizon, **kwargs)`` runs a backtest fold, receiving
``hyperparameters`` over ``predict_params`` plus ``future_x_df``. It returns
``forecast``, ``(forecast, fitted)`` or ``(forecast, fitted, model)``, the forecast
holding ``unique_id``, ``ds`` and ``ypred``. If ``fit_callable`` is declared, it fits
through it, so scored and registered models share one implementation.

``fit_callable(train_df, exog_df=..., **hyperparameters)`` returns the fitted model
and declares no ``**kwargs``, so a mistyped hyperparameter raises.
``save_callable(model_obj, model_dir)`` writes it into the existing, empty
``model_dir`` and returns ``None``.

``future_x_df`` and ``exog_df`` are one frame keyed ``unique_id`` / ``ds`` over every
calendar date, holding only the model's ``exog_features``. Merge it as is.
"""
