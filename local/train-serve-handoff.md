Things to accomplish:

Assumption:
1. Forecast 4 weeks into the future. (28 days; and our grain is 1 day)

1. Backfill some feature transforms on historical data.
  - training set: `2025-01-01` to `2025-11-30` (inclusive)
  - holdout set: `2025-12-01` to `2025-12-28` (inclusive)
  - create features via SQL: `average_daily_total_ride_price_last_7_days`, and some others
  - put these in a table `prod.rides_forecast_features`
    - feature/experiment branches could but data in `<experiment>.rides_forecast_features`.
2. Split data into a training and holdout set by date.
3. Create a 4-file folder to train a model:
   - `train.py` trains the model--uses `infer.py` and `model_io.py`
   - `infer.py` contains a single function `infer(model: TModel, data: pd.DataFrame) -> pd.DataFrame` that runs inference using the model
   - `model_io.py` has a 
      - `save_model(model_id: str, models_dir: pathlib.Path, infer_py_fpath: pathlib.Path, feature_view_query: str, query_timestamp: datetime)` accepts various arguments since different frameworks require saving in different ways
        - writes the model to a `<models_dir>/<model_id>/` directory
        - writes the `infer.py` file to the same directory
        - writes the `model_io.py` file to the same directory
        - writes the `pyproject.toml` file to the same directory
        - writes the `feature_view_query` to a file called `feature_view_query.sql` in the same directory--the WHERE clause should be parameterized so that we can pass in different date ranges or select specific primary keys
        - a JSON object containing
          - the rendered query (including the specific where clause)
          - the where clause
          - the timestamp of when the query was run
          should all be saved so that we can time-travel back to this moment and re-run the same query to reproduce the dataset.
      - `load_model(model_dir: pathlib.Path) -> TModel` loads the model
   - `pyproject.toml` contains top-level (non-transitive) dependencies and entry points for the project (a `uv` app)
      - the most important dependency is the model framework. When we save the model, 
   The reason for doing this is: we want to support multiple different model approaches. Pytorch, sktime, etc. 
4. Create single small python project for evaluating the model
   - have a bash script that uses `uv` to install
     - any dependencies needed for eval
     - `uv add`s the model framework version (possibly pandas or polars as well)
     - we'll run an install script to produce an environment before running eval. In the future, we'd bake a docker image shared between train and eval using a similar script.
   - `evaluate.py` contains functions that
     - accept the model id
     - loads the model
     - moves `infer.py` and `model_io.py` to a location in the PYTHONPATH where it can be imported from
     - calls `from model_io import load_model` and loads the model given the model id
     - uses the saved query, but passes in a different date range to get the holdout set
     - calls `infer(model, data)` to get the predictions
5. do something similar to 4 but for infer. Load the model. Install requirements. Pass in a new dataset using a new data set in the sql query.
