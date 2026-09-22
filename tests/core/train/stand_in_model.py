"""A package-free model on the pipeline callable contract, for tests of model-agnostic
code: final_fit and register_model need a bundle, not a particular booster. Its fit
records everything it is handed, so a test can compare that with an in-process call.

Resolved by the dotted paths below, which pytest's default import mode makes
importable, as it does ``test_final_fit_impl._save_nothing``.
"""

import pickle
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from fcstnyctaxi.lib.config.composition import save_config
from fcstnyctaxi.lib.utils import get_project_root_dir

STAND_IN_MODEL_NAME = "naive"
STAND_IN_EXOG_FEATURES = (
    "fiscal_week_of_month",
    "fiscal_month",
    "weeks_in_month",
    "count_workdays",
)
# Not stand_in_fit's default, so a fit handed freq alone records something else.
STAND_IN_SETTING = 7
STAND_IN_FILENAME = "stand_in.pkl"


def stand_in_fit(
    train_df: pd.DataFrame,
    freq: str,
    exog_df: pd.DataFrame | None = None,
    setting: int = 0,
) -> dict[str, Any]:
    """Everything the model was handed. ``setting`` is a hyperparameter other than
    ``freq``, defaulted as a real model's are, so dropping it is silent."""
    return {"train_df": train_df, "exog_df": exog_df, "freq": freq, "setting": setting}


def stand_in_save(model_obj: dict[str, Any], model_dir: Path) -> None:
    """Pickle the fitted model into the directory final_fit created."""
    (model_dir / STAND_IN_FILENAME).write_bytes(pickle.dumps(model_obj))


def stand_in_load(model_dir: Path) -> dict[str, Any]:
    """What stand_in_save wrote."""
    return pickle.loads((model_dir / STAND_IN_FILENAME).read_bytes())


def stand_in_config_dir(root: Path) -> Path:
    """The committed tree under root, naive the challenger, fitted by this module and
    configured with ``STAND_IN_SETTING``.

    Built once per root, so a test's own edits to the copy survive later staging.
    """
    config_dir = root / "stand_in_config"
    if config_dir.exists():
        return config_dir
    shutil.copytree(get_project_root_dir() / "config", config_dir)
    modeling_path = config_dir / "train" / "modeling.yaml"
    modeling = yaml.safe_load(modeling_path.read_text())
    modeling["model_roles"]["challenger"] = STAND_IN_MODEL_NAME
    modeling["model_settings"][STAND_IN_MODEL_NAME] = {
        "exog_features": list(STAND_IN_EXOG_FEATURES),
        "fit_callable": "stand_in_model.stand_in_fit",
        "save_callable": "stand_in_model.stand_in_save",
    }
    save_config(modeling, modeling_path)
    model_path = config_dir / "train" / "models" / f"{STAND_IN_MODEL_NAME}.yaml"
    model = yaml.safe_load(model_path.read_text())
    model["model"]["hyperparameters"]["setting"] = STAND_IN_SETTING
    save_config(model, model_path)
    return config_dir
