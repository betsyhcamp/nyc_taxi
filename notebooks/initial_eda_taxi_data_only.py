# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
from pathlib import Path

import pandas as pd

import matplotlib.pyplot as plt

# %%
DATA_DIR_NAME = "data"
TAXI_FILENAME = "yellow_tripdata_2025-03.parquet"

# %%
try: 
    base_dir = Path(__file__).parent
except NameError:
    base_dir=Path.cwd()

taxi_file_path = base_dir.parent / DATA_DIR_NAME / TAXI_FILENAME
taxi_file_path

# %%
taxi_df = pd.read_parquet(taxi_file_path)
