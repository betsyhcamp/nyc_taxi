"""Train image probe, run by cloudbuild.yaml inside the image it just built.

Reads the expected config files on stdin, one path per line, relative to config/.
"""

import sys

import pyarrow  # noqa: F401

# Every impl a component body imports. Every other tier runs in the dev venv, so a
# dependency the train group lacks would otherwise surface only on Vertex.
from fcstnyctaxi.core.train import (  # noqa: F401
    backtest_impl,
    evaluate_impl,
    final_fit_impl,
    register_model_impl,
)
from fcstnyctaxi.core.train.compose_configs_impl import compose_train_static_configs
from fcstnyctaxi.lib.config.bindings import available_environments

# Named only by config and imported at run time, so no impl import reaches them.
from fcstnyctaxi.models import lightgbm_weekly, naive, xgboost_weekly  # noqa: F401
from fcstnyctaxi.runtime_paths import CONFIG_DIR  # noqa: TID251


def main() -> None:
    """Compose every baked environment, then match the baked config tree to stdin."""
    envs = available_environments(CONFIG_DIR)
    if not envs:
        raise SystemExit(
            f"No environments under {CONFIG_DIR}/environments; "
            "is COPY config in the runtime stage?"
        )
    for env in envs:
        compose_train_static_configs(CONFIG_DIR, env)
    print(f"image OK: {envs} compose")

    want = set(sys.stdin.read().split())
    have = {
        str(p.relative_to(CONFIG_DIR)) for p in CONFIG_DIR.rglob("*") if p.is_file()
    }
    if want != have:
        raise SystemExit(
            f"baked config tree != config/: missing {sorted(want - have)} "
            f"extra {sorted(have - want)}; check COPY config and .dockerignore"
        )
    print(f"config tree OK: {len(have)} files")


if __name__ == "__main__":
    main()
