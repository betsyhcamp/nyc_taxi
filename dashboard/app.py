"""The backtest monitor over one train run: `streamlit run dashboard/app.py`.

Assembly only. Every number comes from a cached loader and every panel from a
module that takes frames, so nothing here decides anything.
"""

import sys
from pathlib import Path

# Streamlit's bootstrap inserts only the main script's own folder on `sys.path`,
# so `dashboard.*` resolves here only once the project root is on it too. The
# tests reach the same modules by the same name through `pythonpath = ["."]`.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# E402 below is the price of that shim: these cannot precede it and resolve.
import streamlit as st  # noqa: E402

from dashboard.shared import load, runs  # noqa: E402
from dashboard.train import header, scorecard  # noqa: E402

_CONFIG_PATH = Path(__file__).resolve().parent / "config.toml"

# Two rows so skill and bias are readable at once. They open on those two and
# carry no heading of their own: the metric is selectable, so a fixed "skill"
# over a row showing bias would be a label that lies after one click.
_ROWS = (("row_1", "wrmae_pooled"), ("row_2", "signed_bias_pooled"))


def main() -> None:
    """Draw the page in funnel order: which run, is it admissible, how did it do."""
    st.set_page_config(page_title="Backtest monitor", layout="wide")
    st.title("Backtest monitor")

    settings = load.dashboard_config(str(_CONFIG_PATH))
    env, config_dir = settings["env"], settings["config_dir"]

    slice_root = load.slice_root_uri(config_dir, env)
    offered = runs.list_run_ids(slice_root)
    if not offered:
        st.error(f"No run under {slice_root} has written an evaluate manifest.")
        return

    pointed = runs.pointer_run_id(load.pointer_uri(config_dir, env))
    run_id = st.selectbox(
        f"train run, {env}",
        offered,
        index=offered.index(runs.default_run_id(offered, pointed)),
    )
    if pointed is not None and pointed not in offered:
        st.warning(
            f"The run pointer names {pointed}, which is not in the current "
            "listing. Showing the newest listed run instead."
        )

    prefix = load.evaluate_uri(config_dir, env, run_id)
    manifest = load.load_evaluate_manifest(prefix)
    summary_metrics = load.load_summary_metrics(prefix)
    tier_labels = manifest["config"]["tiering"]["tier_labels"]

    header.render_identity_strip(
        manifest, load.load_run_output(load.run_outputs_uri(config_dir, env, run_id))
    )
    st.divider()

    # Above the bar rows and gating them: if `n_series` moved, a skill number
    # below is uninterpretable rather than merely surprising.
    header.render_coverage_table(
        summary_metrics, load.load_fold_metrics(prefix), tier_labels
    )
    st.divider()

    for key, default_metric in _ROWS:
        scorecard.render_bar_row(
            summary_metrics,
            tier_labels,
            manifest["challenger_model"],
            manifest["benchmark_model"],
            key=key,
            default_metric=default_metric,
        )


main()
