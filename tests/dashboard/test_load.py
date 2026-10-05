import json
import re
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pandas as pd
import pytest
import streamlit as st

from dashboard.shared import load, paths
from fcstnyctaxi.core.train import evaluate_impl

REGISTERED = "20260929t045735959418z"
EVALUATED_ONLY = "20260929t045146753815z"
MODEL_TAG = "projects/1/locations/us-central1/models/model-a@11"


@pytest.fixture(autouse=True)
def clear_caches() -> Iterator[None]:
    """Cached loaders outlive a test, so a leaked entry would read as a hit."""
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def _table(run_id: str, rows: int) -> pd.DataFrame:
    """A table carrying the lineage column every evaluate output carries."""
    return pd.DataFrame(
        {
            "value": [float(index) for index in range(rows)],
            "train_run_id": [run_id] * rows,
        }
    )


def _write_run(root: Path, run_id: str, *, registered: bool) -> None:
    """One run's evaluate directory, and its completion marker when it has one."""
    evaluate = root / run_id / "evaluate"
    evaluate.mkdir(parents=True)
    (evaluate / paths.EVALUATE_MANIFEST).write_text(
        json.dumps({"lineage": {"train_run_id": run_id}, "n_series": len(run_id)})
    )
    _table(run_id, rows=2).to_parquet(evaluate / paths.SUMMARY_METRICS)
    _table(run_id, rows=3).to_parquet(evaluate / paths.FOLD_METRICS)
    if registered:
        _register(root, run_id)


def _register(root: Path, run_id: str) -> None:
    """The completion marker registration writes at the run root."""
    (root / run_id / "run_output.json").write_text(
        json.dumps({"train_run_id": run_id, "published": {"model_tag": MODEL_TAG}})
    )


@pytest.fixture
def two_runs(tmp_path: Path) -> dict[str, str]:
    """Two runs, one registered and one not, each stamped with its own lineage."""
    _write_run(tmp_path, REGISTERED, registered=True)
    _write_run(tmp_path, EVALUATED_ONLY, registered=False)

    prefixes = {
        run_id: f"{tmp_path / run_id / 'evaluate'}/"
        for run_id in (REGISTERED, EVALUATED_ONLY)
    }
    # Self-check: the two runs' tables really differ, so an equality below is the
    # loader's doing and not two fixtures that happen to match.
    frames = [load.load_summary_metrics(prefix) for prefix in prefixes.values()]
    assert not frames[0].equals(frames[1])
    st.cache_data.clear()
    return prefixes


# ================================================
# run identity, the loaders' one real hazard
# ================================================


def test_each_run_loads_its_own_tables(two_runs: dict[str, str]) -> None:
    """A mixed cache key is what the two bit-identical runs in the bucket hide."""
    loaded = {
        run_id: load.load_summary_metrics(prefix) for run_id, prefix in two_runs.items()
    }
    stamped = {run_id: set(frame["train_run_id"]) for run_id, frame in loaded.items()}
    assert stamped == {run_id: {run_id} for run_id in two_runs}


def test_each_run_loads_its_own_manifest(two_runs: dict[str, str]) -> None:
    """The identity strip reads this, so a crossed read misnames the whole page."""
    for run_id, prefix in two_runs.items():
        manifest = load.load_evaluate_manifest(prefix)
        assert manifest["lineage"]["train_run_id"] == run_id


def test_the_two_tables_are_cached_apart(two_runs: dict[str, str]) -> None:
    """One prefix serves both reads, so the filename has to separate them."""
    prefix = two_runs[REGISTERED]
    assert len(load.load_summary_metrics(prefix)) != len(load.load_fold_metrics(prefix))


# ================================================
# the lineage check
# ================================================


def _read(prefix: str) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """One run's manifest and tables, through the loaders the page reads them by."""
    return load.load_evaluate_manifest(prefix), {
        paths.SUMMARY_METRICS: load.load_summary_metrics(prefix),
        paths.FOLD_METRICS: load.load_fold_metrics(prefix),
    }


def _plant(two_runs: dict[str, str], filename: str) -> str:
    """Copy the other run's file over the registered run's, as a manual copy would."""
    target = two_runs[REGISTERED]
    shutil.copy(f"{two_runs[EVALUATED_ONLY]}{filename}", f"{target}{filename}")
    return target


def test_a_run_whose_files_are_its_own_passes(two_runs: dict[str, str]) -> None:
    """Every pipeline-written run is in this state, so the check must stay quiet."""
    load.check_lineage(REGISTERED, *_read(two_runs[REGISTERED]))


def test_another_runs_manifest_under_this_run_raises(two_runs: dict[str, str]) -> None:
    """The identity strip would describe one run above another run's numbers."""
    prefix = _plant(two_runs, paths.EVALUATE_MANIFEST)
    with pytest.raises(ValueError, match="manifest"):
        load.check_lineage(REGISTERED, *_read(prefix))


@pytest.mark.parametrize("filename", [paths.SUMMARY_METRICS, paths.FOLD_METRICS])
def test_another_runs_table_under_this_run_raises(
    two_runs: dict[str, str], filename: str
) -> None:
    """The loader accepts it and the page charts it under this run's name. Both
    tables, so a check reading only the first is not enough."""
    prefix = _plant(two_runs, filename)
    with pytest.raises(ValueError, match=re.escape(filename)):
        load.check_lineage(REGISTERED, *_read(prefix))


def test_a_table_mixing_two_runs_raises(two_runs: dict[str, str]) -> None:
    """Containing the selected run is not enough: the other run's rows are not its."""
    manifest, frames = _read(two_runs[REGISTERED])
    other = load.load_summary_metrics(two_runs[EVALUATED_ONLY])
    mixed = pd.concat([frames[paths.SUMMARY_METRICS], other])
    # Self-check: the selected run is among the ids, so only an exact-set check
    # catches this rather than any check that the run is present.
    assert REGISTERED in set(mixed["train_run_id"])
    frames[paths.SUMMARY_METRICS] = mixed
    with pytest.raises(ValueError, match=re.escape(paths.SUMMARY_METRICS)):
        load.check_lineage(REGISTERED, manifest, frames)


# ================================================
# the completion marker
# ================================================


def test_a_run_that_never_registered_reads_as_none(
    tmp_path: Path, two_runs: dict[str, str]
) -> None:
    """Absence is information the badge prints, not an error."""
    assert load.load_run_output(f"{tmp_path / EVALUATED_ONLY}/run_output.json") is None


def test_a_registered_run_reads_its_model_tag(
    tmp_path: Path, two_runs: dict[str, str]
) -> None:
    """What the badge shows beside "registered"."""
    marker = load.load_run_output(f"{tmp_path / REGISTERED}/run_output.json")
    assert marker is not None
    assert marker["published"]["model_tag"] == MODEL_TAG


def test_a_late_registration_is_seen_once_the_cache_expires(
    tmp_path: Path, cache_clock: Mock
) -> None:
    """Otherwise a run registering after its page first loaded badges as
    unregistered for the life of the server."""
    _write_run(tmp_path, EVALUATED_ONLY, registered=False)
    uri = f"{tmp_path / EVALUATED_ONLY}/run_output.json"
    assert load.load_run_output(uri) is None

    _register(tmp_path, EVALUATED_ONLY)
    # Self-check: short of expiry the read is still the cached absence, so the
    # refresh below is the expiry's doing.
    cache_clock.return_value = load.MUTABLE_TTL_SECONDS - 1
    assert load.load_run_output(uri) is None

    cache_clock.return_value = load.MUTABLE_TTL_SECONDS + 1
    assert load.load_run_output(uri) is not None


def test_an_unvalidated_marker_still_loads(tmp_path: Path) -> None:
    """The schema forbids extras, so validating would let a field the dashboard
    never reads blank the page."""
    (tmp_path / "run_output.json").write_text(
        json.dumps({"train_run_id": "x", "a_field_added_upstream": 1})
    )
    assert load.load_run_output(f"{tmp_path}/run_output.json") is not None


# ================================================
# cache semantics the panels depend on
# ================================================


def test_mutating_a_loaded_frame_does_not_poison_the_cache(
    two_runs: dict[str, str],
) -> None:
    """Panels filter and assign on what they load, and every panel loads again."""
    prefix = two_runs[REGISTERED]
    first = load.load_summary_metrics(prefix)
    first.loc[0, "value"] = 999.0

    assert load.load_summary_metrics(prefix).loc[0, "value"] == 0.0


# ================================================
# the dashboard's own config
# ================================================


def test_the_config_supplies_both_settings(tmp_path: Path) -> None:
    """The two values every prefix in the dashboard resolves from."""
    (tmp_path / "config.toml").write_text('env = "dev"\nconfig_dir = "config"\n')
    settings = load.dashboard_config(f"{tmp_path / 'config.toml'}")
    assert settings["env"] == "dev"


def test_a_relative_config_dir_is_resolved_against_the_project_root(
    tmp_path: Path,
) -> None:
    """Streamlit can be launched from anywhere, so a path left relative to the
    working directory composes a different environment or none at all."""
    (tmp_path / "config.toml").write_text('env = "dev"\nconfig_dir = "config"\n')
    resolved = Path(load.dashboard_config(f"{tmp_path / 'config.toml'}")["config_dir"])
    assert resolved.is_absolute()
    assert resolved == tmp_path.resolve().parent / "config"


def test_an_absolute_config_dir_is_left_alone(tmp_path: Path) -> None:
    """Resolving an absolute path against anything would corrupt it."""
    absolute = tmp_path / "elsewhere"
    (tmp_path / "config.toml").write_text(f'env = "dev"\nconfig_dir = "{absolute}"\n')
    settings = load.dashboard_config(f"{tmp_path / 'config.toml'}")
    assert settings["config_dir"] == str(absolute)


@pytest.mark.parametrize("omitted", ["env", "config_dir"])
def test_a_missing_setting_raises_rather_than_defaulting(
    tmp_path: Path, omitted: str
) -> None:
    """A default would quietly open an environment the file does not name."""
    kept = {"env": '"dev"', "config_dir": '"config"'}
    kept.pop(omitted)
    (tmp_path / "config.toml").write_text(
        "\n".join(f"{key} = {value}" for key, value in kept.items())
    )
    with pytest.raises(ValueError, match=omitted):
        load.dashboard_config(f"{tmp_path / 'config.toml'}")


def test_the_shipped_config_parses(tmp_path: Path) -> None:
    """The file the app actually reads, against a tree that must exist."""
    settings = load.dashboard_config("dashboard/config.toml")
    assert Path(
        settings["config_dir"], "environments", f"{settings['env']}.yaml"
    ).is_file()


# ================================================
# the filenames, against the producer's own names
# ================================================


@pytest.mark.parametrize(
    ("dashboard_name", "producer_field"),
    [
        (paths.SUMMARY_METRICS, "summary_metrics"),
        (paths.FOLD_METRICS, "fold_metrics"),
    ],
    ids=["summary_metrics", "fold_metrics"],
)
def test_each_table_filename_is_the_one_the_producer_writes(
    dashboard_name: str, producer_field: str
) -> None:
    """A fixture writing the file through this same constant cannot notice a
    rename, so the producer's own mapping is the other side."""
    assert dashboard_name == evaluate_impl._OUTPUT_FILENAMES[producer_field]
