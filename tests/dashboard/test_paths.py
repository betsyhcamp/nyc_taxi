from pathlib import Path

import pytest

from dashboard.shared import paths
from fcstnyctaxi.core.train import evaluate_impl
from fcstnyctaxi.lib.storage_layout import resolve_run_prefix
from fcstnyctaxi.lib.utils import get_project_root_dir
from fcstnyctaxi.pipelines import local_train_pipeline
from fcstnyctaxi.schemas.storage.common import RUN_OUTPUT_FILENAME

CONFIG_DIR = str(get_project_root_dir() / "config")
ENV = "dev"
RUN_ID = "RUNID"
MODEL = "MODELNAME"


@pytest.fixture
def run_prefix() -> str:
    """The run root the dashboard's prefixes must descend from."""
    return resolve_run_prefix(Path(CONFIG_DIR), ENV, "train", RUN_ID)


@pytest.fixture
def prefixes() -> dict[str, str]:
    """All three prefixes, for the properties every one of them must hold."""
    return {
        "train_slice_root": paths.train_slice_root(CONFIG_DIR, ENV),
        "evaluate_prefix": paths.evaluate_prefix(CONFIG_DIR, ENV, RUN_ID),
        "sidecar_prefix": paths.sidecar_prefix(CONFIG_DIR, ENV, RUN_ID, MODEL),
    }


# ================================================
# train_slice_root tests
# ================================================


def test_train_slice_root_is_the_immediate_parent_of_a_run_root(
    run_prefix: str,
) -> None:
    """Immediate, not merely above: listing the environment root instead would
    return the three slice names as if they were run ids."""
    assert run_prefix == f"{paths.train_slice_root(CONFIG_DIR, ENV)}{RUN_ID}/"


# ================================================
# evaluate_prefix tests
# ================================================


def test_the_manifest_filename_is_the_one_the_producer_writes() -> None:
    """The run gate reads this name. Misspelled, the dropdown is always empty, and
    a fixture writing it through this same constant cannot notice."""
    assert paths.EVALUATE_MANIFEST == evaluate_impl._MANIFEST_FILENAME


def test_evaluate_prefix_names_the_directory_the_producer_writes(
    run_prefix: str,
) -> None:
    """Against `evaluate_impl`'s own step name, which it raises to enforce."""
    expected = f"{run_prefix}{evaluate_impl._STEP_DIR_NAME}/"
    assert paths.evaluate_prefix(CONFIG_DIR, ENV, RUN_ID) == expected


# ================================================
# sidecar_prefix tests
# ================================================


def test_sidecar_prefix_matches_the_uri_the_backtest_step_publishes_to(
    run_prefix: str,
) -> None:
    """Against the producer's own builder, so a misplaced model segment fails."""
    expected = local_train_pipeline._backtest_uri(run_prefix, MODEL)
    assert paths.sidecar_prefix(CONFIG_DIR, ENV, RUN_ID, MODEL) == expected


# ================================================
# properties every prefix shares
# ================================================


def test_every_prefix_ends_in_a_slash(prefixes: dict[str, str]) -> None:
    """A filename concatenated onto a prefix without one addresses a sibling."""
    assert [name for name, uri in prefixes.items() if not uri.endswith("/")] == []


def test_every_prefix_is_a_gcs_uri(prefixes: dict[str, str]) -> None:
    """`fs.ls` strips the scheme off what it returns; what goes in keeps it."""
    assert [n for n, uri in prefixes.items() if not uri.startswith("gs://")] == []


def test_an_unknown_environment_raises_rather_than_building_a_path() -> None:
    """A typo'd env would otherwise read an empty prefix and show no runs."""
    with pytest.raises(ValueError, match="nonsuch"):
        paths.train_slice_root(CONFIG_DIR, "nonsuch")
    with pytest.raises(ValueError, match="nonsuch"):
        paths.evaluate_prefix(CONFIG_DIR, "nonsuch", RUN_ID)
    with pytest.raises(ValueError, match="nonsuch"):
        paths.sidecar_prefix(CONFIG_DIR, "nonsuch", RUN_ID, MODEL)


# ================================================
# the two URIs the badge and the selector read
# ================================================


def test_run_outputs_uri_names_the_marker_at_the_run_root(run_prefix: str) -> None:
    """Resolved under the wrong slice, every run badges "not registered"."""
    expected = f"{run_prefix}{RUN_OUTPUT_FILENAME}"
    assert paths.run_outputs_uri(CONFIG_DIR, ENV, RUN_ID) == expected


def test_the_pointer_sits_above_the_slice_partition() -> None:
    """One file carries a key per slice, so a slice-scoped pointer strands two."""
    pointer = paths.pointer_uri(CONFIG_DIR, ENV)
    environment_root = f"{pointer.rsplit('/', 1)[0]}/"
    assert paths.train_slice_root(CONFIG_DIR, ENV).startswith(environment_root)
    assert environment_root != paths.train_slice_root(CONFIG_DIR, ENV)
