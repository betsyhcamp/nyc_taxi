import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from dashboard.train import header
from fcstnyctaxi.core.train import evaluate_impl
from fcstnyctaxi.core.train.evaluate_impl import _SCORE_KEYS, _derive

# horizon_2 scores fewer folds on purpose: comparing every cell against one count
# would report each of its cells as short when the asymmetry is structural.
FOLDS = {
    "horizon_1": ("2025-03-23", "2025-03-30", "2025-04-06"),
    "horizon_2": ("2025-03-23", "2025-03-30"),
}
MODELS = ("model_a", "model_b")
TIERS = ("global", "low")
METRICS = ("metric_x", "metric_y")
CONFIGURED_TIERS = ("low", "middle", "high")

# A later generation of the same run: one more horizon_1 origin, all else equal.
LATER_FOLDS = {**FOLDS, "horizon_1": (*FOLDS["horizon_1"], "2025-04-13")}

DROPPED_CELL = {
    "model": "model_a",
    "horizon": "horizon_1",
    "tier": "global",
    "metric": "metric_x",
}


def _fold_metrics(
    *,
    drop_one: bool,
    folds: dict[str, tuple[str, ...]] = FOLDS,
    metrics: tuple[str, ...] = METRICS,
) -> pd.DataFrame:
    """Fold grain, optionally with one fold's value NaN as a drop produces. The
    folds and metrics vary to build another generation of the same run."""
    rows = [
        {
            "model": model,
            "horizon": horizon,
            "tier": tier,
            "forecast_origin_date": pd.Timestamp(origin),
            "predicted_fiscal_year_month": 202504,
            "metric": metric,
            "value": 1.0,
            "n_obs": 4,
        }
        for model in MODELS
        for horizon, origins in folds.items()
        for origin in origins
        for tier in TIERS
        for metric in metrics
    ]
    frame = pd.DataFrame(rows)
    # The horizons must score different counts, or the per-horizon expectation is
    # never exercised: with them equal, comparing every cell against one count
    # passes every test in this file.
    scored = {len(origins) for origins in folds.values()}
    assert len(scored) == len(folds), "the horizons must differ in fold count"
    if not drop_one:
        return frame

    dropped = frame["forecast_origin_date"] == pd.Timestamp(FOLDS["horizon_1"][-1])
    for column, value in DROPPED_CELL.items():
        dropped &= frame[column] == value
    assert dropped.sum() == 1, "the plant must name exactly one fold"
    frame.loc[dropped, "value"] = np.nan
    return frame


@pytest.fixture
def full() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A run with every cell at full coverage, as the real runs read."""
    folds = _fold_metrics(drop_one=False)
    summary = _derive(folds, _SCORE_KEYS)
    # Self-check: each horizon's cells read its own fold count, so a shortfall
    # below is the planted drop and not the structural asymmetry.
    for horizon, origins in FOLDS.items():
        used = summary.loc[summary["horizon"] == horizon, "n_folds_used"]
        assert set(used) == {len(origins)}
    return summary, folds


@pytest.fixture
def one_dropped() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The same run with one fold's value NaN, which `_derive` does not count."""
    folds = _fold_metrics(drop_one=True)
    summary = _derive(folds, _SCORE_KEYS)
    # Self-check: the plant really moved exactly one cell, through the producer.
    expected = summary["horizon"].map({h: len(o) for h, o in FOLDS.items()})
    assert (summary["n_folds_used"] < expected).sum() == 1
    return summary, folds


# ================================================
# the grains, against the producer's own
# ================================================


@pytest.mark.parametrize(
    ("dashboard_keys", "producer_keys"),
    [
        (header._FOLD_KEYS, evaluate_impl._FOLD_KEYS),
        (header._CELL_KEYS, evaluate_impl._SCORE_KEYS),
    ],
    ids=["fold_grain", "cell_grain"],
)
def test_each_grain_matches_the_producer_s(
    dashboard_keys: list[str], producer_keys: list[str]
) -> None:
    """Narrowing the fold key leaves the per-horizon count right by accident,
    since within one horizon an origin determines the predicted month. The grain
    still has to be the grain the tables are keyed on."""
    assert dashboard_keys == producer_keys


# ================================================
# row 1, folds per horizon
# ================================================


def test_folds_per_horizon_counts_distinct_fold_keys(
    full: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """One fold is many rows, so a row count would overstate every horizon."""
    _, folds = full
    assert header.folds_per_horizon(folds) == {
        horizon: len(origins) for horizon, origins in FOLDS.items()
    }


def test_every_horizon_present_gets_its_own_count(
    full: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Pooling the two would hide the asymmetry this row exists to label."""
    _, folds = full
    assert set(header.folds_per_horizon(folds)) == set(FOLDS)


# ================================================
# row 2, fold coverage per cell
# ================================================


def test_a_dropped_fold_is_reported_and_its_cell_named(
    one_dropped: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """The only witness of the upward nanmean dropping a fold."""
    summary, folds = one_dropped
    coverage = header.fold_coverage(summary, folds)

    assert len(coverage.mismatched) == 1
    named = coverage.mismatched.iloc[0]
    assert {key: named[key] for key in DROPPED_CELL} == DROPPED_CELL
    assert named["n_folds_used"] < named["folds_scored"]


def test_the_structural_horizon_asymmetry_is_not_a_shortfall(
    full: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Every horizon_2 cell reads fewer folds and none of them is short."""
    summary, folds = full
    coverage = header.fold_coverage(summary, folds)
    assert coverage.mismatched.empty
    assert coverage.missing.empty


def test_the_rollup_counts_every_cell_in_the_file(
    full: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Narrowing to the displayed metrics would hide an integrity failure in the
    others, which is why this is a data check and not a display check."""
    summary, folds = full
    coverage = header.fold_coverage(summary, folds)
    assert coverage.cells == len(summary)


def test_a_horizon_with_no_folds_raises_rather_than_passing(
    full: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """A horizon only the summary scores means the tables are not one run's."""
    summary, folds = full
    with pytest.raises(ValueError, match="horizon_2"):
        header.fold_coverage(summary, folds[folds["horizon"] == "horizon_1"])


# ================================================
# row 2, the two tables against each other
# ================================================


def test_a_cell_using_more_folds_than_its_horizon_scored_is_reported() -> None:
    """A summary from a generation with one more origin: an excess is as much a
    disagreement as a shortfall."""
    folds = _fold_metrics(drop_one=False)
    summary = _derive(_fold_metrics(drop_one=False, folds=LATER_FOLDS), _SCORE_KEYS)
    later = summary[summary["horizon"] == "horizon_1"]
    # Self-check: the later generation really scored more horizon_1 folds.
    assert set(later["n_folds_used"]) == {len(LATER_FOLDS["horizon_1"])}

    coverage = header.fold_coverage(summary, folds)
    assert len(coverage.mismatched) == len(later)
    assert set(coverage.mismatched["horizon"]) == {"horizon_1"}
    excess = coverage.mismatched["n_folds_used"] > coverage.mismatched["folds_scored"]
    assert excess.all()


def test_a_cell_the_summary_lacks_is_reported_and_named() -> None:
    """Counted off the summary instead, a dropped metric shrinks the denominator
    and reads as full coverage."""
    folds = _fold_metrics(drop_one=False)
    summary = _derive(_fold_metrics(drop_one=False, metrics=METRICS[:1]), _SCORE_KEYS)
    # Self-check: the summary really lacks a metric the fold table holds.
    assert set(summary["metric"]) < set(folds["metric"])

    coverage = header.fold_coverage(summary, folds)
    assert set(coverage.missing["metric"]) == set(METRICS[1:])
    assert coverage.cells == len(_derive(folds, _SCORE_KEYS))
    assert coverage.mismatched.empty


def test_a_cell_the_fold_table_lacks_is_reported_as_scoring_none() -> None:
    """The reverse disagreement: a summary cell with no fold behind it."""
    folds = _fold_metrics(drop_one=False, metrics=METRICS[:1])
    summary = _derive(_fold_metrics(drop_one=False), _SCORE_KEYS)
    # Self-check: the fold table really lacks a metric the summary holds.
    assert set(folds["metric"]) < set(summary["metric"])

    coverage = header.fold_coverage(summary, folds)
    assert set(coverage.mismatched["metric"]) == set(METRICS[1:])
    assert (coverage.mismatched["folds_scored"] == 0).all()
    assert coverage.missing.empty


# ================================================
# row 3, tier label completeness
# ================================================


def test_a_configured_tier_no_row_uses_is_reported(
    full: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """The collapse `present_tier_labels` permits, since a prefix passes it."""
    summary, _ = full
    assert header.absent_tier_labels(summary, CONFIGURED_TIERS) == ("middle", "high")


def test_a_run_using_every_configured_tier_reports_none(
    full: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """The expected state, so the row has to be quiet in it."""
    summary, _ = full
    assert header.absent_tier_labels(summary, ("global", "low")) == ()


def test_presence_is_read_off_values_not_categories(
    full: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """A categorical can carry a tier no row uses, which `.cat.categories` would
    report as present and this row exists to catch."""
    summary, _ = full
    summary["tier"] = pd.Categorical(
        summary["tier"], categories=[*TIERS, "unused_tier"]
    )
    assert "unused_tier" in summary["tier"].cat.categories
    assert header.absent_tier_labels(summary, ("unused_tier",)) == ("unused_tier",)


# ================================================
# the renderer
# ================================================


# Unannotated: `AppTest.from_function` runs this function's source as a script of
# its own, where the annotations are evaluated and `pd` is undefined.
def _coverage_page(summary, fold_metrics, tier_labels):
    """The coverage table alone, as a script `AppTest` can run."""
    from dashboard.train import header

    header.render_coverage_table(summary, fold_metrics, tier_labels)


def _fold_coverage_line(summary: pd.DataFrame, fold_metrics: pd.DataFrame) -> str:
    """What the fold coverage row prints for this pair of tables."""
    app = AppTest.from_function(
        _coverage_page, args=(summary, fold_metrics, CONFIGURED_TIERS)
    ).run()
    assert not app.exception, app.exception[0].message if app.exception else ""
    return next(line.value for line in app.markdown if "fold coverage" in line.value)


def test_the_rollup_reads_full_coverage_only_when_the_tables_agree(
    full: tuple[pd.DataFrame, pd.DataFrame],
    one_dropped: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """A disagreement printed as full coverage is what this row exists to prevent,
    whether a cell is off its count or absent."""
    _, folds = full
    lacking = _derive(_fold_metrics(drop_one=False, metrics=METRICS[:1]), _SCORE_KEYS)

    assert "full coverage" in _fold_coverage_line(*full)
    assert "full coverage" not in _fold_coverage_line(*one_dropped)
    assert "full coverage" not in _fold_coverage_line(lacking, folds)
