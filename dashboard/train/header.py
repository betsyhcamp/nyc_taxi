from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st

REGISTERED = "registered"
NOT_REGISTERED = "evaluated, not registered"

# The labels, in reading order: which run, what was compared, over what window
# and how much data. `git_hash`, the three input URIs and the config hashes are
# deliberately absent, their value being run-over-run comparison.
_HEAD = "run"
_ROWS = (
    ("challenger", "benchmark", "feature run"),
    ("first origin", "last origin", "origins", "series"),
)

_FOLD_KEYS = ["forecast_origin_date", "predicted_fiscal_year_month"]
_CELL_KEYS = ["model", "horizon", "tier", "metric"]


@dataclass(frozen=True)
class Registration:
    """The badge's two parts. A tag needs registration, but not the reverse."""

    label: str
    model_tag: str | None


def identity_fields(manifest: dict[str, Any]) -> dict[str, str]:
    """The fields the strip prints, keyed by display label.

    Raises rather than printing a gap: the strip gates every number below it, so
    a run whose manifest is missing a field is a run whose skill number cannot be
    read. No current run is missing one, which makes this hardening.
    """
    lineage = manifest.get("lineage", {})
    origins = manifest.get("origins", {})
    found = {
        _HEAD: lineage.get("train_run_id"),
        "challenger": manifest.get("challenger_model"),
        "benchmark": manifest.get("benchmark_model"),
        "feature run": lineage.get("feature_run_id"),
        "first origin": origins.get("first_origin"),
        "last origin": origins.get("last_origin"),
        "origins": origins.get("n_origins"),
        "series": manifest.get("n_series"),
    }
    absent = [label for label, value in found.items() if value is None]
    if absent:
        raise ValueError(
            f"the evaluate manifest supplies no {absent}, so the identity strip "
            "cannot be drawn. It gates every number below it, and a partial strip "
            "would leave a skill number uninterpretable rather than unreadable."
        )
    return {label: str(value) for label, value in found.items()}


def registration(run_output: dict[str, Any] | None) -> Registration:
    """The badge, from the completion marker's presence.

    Absence is information, not an error: a run that evaluated and then failed at
    registration has good tables. Runs predating the marker's rename badge as
    unregistered incorrectly, which is accepted rather than branched on.
    """
    if run_output is None:
        return Registration(NOT_REGISTERED, None)
    published = run_output.get("published") or {}
    return Registration(REGISTERED, published.get("model_tag"))


def render_identity_strip(
    manifest: dict[str, Any], run_output: dict[str, Any] | None
) -> None:
    """Draw the strip. Reads nothing; both arguments come from cached loaders."""
    fields = identity_fields(manifest)
    status = registration(run_output)

    head, badge = st.columns([3, 2])
    _cell(head, _HEAD, fields[_HEAD])
    _cell(badge, "status", status.label)
    if status.model_tag is not None:
        badge.caption(status.model_tag)

    for labels in _ROWS:
        for column, label in zip(st.columns(len(labels)), labels, strict=True):
            _cell(column, label, fields[label])


def _cell(column: Any, label: str, value: str) -> None:
    """One label over one value. Not `st.metric`, which sets a 22-character run
    id in headline type and overflows its column."""
    column.caption(label)
    column.markdown(f"**{value}**")


@dataclass(frozen=True)
class FoldCoverage:
    """The rollup's denominator, and the cells behind it when the tables disagree."""

    cells: int
    mismatched: pd.DataFrame
    missing: pd.DataFrame


def folds_per_horizon(fold_metrics: pd.DataFrame) -> dict[str, int]:
    """Distinct folds each horizon scored. Informational, not a check.

    horizon_2 scoring fewer than horizon_1 is structural: the final month's
    origins have horizons shrinking toward one week and produce horizon_1 only.
    """
    distinct = fold_metrics[_FOLD_KEYS + ["horizon"]].drop_duplicates()
    counts = distinct.groupby("horizon", observed=True).size()
    return {str(horizon): int(count) for horizon, count in counts.items()}


def fold_coverage(
    summary_metrics: pd.DataFrame, fold_metrics: pd.DataFrame
) -> FoldCoverage:
    """Each summary cell's `n_folds_used` against the folds its horizon scored,
    over every cell either table holds.

    Over every metric, not the three displayed: the only witness of the upward
    nanmean dropping a fold. Per horizon, since horizon_2 scoring fewer is
    structural. A healthy run derives one table from the other, so an excess
    fold or a cell only one table holds is a disagreement too.
    """
    scored = folds_per_horizon(fold_metrics)
    unmapped = set(summary_metrics["horizon"].astype(str)) - set(scored)
    if unmapped:
        raise ValueError(
            f"summary_metrics scores horizon(s) {sorted(unmapped)} that "
            "fold_metrics holds no fold for, so the two tables come from "
            "different runs or different generations of one."
        )

    # As strings: `tier` is an ordered categorical, and two generations of a
    # table can carry different category sets.
    expected = fold_metrics[_CELL_KEYS].drop_duplicates().astype(str)
    expected = expected.assign(folds_scored=expected["horizon"].map(scored))
    used = summary_metrics[_CELL_KEYS + ["n_folds_used"]].astype(
        {key: str for key in _CELL_KEYS}
    )
    cells = expected.merge(used, on=_CELL_KEYS, how="outer", indicator=True)
    # A summary cell the fold table never scored expects no folds at all.
    cells["folds_scored"] = cells["folds_scored"].fillna(0).astype(int)

    in_summary = cells["_merge"] != "left_only"
    off_count = in_summary & (cells["n_folds_used"] != cells["folds_scored"])
    return FoldCoverage(
        cells=len(cells),
        mismatched=cells.loc[
            off_count, _CELL_KEYS + ["n_folds_used", "folds_scored"]
        ].astype({"n_folds_used": int}),
        missing=cells.loc[~in_summary, _CELL_KEYS],
    )


def absent_tier_labels(
    summary_metrics: pd.DataFrame, tier_labels: Sequence[str]
) -> tuple[str, ...]:
    """Configured tiers that no row uses.

    Not redundant with the impl's own guard: `present_tier_labels` raises only
    when the observed labels are not a prefix of the configured list, and a
    collapse from five tiers to three is a prefix. Read off values, because a
    categorical's `categories` can name a tier no row carries.
    """
    present = set(summary_metrics["tier"].astype(str))
    return tuple(label for label in tier_labels if label not in present)


def render_coverage_table(
    summary_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    tier_labels: Sequence[str],
) -> None:
    """The four rows, as a rollup rather than one row per cell."""
    scored = folds_per_horizon(fold_metrics)
    coverage = fold_coverage(summary_metrics, fold_metrics)
    absent = absent_tier_labels(summary_metrics, tier_labels)

    counts = ", ".join(
        f"{horizon} {count}" for horizon, count in sorted(scored.items())
    )
    st.markdown(f"**folds scored**  {counts}")
    st.caption(
        "horizon_2 scoring fewer is structural, not loss: the final month's "
        "origins have horizons shrinking toward one week, so they produce "
        "horizon_1 only."
    )

    if coverage.mismatched.empty and coverage.missing.empty:
        st.markdown(f"**fold coverage**  all {coverage.cells} cells at full coverage")
    else:
        st.markdown(
            f"**fold coverage**  of {coverage.cells} cells, "
            f"{len(coverage.mismatched)} off their fold count and "
            f"{len(coverage.missing)} absent from summary_metrics"
        )
        with st.expander("the cells the two tables disagree on"):
            for caption, disagreeing in (
                ("off their fold count", coverage.mismatched),
                ("in fold_metrics, absent from summary_metrics", coverage.missing),
            ):
                if not disagreeing.empty:
                    st.caption(caption)
                    st.dataframe(disagreeing, hide_index=True)

    if absent:
        st.markdown(f"**tier labels**  configured but unused: {', '.join(absent)}")
    else:
        st.markdown(f"**tier labels**  all {len(tier_labels)} configured tiers present")

    st.caption(
        "`origins` reading 30 or 31 is the fiscal calendar's 4/4/5 pattern, not "
        "a defect: where the five-week month falls decides which."
    )
