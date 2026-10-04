from dataclasses import dataclass
from typing import Any

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
