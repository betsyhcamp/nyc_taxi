# TODO: Once conventions are understood by the team, trim down comments
"""Concepts shared by more than one slice's configuration schemas.

Only genuinely shared *concepts* belong here. Three blocks of ``EnvironmentConfig``
each declare a ``project_id`` and a ``location``, and that is deliberately **not**
factored into a shared base: a compute region, a registry region, and a dataset
location are three different facts that share a spelling, and inheritance would imply
a substitutability they do not have.

tsbricks classes stay in tsbricks; its hierarchy is not mirrored. The rules this
config tree follows are in ``config/README.md``.
"""

from typing import Literal

SliceName = Literal["feature", "train", "inference"]
"""The three pipelines, as a closed vocabulary — and the project's **only** one.

The same three tokens name every place a slice appears:

    config/<slice>/                        configuration fragments
    core/<slice>/, components/<slice>/     code layout
    schemas/config/<slice>.py              destination schemas
    <env>/<slice>/<run_id>/<step>/         the storage convention
    <slice>_run_id                         the run-id prefix
    artifact_registry.images.<slice>       image references

No consumer yet. ``SliceImages`` declares its three fields explicitly rather than
keying a dict by this type, so that a missing slice is a validation error at
composition rather than a KeyError at submit time; and ``lib/config/bindings``
exposes one function per slice rather than a registry. The expected first
consumer is per-slice env-var construction — ``FCST_{SLICE}_SERVICE_ACCOUNT``.
"""
