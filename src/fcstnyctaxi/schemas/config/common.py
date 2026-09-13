# TODO: Once conventions are understood by the team, trim down comments
"""Concepts shared by more than one slice's configuration schemas.

Only genuinely shared *concepts*. ``EnvironmentConfig``'s three ``project_id`` and
``location`` pairs are deliberately **not** factored into a base: a compute region, a
registry region, and a dataset location share a spelling, not a meaning, and a base
class would imply they are substitutable.

tsbricks classes stay in tsbricks. The rules this config tree follows are in
``config/README.md``.
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

``lib/storage_layout._build_run_prefix`` is the first consumer: it types the slice
segment and checks membership with ``get_args``, since no type checker runs in CI.

Elsewhere the type is deliberately not a key. ``SliceImages`` declares three fields
rather than keying a dict by it, so a missing slice is a validation error at
composition rather than a KeyError at submit time; ``lib/config/bindings`` exposes a
named function per destination rather than a registry keyed by it.
"""
