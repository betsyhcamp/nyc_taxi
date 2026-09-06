"""Concepts shared by more than one slice's configuration schemas.

Only genuinely shared *concepts* belong here, never coincidentally similar
shapes (configs_schemas_images §4.5). tsbricks classes stay in tsbricks; its
hierarchy is not mirrored.
"""

from typing import Literal

SliceName = Literal["feature", "train", "inference"]
"""The three pipelines, as a closed vocabulary.

A slice name is a segment of the storage convention
``<env>/<pipeline_name>/<run_id>/<step>/`` and the key of the per-slice
bindings registry, so the same three tokens appear in paths, in
``EnvironmentConfig.artifact_registry.images``, and in ``lib/config/bindings``.
"""
