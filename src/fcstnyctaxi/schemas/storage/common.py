"""Fixed layout names read across slices."""

BUNDLE_MODEL_DIR = "model"
"""The bundle's subdirectory the save callable owns, apart from the impl's files."""

RUN_IDENTITY_FILENAME = "run_identity.json"

RUN_OUTPUT_FILENAME = "run_output.json"
"""Singular, unlike the `run_outputs` module and schemas: storage matches this."""

LATEST_POINTER_FILENAME = "_latest.json"
"""The environment-root pointer every slice rewrites its own key in."""
