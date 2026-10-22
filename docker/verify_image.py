"""Shared image probe, run by cloudbuild.yaml inside every image it builds.

Arrives on stdin, so argv[1] is the git hash the build passed as GIT_HASH.
"""

import os
import sys

# Resolves only through the image's venv, so this also proves the venv is on PATH.
import fcstnyctaxi  # noqa: F401

expected = sys.argv[1]
baked = os.environ.get("FCST_GIT_HASH")
if baked != expected:
    raise SystemExit(
        f"FCST_GIT_HASH is {baked!r} but the build passed {expected!r}; "
        "check ARG GIT_HASH and ENV FCST_GIT_HASH in the runtime stage"
    )
print(f"image OK: FCST_GIT_HASH={baked}")
