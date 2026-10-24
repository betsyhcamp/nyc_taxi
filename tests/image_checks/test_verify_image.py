import os
import subprocess
import sys

from fcstnyctaxi.lib.utils import get_project_root_dir

_PROBE = get_project_root_dir() / "docker" / "verify_image.py"
# require_git_hash's <full sha>[-dirty] form.
_PASSED = "a" * 40 + "-dirty"


def _run_probe(baked: str) -> subprocess.CompletedProcess[str]:
    """Run the probe as cloudbuild.yaml does: script on stdin, hash as argv[1]."""
    return subprocess.run(
        [sys.executable, "-", _PASSED],
        input=_PROBE.read_text(),
        env={**os.environ, "FCST_GIT_HASH": baked},
        capture_output=True,
        text=True,
    )


def test_shared_probe_accepts_the_hash_the_build_passed():
    """A baked hash equal to the passed one exits 0."""
    assert _run_probe(_PASSED).returncode == 0


def test_shared_probe_rejects_a_baked_hash_the_build_did_not_pass():
    """A baked hash that lost its -dirty suffix fails the build step."""
    result = _run_probe("a" * 40)

    assert result.returncode != 0
    assert f"FCST_GIT_HASH is {'a' * 40!r}" in result.stderr
