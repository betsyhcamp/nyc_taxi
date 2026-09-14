import re

# <host>/<project>/<repository>/<image>@sha256:<64 hex>. `prefix` is the first three
# segments at any image-name depth, so a nested name is legal. Not `repository`:
# that word already names the bare registry segment in ImageRef and dev.yaml.
_IMAGE_REF = re.compile(
    r"^(?P<prefix>[^/@]+/[^/@]+/[^/@]+)/(?P<name>[^@]+)@sha256:(?P<digest>[0-9a-f]{64})$"
)

_EXPECTED_SHAPE = "<host>/<project>/<repository>/<image>@sha256:<64 hex>"


def require_digest_ref(image_ref: str | None, source: str) -> str:
    """Validate a digest-pinned registry reference.

    Args:
        image_ref: The reference to check.
        source: What supplied the value, named in the error so the reader knows
            what to fix. Keeps this module free of any variable name.

    Returns:
        The reference, unchanged.

    Raises:
        ValueError: `image_ref` is empty or is not digest-pinned.
    """
    if not image_ref:
        raise ValueError(
            f"{source} not set. Must be {_EXPECTED_SHAPE}, set before import."
        )
    if not _IMAGE_REF.match(image_ref):
        raise ValueError(
            f"{source} must be {_EXPECTED_SHAPE}, got {image_ref!r}. A tag can be "
            "repointed, so one spec would run different code over time."
        )
    return image_ref


def artifact_registry_prefix(image_ref: str) -> str:
    """The <host>/<project>/<repository> location, at any image-name depth.

    Callers classify images by this prefix, so it must stop at the repository: one
    segment deeper and a sibling image in the same repository stops matching.

    Args:
        image_ref: A digest-pinned registry reference.

    Returns:
        The first three segments, no trailing slash.

    Raises:
        ValueError: `image_ref` is not digest-pinned.
    """
    match = _IMAGE_REF.match(image_ref)
    if match is None:
        raise ValueError(f"image_ref must be {_EXPECTED_SHAPE}, got {image_ref!r}.")
    return match["prefix"]
