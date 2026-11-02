"""Descriptor-anchored VAE checkpoint state-artifact verification.

All state-file operations use an already-open directory descriptor.

This establishes bounded byte-integrity verification, not an
immutable filesystem snapshot or authenticated checkpoint origin.
"""

from __future__ import annotations

import hashlib

from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_artifact import (
    MAX_STATE_FILE_BYTES,
    STATE_PREFIX,
    STATE_SUFFIX,
)

from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_manifest import (
    STATE_FILENAMES,
    validate_vae_checkpoint_manifest,
)

from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_directory_anchor import (
    anchored_member_names,
    anchored_member_exists,
    read_anchored_regular_file,
)


PUBLICATION_LOCK = ".vae_checkpoint_publication.lock"


def verify_anchored_vae_checkpoint_state_artifacts(
    directory_fd,
    metadata,
):
    """Return verified state bytes from one anchored directory.

    Rejects incomplete publication, unexpected state inventory,
    invalid files, and SHA-256 mismatches before returning bytes.
    """
    validated = validate_vae_checkpoint_manifest(metadata)

    if anchored_member_exists(directory_fd, PUBLICATION_LOCK):
        raise RuntimeError(
            "VAE checkpoint publication incomplete"
        )

    state_files = validated["state_files"]

    if set(state_files) != STATE_FILENAMES:
        raise ValueError(
            "VAE checkpoint state inventory mismatch"
        )

    actual = {
        name
        for name in anchored_member_names(directory_fd)
        if (
            name.startswith(STATE_PREFIX)
            and name.endswith(STATE_SUFFIX)
        )
    }

    if actual != STATE_FILENAMES:
        raise ValueError(
            "VAE checkpoint state-file inventory mismatch"
        )

    verified = {}

    for name in sorted(STATE_FILENAMES):
        raw = read_anchored_regular_file(
            directory_fd,
            name,
            max_bytes=MAX_STATE_FILE_BYTES,
        )

        digest = hashlib.sha256(raw).hexdigest()

        if digest != state_files[name]:
            raise ValueError(
                f"VAE checkpoint state SHA-256 mismatch: {name}"
            )

        verified[name] = raw

    if anchored_member_exists(directory_fd, PUBLICATION_LOCK):
        raise RuntimeError(
            "VAE checkpoint publication incomplete"
        )

    return verified


__all__ = [
    "verify_anchored_vae_checkpoint_state_artifacts",
]
