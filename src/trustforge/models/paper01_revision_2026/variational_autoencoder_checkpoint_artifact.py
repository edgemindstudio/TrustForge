"""Secure byte-integrity verification for VAE checkpoint state files.

Paper 1 revision 2026.

This module verifies the exact state-file inventory and SHA-256
digests using securely opened, size-bounded regular files.

It does not decode NPZ archives, validate tensor contents,
restore models, publish checkpoints, or authorize training.

Verified bytes are returned so a later archive decoder need
not reopen potentially mutable filesystem paths.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from trustforge.models.paper01_revision_2026.gan_checkpoint_v3_secure_io import (
    read_regular_file_securely,
)
from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_manifest import (
    STATE_FILENAMES,
    validate_vae_checkpoint_manifest,
)
from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_npz import (
    MAX_ARCHIVE_BYTES,
)


# Provisional engineering limit shared with the canonical NPZ codec.
# This is not an accepted production checkpoint budget.
MAX_STATE_FILE_BYTES = MAX_ARCHIVE_BYTES

STATE_PREFIX = "vae_"
STATE_SUFFIX = "_state.npz"


def verify_vae_checkpoint_state_artifacts(
    directory,
    metadata,
):
    """Return verified encoder/decoder state-file bytes.

    Validation includes:
    - Exact proposed manifest schema
    - Real, nonsymlink checkpoint directory
    - No active VAE publication lock
    - Exact inventory of VAE state-file candidates
    - Secure, bounded reads of regular files
    - Exact SHA-256 matching

    This is byte-integrity verification only, not complete
    checkpoint verification or model restoration.
    """
    validated = validate_vae_checkpoint_manifest(metadata)

    requested_root = Path(directory).expanduser()

    if requested_root.is_symlink() or not requested_root.is_dir():
        raise ValueError("invalid VAE checkpoint directory")

    root = requested_root.resolve()

    if (root / ".vae_checkpoint_publication.lock").exists():
        raise RuntimeError("VAE checkpoint publication incomplete")

    state_files = validated["state_files"]

    if set(state_files) != STATE_FILENAMES:
        raise ValueError("VAE checkpoint state inventory mismatch")

    actual = {
        entry.name
        for entry in root.iterdir()
        if (
            entry.name.startswith(STATE_PREFIX)
            and entry.name.endswith(STATE_SUFFIX)
        )
    }

    if actual != STATE_FILENAMES:
        raise ValueError(
            "VAE checkpoint state-file inventory mismatch"
        )

    verified = {}

    for name in sorted(STATE_FILENAMES):
        path = root / name

        raw = read_regular_file_securely(
            path,
            max_bytes=MAX_STATE_FILE_BYTES,
        )

        digest = hashlib.sha256(raw).hexdigest()

        if digest != state_files[name]:
            raise ValueError(
                f"VAE checkpoint state SHA-256 mismatch: {name}"
            )

        verified[name] = raw

    return verified


__all__ = [
    "MAX_STATE_FILE_BYTES",
    "verify_vae_checkpoint_state_artifacts",
]
