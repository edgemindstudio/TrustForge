"""Descriptor-anchored governed VAE checkpoint bundle reader.

Reads one candidate checkpoint bundle through a single directory
descriptor, returning validated metadata, provenance, and verified
in-memory NPZ bytes.

Does not construct models, authenticate provenance independently,
provide an immutable snapshot, or authorize training.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat

from trustforge.provenance import canonical_json_bytes

from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_directory_anchor import (
    open_checkpoint_directory,
    anchored_member_exists,
    read_anchored_regular_file,
)

from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_anchored_artifact import (
    verify_anchored_vae_checkpoint_state_artifacts,
)

from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_manifest import (
    MANIFEST_FILENAME,
    validate_vae_checkpoint_manifest,
)

from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_compatibility import (
    validate_vae_checkpoint_architecture_compatibility,
)

from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_provenance import (
    VAE_PROVENANCE_FILENAME,
    validate_vae_checkpoint_provenance,
)

from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_contract import (
    MAX_MANIFEST_BYTES,
    MAX_PROVENANCE_BYTES,
    PUBLICATION_LOCK,
    _validate_requested_identity,
)


def _read_anchored_canonical_json(
    directory_fd,
    filename,
    *,
    max_bytes,
    newline,
):
    raw = read_anchored_regular_file(
        directory_fd,
        filename,
        max_bytes=max_bytes,
    )

    try:
        loaded = json.loads(raw)
    except (UnicodeError, ValueError) as exc:
        raise ValueError(
            "invalid VAE checkpoint JSON"
        ) from exc

    if type(loaded) is not dict:
        raise ValueError(
            "VAE checkpoint JSON must be an object"
        )

    canonical = canonical_json_bytes(loaded)

    if newline:
        canonical += b"\n"

    if raw != canonical:
        raise ValueError(
            "noncanonical VAE checkpoint JSON"
        )

    return loaded, raw


def read_anchored_vae_checkpoint_bundle_fd(
    directory_fd,
    *,
    requested_dataset_id,
    requested_experimental_seed,
    requested_class_mapping,
):
    """Verify a checkpoint using a caller-owned directory FD."""
    if type(directory_fd) is not int:
        raise TypeError("checkpoint directory FD must be an integer")
    try:
        directory_stat = os.fstat(directory_fd)
    except OSError as exc:
        raise ValueError("invalid checkpoint directory FD") from exc
    if not stat.S_ISDIR(directory_stat.st_mode):
        raise ValueError("checkpoint FD is not a directory")

    if anchored_member_exists(directory_fd, PUBLICATION_LOCK):
        raise RuntimeError(
            "VAE checkpoint publication incomplete"
        )

    loaded_metadata, _ = _read_anchored_canonical_json(
        directory_fd,
        MANIFEST_FILENAME,
        max_bytes=MAX_MANIFEST_BYTES,
        newline=True,
    )

    metadata = validate_vae_checkpoint_manifest(
        loaded_metadata
    )

    metadata = validate_vae_checkpoint_architecture_compatibility(
        metadata
    )

    _validate_requested_identity(
        metadata,
        dataset_id=requested_dataset_id,
        experimental_seed=requested_experimental_seed,
        class_mapping=requested_class_mapping,
    )

    loaded_provenance, provenance_bytes = (
        _read_anchored_canonical_json(
            directory_fd,
            VAE_PROVENANCE_FILENAME,
            max_bytes=MAX_PROVENANCE_BYTES,
            newline=False,
        )
    )

    digest = hashlib.sha256(provenance_bytes).hexdigest()

    if digest != metadata["dataset_provenance_sha256"]:
        raise ValueError(
            "VAE checkpoint provenance SHA-256 mismatch"
        )

    provenance = validate_vae_checkpoint_provenance(
        loaded_provenance,
        expected_dataset_id=metadata["dataset_id"],
    )

    verified_bytes = (
        verify_anchored_vae_checkpoint_state_artifacts(
            directory_fd,
            metadata,
        )
    )

    if anchored_member_exists(directory_fd, PUBLICATION_LOCK):
        raise RuntimeError(
            "VAE checkpoint publication incomplete"
        )

    return {
        "metadata": metadata,
        "dataset_provenance": provenance,
        "verified_state_bytes": verified_bytes,
    }


def read_anchored_vae_checkpoint_bundle(
    directory,
    *,
    requested_dataset_id,
    requested_experimental_seed,
    requested_class_mapping,
):
    """Verify a candidate checkpoint through one anchored directory."""
    with open_checkpoint_directory(directory) as directory_fd:
        return read_anchored_vae_checkpoint_bundle_fd(
            directory_fd,
            requested_dataset_id=requested_dataset_id,
            requested_experimental_seed=requested_experimental_seed,
            requested_class_mapping=requested_class_mapping,
        )


__all__ = [
    "read_anchored_vae_checkpoint_bundle",
    "read_anchored_vae_checkpoint_bundle_fd",
]
