"""Pure input validation for governed VAE checkpoint publication.

Does not write files, publish checkpoints, or authorize training.
"""

from __future__ import annotations

import hashlib

from trustforge.provenance import canonical_json_bytes

from .variational_autoencoder_checkpoint_manifest import (
    MANIFEST_FILENAME,
    validate_vae_checkpoint_manifest,
)
from .variational_autoencoder_checkpoint_preparation import (
    VAE_PROVENANCE_FILENAME,
)


STATE_FILENAMES = frozenset({
    "vae_encoder_state.npz",
    "vae_decoder_state.npz",
})

PREPARED_FIELDS = frozenset({
    "metadata",
    "dataset_provenance",
    "artifacts",
})


def validate_vae_publication_input(prepared):
    """Validate the exact prepared artifact package without I/O."""
    if type(prepared) is not dict:
        raise TypeError("prepared publication must be a dictionary")

    if set(prepared) != PREPARED_FIELDS:
        raise ValueError("prepared publication field mismatch")

    metadata = prepared["metadata"]
    provenance = prepared["dataset_provenance"]
    artifacts = prepared["artifacts"]

    if type(metadata) is not dict:
        raise TypeError("publication metadata must be a dictionary")

    if type(provenance) is not dict:
        raise TypeError("publication provenance must be a dictionary")

    if type(artifacts) is not dict:
        raise TypeError("publication artifacts must be a dictionary")

    validated = validate_vae_checkpoint_manifest(metadata)

    if validated != metadata:
        raise ValueError("publication metadata validation mismatch")

    if metadata["dataset_provenance_file"] != VAE_PROVENANCE_FILENAME:
        raise ValueError("publication provenance filename mismatch")

    state_hashes = metadata["state_files"]

    if type(state_hashes) is not dict:
        raise TypeError("publication state hashes must be a dictionary")

    if set(state_hashes) != STATE_FILENAMES:
        raise ValueError("publication state inventory mismatch")

    expected_names = (
        STATE_FILENAMES
        | {MANIFEST_FILENAME, VAE_PROVENANCE_FILENAME}
    )

    if set(artifacts) != expected_names:
        raise ValueError("publication artifact inventory mismatch")

    for name, raw in artifacts.items():
        if type(name) is not str or type(raw) is not bytes or not raw:
            raise ValueError("publication artifacts must be nonempty bytes")

    expected_manifest = canonical_json_bytes(metadata) + b"\n"
    expected_provenance = canonical_json_bytes(provenance)

    if artifacts[MANIFEST_FILENAME] != expected_manifest:
        raise ValueError("noncanonical publication manifest")

    if artifacts[VAE_PROVENANCE_FILENAME] != expected_provenance:
        raise ValueError("noncanonical publication provenance")

    provenance_digest = hashlib.sha256(
        artifacts[VAE_PROVENANCE_FILENAME]
    ).hexdigest()

    if provenance_digest != metadata["dataset_provenance_sha256"]:
        raise ValueError("publication provenance SHA-256 mismatch")

    for name in sorted(STATE_FILENAMES):
        actual_digest = hashlib.sha256(artifacts[name]).hexdigest()

        if actual_digest != state_hashes[name]:
            raise ValueError(
                f"publication state SHA-256 mismatch: {name}"
            )

    return {
        "metadata": metadata,
        "dataset_provenance": provenance,
        "artifacts": dict(artifacts),
    }


__all__ = [
    "STATE_FILENAMES",
    "PREPARED_FIELDS",
    "validate_vae_publication_input",
]
