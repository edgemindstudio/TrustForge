"""In-memory preparation of candidate VAE checkpoint artifacts.

Requires caller-supplied scientific metadata and dataset provenance.
Does not train, select, publish, authenticate, or authorize checkpoints.
"""

from __future__ import annotations

import hashlib

from trustforge.provenance import canonical_json_bytes

from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_manifest import (
    MANIFEST_FILENAME,
    REQUIRED_FIELDS,
    validate_vae_checkpoint_manifest,
)

from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_compatibility import (
    validate_vae_checkpoint_architecture_compatibility,
)

from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_provenance import (
    VAE_PROVENANCE_FILENAME,
    validate_vae_checkpoint_provenance,
)

from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_fingerprint_registry import (
    validate_vae_architecture_fingerprint,
)

from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_state import (
    capture_vae_state,
    variable_specification,
)

from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_npz import (
    encode_network_state,
)


DERIVED_FIELDS = frozenset({
    "state_files",
    "dataset_provenance_file",
    "dataset_provenance_sha256",
})


def prepare_vae_checkpoint_artifacts(
    *,
    encoder,
    decoder,
    manifest_evidence,
    dataset_provenance,
):
    """Construct validated checkpoint artifacts entirely in memory."""
    if type(manifest_evidence) is not dict:
        raise TypeError("manifest evidence must be a dictionary")

    expected = REQUIRED_FIELDS - DERIVED_FIELDS

    if set(manifest_evidence) != expected:
        raise ValueError(
            "VAE checkpoint preparation evidence fields mismatch"
        )

    dataset_id = manifest_evidence["dataset_id"]

    validated_provenance = validate_vae_checkpoint_provenance(
        dataset_provenance,
        expected_dataset_id=dataset_id,
    )

    validate_vae_architecture_fingerprint(
        dataset_id=dataset_id,
        encoder=encoder,
        decoder=decoder,
    )

    captured = capture_vae_state(
        encoder=encoder,
        decoder=decoder,
    )

    archives = {}

    for role, network in (
        ("encoder", encoder),
        ("decoder", decoder),
    ):
        specification = variable_specification(
            network,
            role=role,
        )

        archives[f"vae_{role}_state.npz"] = (
            encode_network_state(
                role=role,
                state=captured[role],
                specification=specification,
            )
        )

    provenance_bytes = canonical_json_bytes(
        validated_provenance
    )

    metadata = {
        **manifest_evidence,
        "state_files": {
            name: hashlib.sha256(raw).hexdigest()
            for name, raw in archives.items()
        },
        "dataset_provenance_file": VAE_PROVENANCE_FILENAME,
        "dataset_provenance_sha256": hashlib.sha256(
            provenance_bytes
        ).hexdigest(),
    }

    metadata = validate_vae_checkpoint_architecture_compatibility(
        validate_vae_checkpoint_manifest(metadata)
    )

    manifest_bytes = canonical_json_bytes(metadata) + b"\n"

    artifacts = {
        MANIFEST_FILENAME: manifest_bytes,
        VAE_PROVENANCE_FILENAME: provenance_bytes,
        **archives,
    }

    return {
        "metadata": metadata,
        "dataset_provenance": validated_provenance,
        "artifacts": artifacts,
    }


__all__ = [
    "DERIVED_FIELDS",
    "prepare_vae_checkpoint_artifacts",
]
