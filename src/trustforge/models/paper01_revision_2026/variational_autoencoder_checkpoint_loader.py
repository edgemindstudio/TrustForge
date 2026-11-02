"""Candidate governed artifact-to-model VAE checkpoint loader.

Paper 1 revision 2026.

This module verifies manifest identity, canonical provenance, and
state-file byte integrity before isolated architecture-gated
restoration.

Successful loading does not establish independently authenticated
checkpoint origin, training authenticity, scientific acceptance,
publication approval, deployment approval, or training authorization.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from trustforge.provenance import canonical_json_bytes

from trustforge.models.paper01_revision_2026.gan_checkpoint_v3_secure_io import (
    read_regular_file_securely,
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

from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_artifact import (
    verify_vae_checkpoint_state_artifacts,
)

from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_npz import (
    decode_network_state,
)

from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_state import (
    variable_specification,
)

from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_fingerprint_registry import (
    validate_vae_architecture_fingerprint,
)

from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_anchored_bundle import (
    read_anchored_vae_checkpoint_bundle,
)

from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_restore import (
    restore_vae_state_into_isolated_models,
)


MAX_MANIFEST_BYTES = 1024 * 1024
MAX_PROVENANCE_BYTES = 1024 * 1024

PUBLICATION_LOCK = ".vae_checkpoint_publication.lock"

STATE_NAMES = {
    "encoder": "vae_encoder_state.npz",
    "decoder": "vae_decoder_state.npz",
}


def _read_canonical_json(path, *, max_bytes, newline):
    raw = read_regular_file_securely(
        path,
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


def _validate_requested_identity(
    metadata,
    *,
    dataset_id,
    experimental_seed,
    class_mapping,
):
    if type(dataset_id) is not str:
        raise TypeError(
            "requested dataset ID must be a string"
        )

    if type(experimental_seed) is not int:
        raise TypeError(
            "requested experimental seed must be an integer"
        )

    if type(class_mapping) not in (tuple, list):
        raise TypeError(
            "requested class mapping must be a sequence"
        )

    if metadata["dataset_id"] != dataset_id:
        raise ValueError(
            "VAE checkpoint dataset identity mismatch"
        )

    if metadata["experimental_seed"] != experimental_seed:
        raise ValueError(
            "VAE checkpoint experimental seed mismatch"
        )

    if metadata["class_mapping"] != list(class_mapping):
        raise ValueError(
            "VAE checkpoint ordered class mapping mismatch"
        )


def load_governed_vae_checkpoint(
    directory,
    *,
    requested_dataset_id,
    requested_experimental_seed,
    requested_class_mapping,
):
    """Load one candidate VAE checkpoint into private models.

    Requires explicit caller-supplied experimental identity.

    Reads each state archive once, validates its SHA-256 against the
    manifest, and decodes precisely the verified in-memory bytes.

    This engineering loader does not authorize production use.
    """
    if not os.environ.get("SLURM_JOB_ID", "").strip():
        raise RuntimeError(
            "VAE checkpoint loading requires Slurm"
        )

    bundle = read_anchored_vae_checkpoint_bundle(
        directory,
        requested_dataset_id=requested_dataset_id,
        requested_experimental_seed=requested_experimental_seed,
        requested_class_mapping=requested_class_mapping,
    )

    metadata = bundle["metadata"]
    provenance = bundle["dataset_provenance"]
    verified = bundle["verified_state_bytes"]

    def validate_architecture(*, encoder, decoder):
        return validate_vae_architecture_fingerprint(
            dataset_id=metadata["dataset_id"],
            encoder=encoder,
            decoder=decoder,
        )

    def decode_verified_state(*, encoder, decoder):
        result = {}

        for role, network in (
            ("encoder", encoder),
            ("decoder", decoder),
        ):
            specification = variable_specification(
                network,
                role=role,
            )

            result[role] = decode_network_state(
                role=role,
                raw=verified[STATE_NAMES[role]],
                specification=specification,
            )

        return result

    encoder, decoder = restore_vae_state_into_isolated_models(
        num_classes=metadata["class_count"],
        latent_dim=metadata["latent_dim"],
        pre_assignment_validator=validate_architecture,
        verified_state_decoder=decode_verified_state,
    )

    validate_vae_architecture_fingerprint(
        dataset_id=metadata["dataset_id"],
        encoder=encoder,
        decoder=decoder,
    )

    return {
        "metadata": metadata,
        "dataset_provenance": provenance,
        "encoder": encoder,
        "decoder": decoder,
    }


__all__ = [
    "MAX_MANIFEST_BYTES",
    "MAX_PROVENANCE_BYTES",
    "load_governed_vae_checkpoint",
]
