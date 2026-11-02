"""Read-only pre-commit verification of prepared VAE artifacts.

This module does not publish commit records, authenticate checkpoint
origin, establish immutable filesystem contents, or authorize training.
"""

from __future__ import annotations

from .variational_autoencoder_checkpoint_publication_input import (
    validate_vae_publication_input,
)

from .variational_autoencoder_checkpoint_anchored_bundle import (
    read_anchored_vae_checkpoint_bundle_fd,
)

from .variational_autoencoder_checkpoint_manifest import (
    MANIFEST_FILENAME,
)
from .variational_autoencoder_checkpoint_directory_anchor import (
    read_anchored_regular_file,
)
from .variational_autoencoder_checkpoint_contract import (
    MAX_MANIFEST_BYTES,
)


def verify_precommit_vae_generation(
    generation_fd,
    *,
    prepared,
    requested_dataset_id,
    requested_experimental_seed,
    requested_class_mapping,
):
    """Compare anchored checkpoint evidence with validated prepared bytes."""
    validated = validate_vae_publication_input(prepared)

    manifest_bytes = read_anchored_regular_file(
        generation_fd,
        MANIFEST_FILENAME,
        max_bytes=MAX_MANIFEST_BYTES,
    )

    if manifest_bytes != validated['artifacts'][MANIFEST_FILENAME]:
        raise ValueError('precommit VAE manifest bytes mismatch')

    bundle = read_anchored_vae_checkpoint_bundle_fd(
        generation_fd,
        requested_dataset_id=requested_dataset_id,
        requested_experimental_seed=requested_experimental_seed,
        requested_class_mapping=requested_class_mapping,
    )

    if bundle["metadata"] != validated["metadata"]:
        raise ValueError("precommit VAE metadata mismatch")

    if bundle["dataset_provenance"] != validated["dataset_provenance"]:
        raise ValueError("precommit VAE provenance mismatch")

    expected_states = {
        name: raw
        for name, raw in validated["artifacts"].items()
        if name.endswith("_state.npz")
    }

    if set(bundle["verified_state_bytes"]) != set(expected_states):
        raise ValueError("precommit VAE state inventory mismatch")

    for name, expected in expected_states.items():
        if bundle["verified_state_bytes"][name] != expected:
            raise ValueError(
                f"precommit VAE state bytes mismatch: {name}"
            )

    return {
        "metadata": bundle["metadata"],
        "dataset_provenance": bundle["dataset_provenance"],
        "verified_state_bytes": bundle["verified_state_bytes"],
    }


__all__ = ["verify_precommit_vae_generation"]
