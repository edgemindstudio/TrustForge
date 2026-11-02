"""Candidate fingerprint-gated isolated VAE restoration.

Paper 1 revision 2026.

This module provides a separately testable architecture gate
around existing isolated restoration.

It does not authenticate checkpoint provenance or authorize
production restoration, publication, deployment, or training.
"""

from __future__ import annotations

from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_manifest import (
    DATASET_CLASS_COUNTS,
)
from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_fingerprint_registry import (
    expected_vae_architecture_fingerprint,
    validate_vae_architecture_fingerprint,
)
from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_restore import (
    restore_vae_state_into_isolated_models,
)


def restore_vae_state_with_fingerprint_gate(
    *,
    dataset_id,
    state,
):
    """Restore into isolated models with candidate architecture checks.

    The actual isolated restoration pair is fingerprint-checked
    before checkpoint assignment. The returned pair is checked
    again before crossing the wrapper's success boundary.
    """
    if type(dataset_id) is not str:
        raise TypeError("dataset_id must be a string")

    expected_vae_architecture_fingerprint(
        dataset_id=dataset_id,
    )

    class_count = DATASET_CLASS_COUNTS[dataset_id]

    restored_encoder, restored_decoder = (
        restore_vae_state_into_isolated_models(
            num_classes=class_count,
            state=state,
            latent_dim=100,
            pre_assignment_validator=(
                lambda *, encoder, decoder:
                    validate_vae_architecture_fingerprint(
                        dataset_id=dataset_id,
                        encoder=encoder,
                        decoder=decoder,
                    )
            ),
        )
    )

    validate_vae_architecture_fingerprint(
        dataset_id=dataset_id,
        encoder=restored_encoder,
        decoder=restored_decoder,
    )

    return restored_encoder, restored_decoder


__all__ = [
    "restore_vae_state_with_fingerprint_gate",
]
