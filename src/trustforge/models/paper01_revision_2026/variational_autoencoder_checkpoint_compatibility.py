"""Candidate VAE checkpoint architecture compatibility contract.

Paper 1 revision 2026.

This module establishes compatibility with the current
governed VAE model implementation.

Compatibility does not prove checkpoint authenticity,
training provenance, scientific acceptance, or authority
to restore or publish a checkpoint.
"""

from __future__ import annotations

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_checkpoint_manifest as manifest,
    variational_autoencoder_model as model,
)


CANDIDATE_ARCHITECTURE_VERSION = (
    "paper01_revision2026_conditional_vae_arch_v1"
)


def validate_vae_checkpoint_architecture_compatibility(
    metadata,
):
    """Validate a manifest against the candidate VAE architecture.

    Returns validated manifest metadata.

    This function does not construct models, decode archives,
    access datasets, assign variables, or authorize training.
    """
    validated = manifest.validate_vae_checkpoint_manifest(
        metadata
    )

    if (
        validated["architecture_version"]
        != CANDIDATE_ARCHITECTURE_VERSION
    ):
        raise ValueError(
            "unsupported VAE checkpoint architecture version"
        )

    expected_constants = {
        "image_shape": list(model.IMAGE_SHAPE),
        "latent_dim": model.LATENT_DIM,
        "encoder_dense_units": model.ENCODER_DENSE_UNITS,
        "decoder_feature_channels": (
            model.DECODER_FEATURE_CHANNELS
        ),
    }

    for name, expected in expected_constants.items():
        if validated[name] != expected:
            raise ValueError(
                f"VAE architecture compatibility mismatch: {name}"
            )

    expected_classes = manifest.DATASET_CLASS_COUNTS[
        validated["dataset_id"]
    ]

    if validated["class_count"] != expected_classes:
        raise ValueError(
            "VAE checkpoint dataset/class-count mismatch"
        )

    return validated


__all__ = [
    "CANDIDATE_ARCHITECTURE_VERSION",
    "validate_vae_checkpoint_architecture_compatibility",
]
