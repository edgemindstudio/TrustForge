"""Pinned candidate VAE architecture fingerprint references.

Paper 1 revision 2026.

Reference values were observed from fresh model constructions
with seeds 42 and 43 under TensorFlow 2.20.0, Keras 3.11.3.

These are engineering reference values, not independently
authenticated architecture approvals or training authority.
"""

from __future__ import annotations

import re

from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_manifest import (
    DATASET_CLASS_COUNTS,
)
from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_fingerprint import (
    fingerprint_vae_architecture,
)


REFERENCE_FINGERPRINTS = {
    "malex": (
        "e5155362ee74b77e75cc62612286fcf681d24f71d1af0e0395a37e7e04d5eea5"
    ),
    "cicmaldroid2020": (
        "d9379b715f60d9688a161a67502889b01c1b26109f8051b1db3bcad607a5f7e5"
    ),
    "ustc_tfc2016": (
        "0fc28839865f075052c499e3ac58c397f0bbfa99e7565da00f916f7d1cc0e1c5"
    ),
    "malnet_image": (
        "76e1d059b47b32411ea764d03a0ed3df759582b3888e7b1349286cd4eded28de"
    ),
}

REFERENCE_FINGERPRINTS = dict(REFERENCE_FINGERPRINTS)


def expected_vae_architecture_fingerprint(*, dataset_id):
    """Return a pinned engineering reference for one dataset."""
    if type(dataset_id) is not str:
        raise TypeError("dataset_id must be a string")

    if dataset_id not in DATASET_CLASS_COUNTS:
        raise ValueError("unsupported governed dataset")

    if set(REFERENCE_FINGERPRINTS) != set(DATASET_CLASS_COUNTS):
        raise RuntimeError("architecture fingerprint registry incomplete")

    digest = REFERENCE_FINGERPRINTS[dataset_id]

    if (
        type(digest) is not str
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
    ):
        raise RuntimeError("invalid pinned architecture fingerprint")

    return digest


def validate_vae_architecture_fingerprint(
    *,
    dataset_id,
    encoder,
    decoder,
):
    """Compare actual paired-model structure to a pinned reference.

    This does not authenticate a checkpoint or authorize
    restoration, publication, deployment, or training.
    """
    expected = expected_vae_architecture_fingerprint(
        dataset_id=dataset_id,
    )

    class_count = DATASET_CLASS_COUNTS[dataset_id]

    actual = fingerprint_vae_architecture(
        encoder=encoder,
        decoder=decoder,
        class_count=class_count,
        latent_dim=100,
    )

    if actual != expected:
        raise ValueError(
            "VAE architecture fingerprint mismatch"
        )

    return actual


__all__ = [
    "REFERENCE_FINGERPRINTS",
    "expected_vae_architecture_fingerprint",
    "validate_vae_architecture_fingerprint",
]
