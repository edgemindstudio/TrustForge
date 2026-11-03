"""Governed real-dataset training boundary for Paper 1 VAE.

Real-data training is disabled pending implementation acceptance.
This module does not prepare or publish checkpoints.
The synthetic-only runner remains an independent entry point.
"""

from __future__ import annotations

import os

from .gan_adapter_identity import require_gan_adapter_identity
from .gan_dataset_provenance import require_gan_dataset_provenance
from .gan_v3_provenance_adapter import build_gan_v3_provenance
from .gan_checkpoint_manifest import DATASET_CLASS_COUNTS
from .variational_autoencoder_training_runner import (
    _run_vae_training_core,
)


VAE_REAL_DATASET_TRAINING_ENABLED = False

APPROVED_EPOCHS = 2
APPROVED_BATCH_SIZE = 64
APPROVED_LEARNING_RATE = 0.001


def run_governed_vae_training(
    *,
    adapter,
    dataset_id,
    experimental_seed,
):
    """Run only after independent authorization and source preflight.

    Dataset provenance verification must occur on a Slurm
    compute node before any training batches are consumed.

    The returned provenance is candidate checkpoint evidence;
    its presence does not authorize checkpoint publication.
    """
    if VAE_REAL_DATASET_TRAINING_ENABLED is not True:
        raise PermissionError(
            "VAE real-dataset training is not authorized"
        )

    if not os.environ.get("SLURM_JOB_ID"):
        raise PermissionError(
            "VAE real-dataset training requires Slurm allocation"
        )

    require_gan_adapter_identity(
        dataset_id=dataset_id,
        adapter=adapter,
    )

    verified_preflight = require_gan_dataset_provenance(
        dataset_id=dataset_id,
        adapter=adapter,
    )

    dataset_provenance = build_gan_v3_provenance(
        dataset_id=dataset_id,
        verified_preflight=verified_preflight,
    )

    num_classes = DATASET_CLASS_COUNTS[dataset_id]

    trained = _run_vae_training_core(
        adapter=adapter,
        num_classes=num_classes,
        experimental_seed=experimental_seed,
        epochs=APPROVED_EPOCHS,
        batch_size=APPROVED_BATCH_SIZE,
        learning_rate=APPROVED_LEARNING_RATE,
    )

    if trained["epochs_completed"] != APPROVED_EPOCHS:
        raise RuntimeError("incomplete governed VAE training")

    expected_updates = APPROVED_EPOCHS * (
        (
            adapter.split_size("train")
            + APPROVED_BATCH_SIZE - 1
        ) // APPROVED_BATCH_SIZE
    )

    if trained["completed_updates"] != expected_updates:
        raise RuntimeError("governed VAE update count mismatch")

    if trained["publication_status"] != "NOT_PUBLISHED":
        raise RuntimeError(
            "VAE training unexpectedly reported publication"
        )

    return {
        "dataset_id": dataset_id,
        "experimental_seed": experimental_seed,
        "class_count": num_classes,
        "dataset_provenance": dataset_provenance,
        "training_result": trained,
        "checkpoint_status": "NOT_PREPARED",
        "publication_status": "NOT_PUBLISHED",
    }


__all__ = [
    "VAE_REAL_DATASET_TRAINING_ENABLED",
    "run_governed_vae_training",
]
