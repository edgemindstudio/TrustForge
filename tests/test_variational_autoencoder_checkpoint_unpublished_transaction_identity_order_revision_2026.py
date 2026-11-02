"""Regression for VAE transaction identity-validation ordering.

Uses disposable synthetic fixtures. No commit publication or training.
"""

import os

import optree
import keras
import pytest
import tensorflow as tf

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_checkpoint_unpublished_transaction as transaction,
)

from tests.test_variational_autoencoder_checkpoint_loader_roundtrip_revision_2026 import (
    synthetic_bundle,
    DATASET_ID,
    SEED,
    CLASS_MAPPING,
)

from tests.test_variational_autoencoder_checkpoint_unpublished_transaction_filesystem_revision_2026 import (
    transaction_inputs,
    open_parent,
    GENERATION_ID,
)


def test_seed_identity_rejected_before_reservation(
    transaction_inputs,
):
    generations, prepared = transaction_inputs

    parent_fd = open_parent(generations)

    try:
        with pytest.raises(
            ValueError,
            match="experimental seed mismatch",
        ):
            transaction.prepare_unpublished_vae_generation(
                parent_fd,
                GENERATION_ID,
                prepared=prepared,
                requested_dataset_id=DATASET_ID,
                requested_experimental_seed=SEED + 1,
                requested_class_mapping=CLASS_MAPPING,
            )

        assert not (generations / GENERATION_ID).exists(), (
            "Identity mismatch must be rejected before "
            "generation directory creation"
        )

        os.fstat(parent_fd)

    finally:
        os.close(parent_fd)
