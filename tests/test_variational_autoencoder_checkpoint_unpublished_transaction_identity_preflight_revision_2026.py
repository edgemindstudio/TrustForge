"""Additional identity-preflight regressions for unpublished VAE generations.

Disposable synthetic fixtures only. No training or commit publication.
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


@pytest.mark.parametrize(
    "changes, error_type, message",
    [
        (
            {"requested_dataset_id": "unsupported_dataset"},
            ValueError,
            "dataset identity mismatch",
        ),
        (
            {"requested_class_mapping": list(reversed(CLASS_MAPPING))},
            ValueError,
            "ordered class mapping mismatch",
        ),
        (
            {"requested_experimental_seed": "42"},
            TypeError,
            "requested experimental seed must be an integer",
        ),
        (
            {"requested_class_mapping": "invalid"},
            TypeError,
            "requested class mapping must be a sequence",
        ),
    ],
)
def test_invalid_requested_identity_never_reserves_generation(
    transaction_inputs,
    changes,
    error_type,
    message,
):
    generations, prepared = transaction_inputs

    parent_fd = open_parent(generations)

    request = {
        "requested_dataset_id": DATASET_ID,
        "requested_experimental_seed": SEED,
        "requested_class_mapping": CLASS_MAPPING,
    }
    request.update(changes)

    try:
        with pytest.raises(error_type, match=message):
            transaction.prepare_unpublished_vae_generation(
                parent_fd,
                GENERATION_ID,
                prepared=prepared,
                **request,
            )

        assert not (generations / GENERATION_ID).exists()
        assert list(generations.iterdir()) == []

        # The caller retains ownership of its descriptor.
        os.fstat(parent_fd)

    finally:
        os.close(parent_fd)
