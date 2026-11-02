"""Synthetic contract for the unpublished VAE generation coordinator.

No authoritative commit marker, real-data training, or publication.
"""

import optree
import keras
import pytest
import tensorflow as tf


def test_coordinator_exports_expected_entrypoint():
    from trustforge.models.paper01_revision_2026 import (
        variational_autoencoder_checkpoint_unpublished_transaction as transaction,
    )

    assert callable(
        transaction.prepare_unpublished_vae_generation
    )


def test_coordinator_does_not_export_commit_operation():
    from trustforge.models.paper01_revision_2026 import (
        variational_autoencoder_checkpoint_unpublished_transaction as transaction,
    )

    assert not hasattr(
        transaction,
        "publish_vae_commit",
    )


def test_coordinator_requires_explicit_generation_identity():
    from trustforge.models.paper01_revision_2026 import (
        variational_autoencoder_checkpoint_unpublished_transaction as transaction,
    )

    import inspect

    signature = inspect.signature(
        transaction.prepare_unpublished_vae_generation
    )

    required = {
        "generations_fd",
        "generation_id",
        "prepared",
        "requested_dataset_id",
        "requested_experimental_seed",
        "requested_class_mapping",
    }

    assert required.issubset(signature.parameters)
