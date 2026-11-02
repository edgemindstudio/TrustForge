"""Integrated synthetic lifecycle regression for revised Paper 1 VAE."""

import numpy as np
import pytest
import optree
import keras
import tensorflow as tf

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_model as model,
    variational_autoencoder_traversal as traversal,
    variational_autoencoder_data_integration as data,
    variational_autoencoder_training_step as training,
    variational_autoencoder_validation_step as validation,
)


class SyntheticAdapter:
    def __init__(self):
        self.sizes = {"train": 5, "validation": 3}
        self.requests = []

    def split_size(self, split):
        if split not in self.sizes:
            raise AssertionError("Unauthorized partition access")
        self.requests.append(("size", split))
        return self.sizes[split]

    def canonical_batch(self, split, logical_positions):
        if split not in self.sizes:
            raise AssertionError("Unauthorized partition access")

        positions = np.asarray(logical_positions)
        self.requests.append(("batch", split))

        images = np.full(
            (len(positions), 40, 40, 3),
            0.5,
            dtype=np.float32,
        )
        labels = np.zeros(
            (len(positions), 2),
            dtype=np.float32,
        )
        labels[np.arange(len(positions)), positions % 2] = 1.0
        return images, labels


def make_models():
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)
    encoder, decoder = model.build_models(
        latent_dim=100,
        num_classes=2,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    return encoder, decoder, optimizer


def snapshot(encoder, decoder):
    return [
        v.numpy().copy()
        for v in list(encoder.variables) + list(decoder.variables)
    ]


def test_integrated_training_and_complete_validation():
    adapter = SyntheticAdapter()
    encoder, decoder, optimizer = make_models()

    train_positions = next(
        traversal.position_batches(
            adapter=adapter,
            split="train",
            batch_size=2,
            experimental_seed=42,
            epoch=0,
        )
    )

    x, y = data.canonical_vae_batch(
        adapter=adapter,
        split="train",
        logical_positions=train_positions,
        num_classes=2,
    )

    train_result = training.vae_training_step(
        encoder=encoder,
        decoder=decoder,
        optimizer=optimizer,
        canonical_images=tf.convert_to_tensor(x),
        class_labels=tf.convert_to_tensor(y),
        seed=tf.constant([42, 0], tf.int32),
    )

    assert int(optimizer.iterations.numpy()) == 1
    assert all(np.isfinite(v) for v in train_result.values())

    before = snapshot(encoder, decoder)
    results = []
    seen = []

    for index, positions in enumerate(
        traversal.position_batches(
            adapter=adapter,
            split="validation",
            batch_size=2,
        )
    ):
        seen.extend(positions.tolist())

        x, y = data.canonical_vae_batch(
            adapter=adapter,
            split="validation",
            logical_positions=positions,
            num_classes=2,
        )

        results.append(
            validation.vae_validation_batch(
                encoder=encoder,
                decoder=decoder,
                canonical_images=tf.convert_to_tensor(x),
                class_labels=tf.convert_to_tensor(y),
                seed=tf.constant([42, index], tf.int32),
            )
        )

    aggregated = validation.aggregate_validation_batches(results)

    assert seen == [0, 1, 2]
    assert [item["count"] for item in results] == [2, 1]
    assert aggregated["count"] == 3
    assert all(
        np.isfinite(aggregated[key])
        for key in ("total_loss", "reconstruction_loss", "kl_loss")
    )

    np.testing.assert_allclose(
        aggregated["total_loss"],
        aggregated["reconstruction_loss"] + aggregated["kl_loss"],
        rtol=1e-5,
        atol=1e-4,
    )

    for old, new in zip(before, snapshot(encoder, decoder)):
        np.testing.assert_array_equal(old, new)

    assert int(optimizer.iterations.numpy()) == 1
    assert all(
        split in {"train", "validation"}
        for _, split in adapter.requests
    )


def test_integrated_test_partition_rejected_before_adapter_access():
    adapter = SyntheticAdapter()

    with pytest.raises(ValueError):
        list(
            traversal.position_batches(
                adapter=adapter,
                split="test",
                batch_size=2,
            )
        )

    assert adapter.requests == []

    with pytest.raises(ValueError):
        data.canonical_vae_batch(
            adapter=adapter,
            split="test",
            logical_positions=np.array([0], dtype=np.int64),
            num_classes=2,
        )

    assert adapter.requests == []
