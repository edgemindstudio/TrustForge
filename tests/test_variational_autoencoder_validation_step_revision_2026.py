"""VAE validation regression tests for Paper 1 revision 2026."""

import numpy as np
import pytest

# Preserve the working dependency initialization sequence.
import optree
import keras
import tensorflow as tf

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_model as model,
    variational_autoencoder_validation_step as validation,
)


def make_case():
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)

    encoder, decoder = model.build_models(
        latent_dim=100,
        num_classes=9,
    )

    images = tf.fill((2, 40, 40, 3), 0.5)
    labels = tf.one_hot([0, 1], 9, dtype=tf.float32)
    seed = tf.constant([42, 0], tf.int32)

    return encoder, decoder, images, labels, seed


def evaluate(encoder, decoder, images, labels, seed):
    return validation.vae_validation_batch(
        encoder=encoder,
        decoder=decoder,
        canonical_images=images,
        class_labels=labels,
        seed=seed,
    )


def snapshot_all(encoder, decoder):
    return [
        variable.numpy().copy()
        for variable in (
            list(encoder.variables)
            + list(decoder.variables)
        )
    ]


def test_validation_preserves_all_model_variables():
    encoder, decoder, images, labels, seed = make_case()

    before = snapshot_all(encoder, decoder)
    result = evaluate(encoder, decoder, images, labels, seed)
    after = snapshot_all(encoder, decoder)

    assert result["count"] == 2
    assert len(before) == len(after)

    for old, new in zip(before, after):
        np.testing.assert_array_equal(old, new)


def test_validation_is_reproducible():
    encoder, decoder, images, labels, seed = make_case()

    first = evaluate(encoder, decoder, images, labels, seed)
    second = evaluate(encoder, decoder, images, labels, seed)

    assert first == second


def test_validation_objective_is_finite_and_consistent():
    encoder, decoder, images, labels, seed = make_case()

    result = evaluate(encoder, decoder, images, labels, seed)

    for key in ("total_loss", "reconstruction_loss", "kl_loss"):
        assert np.isfinite(result[key])

    np.testing.assert_allclose(
        result["total_loss"],
        result["reconstruction_loss"] + result["kl_loss"],
        rtol=1e-5,
        atol=1e-4,
    )


def test_aggregation_weights_by_example_count():
    batches = [
        {
            "count": 2,
            "total_loss": 10.0,
            "reconstruction_loss": 7.0,
            "kl_loss": 3.0,
        },
        {
            "count": 3,
            "total_loss": 20.0,
            "reconstruction_loss": 16.0,
            "kl_loss": 4.0,
        },
    ]

    result = validation.aggregate_validation_batches(batches)

    assert result["count"] == 5
    np.testing.assert_allclose(result["total_loss"], 16.0)
    np.testing.assert_allclose(
        result["reconstruction_loss"], 12.4
    )
    np.testing.assert_allclose(result["kl_loss"], 3.6)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_invalid_images_rejected_without_state_change(value):
    encoder, decoder, _, labels, seed = make_case()

    images = tf.fill((2, 40, 40, 3), value)
    before = snapshot_all(encoder, decoder)

    with pytest.raises((ValueError, tf.errors.OpError)):
        evaluate(encoder, decoder, images, labels, seed)

    for old, new in zip(before, snapshot_all(encoder, decoder)):
        np.testing.assert_array_equal(old, new)


@pytest.mark.parametrize(
    "invalid_labels",
    [
        [[0.0] * 9, [0.0] * 9],
        [[1.0] * 9, [0.0] * 9],
        [[0.5, 0.5] + [0.0] * 7,
         [0.0, 1.0] + [0.0] * 7],
    ],
)
def test_invalid_labels_rejected_without_state_change(
    invalid_labels,
):
    encoder, decoder, images, _, seed = make_case()

    labels = tf.constant(invalid_labels, tf.float32)
    before = snapshot_all(encoder, decoder)

    with pytest.raises((ValueError, tf.errors.OpError)):
        evaluate(encoder, decoder, images, labels, seed)

    for old, new in zip(before, snapshot_all(encoder, decoder)):
        np.testing.assert_array_equal(old, new)


def test_invalid_seed_dtype_rejected():
    encoder, decoder, images, labels, _ = make_case()

    with pytest.raises(TypeError):
        evaluate(
            encoder,
            decoder,
            images,
            labels,
            tf.constant([42.0, 0.0], tf.float32),
        )


def test_invalid_seed_shape_rejected():
    encoder, decoder, images, labels, _ = make_case()

    with pytest.raises(tf.errors.OpError):
        evaluate(
            encoder,
            decoder,
            images,
            labels,
            tf.constant([42], tf.int32),
        )


def test_empty_aggregation_rejected():
    with pytest.raises(ValueError):
        validation.aggregate_validation_batches([])


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_aggregation_counts_rejected(count):
    batch = {
        "count": count,
        "total_loss": 3.0,
        "reconstruction_loss": 2.0,
        "kl_loss": 1.0,
    }

    with pytest.raises((TypeError, ValueError)):
        validation.aggregate_validation_batches([batch])


@pytest.mark.parametrize(
    "metric",
    ["total_loss", "reconstruction_loss", "kl_loss"],
)
def test_nonfinite_aggregation_metric_rejected(metric):
    batch = {
        "count": 2,
        "total_loss": 3.0,
        "reconstruction_loss": 2.0,
        "kl_loss": 1.0,
    }
    batch[metric] = float("nan")

    with pytest.raises(ValueError):
        validation.aggregate_validation_batches([batch])


def test_missing_aggregation_field_rejected():
    batch = {
        "count": 2,
        "total_loss": 3.0,
        "reconstruction_loss": 2.0,
    }

    with pytest.raises(ValueError):
        validation.aggregate_validation_batches([batch])
