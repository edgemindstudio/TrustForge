import numpy as np
import tensorflow as tf

from trustforge.models.paper01_revision_2026.masked_autoflow_model import (
    ConditionalBlockMAF,
)
from trustforge.models.paper01_revision_2026.masked_autoflow_training import (
    conditional_nll,
    train_step,
)


def _model():
    tf.keras.utils.set_random_seed(
        42
    )

    model = ConditionalBlockMAF(
        num_classes=9,
    )

    # Deterministically build trainable variables.
    images = tf.fill(
        (1, 40, 40, 3),
        tf.constant(
            0.5,
            dtype=tf.float32,
        ),
    )

    labels = tf.constant(
        [0],
        dtype=tf.int32,
    )

    model.log_prob(
        images,
        labels,
        training=False,
    )

    return model


def _training_batch():
    images = tf.random.stateless_uniform(
        (1, 40, 40, 3),
        seed=(600, 601),
        minval=0.05,
        maxval=0.95,
        dtype=tf.float32,
    )

    labels = tf.constant(
        [3],
        dtype=tf.int32,
    )

    return images, labels


def test_conditional_nll_is_finite():
    model = _model()

    images, labels = (
        _training_batch()
    )

    loss = conditional_nll(
        model,
        images,
        labels,
        training=False,
    )

    assert tuple(
        loss.shape
    ) == ()

    assert bool(
        tf.math.is_finite(
            loss
        ).numpy()
    )


def test_one_training_step_has_finite_loss_and_gradients():
    model = _model()

    images, labels = (
        _training_batch()
    )

    # Smoke-test optimizer only.
    # This is not the final governed experiment optimizer configuration.
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-4,
    )

    metrics = train_step(
        model,
        optimizer,
        images,
        labels,
    )

    assert bool(
        tf.math.is_finite(
            metrics["loss"]
        ).numpy()
    )

    assert bool(
        tf.math.is_finite(
            metrics[
                "global_gradient_norm"
            ]
        ).numpy()
    )

    assert float(
        metrics[
            "global_gradient_norm"
        ].numpy()
    ) > 0.0

    assert int(
        metrics[
            "trainable_variable_count"
        ].numpy()
    ) > 0


def test_optimizer_step_updates_trainable_parameters():
    model = _model()

    images, labels = (
        _training_batch()
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-4,
    )

    before = [
        variable.numpy().copy()
        for variable
        in model.trainable_variables
    ]

    train_step(
        model,
        optimizer,
        images,
        labels,
    )

    after = [
        variable.numpy().copy()
        for variable
        in model.trainable_variables
    ]

    assert len(before) == len(after)
    assert len(before) > 0

    changed = [
        not np.array_equal(
            before_value,
            after_value,
        )
        for before_value, after_value
        in zip(
            before,
            after,
        )
    ]

    assert any(changed)


def test_training_step_does_not_mutate_canonical_inputs():
    model = _model()

    images, labels = (
        _training_batch()
    )

    original_images = (
        images.numpy().copy()
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-4,
    )

    train_step(
        model,
        optimizer,
        images,
        labels,
    )

    np.testing.assert_array_equal(
        images.numpy(),
        original_images,
    )
