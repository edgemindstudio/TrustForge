"""Regression tests for the governed Paper 1 revision-2026 PixelCNN."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from trustforge.models.paper01_revision_2026.autoregressive import (
    build_revision_pixelcnn,
)
from trustforge.models.paper01_revision_2026.autoregressive_observation import (
    coupled_rgb_censored_logistic_nll,
    sample_coupled_rgb_from_parameters,
    split_censored_logistic_parameters,
)
from trustforge.models.paper01_revision_2026.autoregressive_sampling import (
    sample_revision_pixelcnn,
)


def _small_model(
    *,
    img_shape=(4, 4, 3),
    num_classes=3,
):
    tf.random.set_seed(42)

    model = build_revision_pixelcnn(
        img_shape,
        num_classes,
        filters=8,
        masked_layers=1,
    )

    return model


def test_revision_pixelcnn_output_contract():
    model = _small_model()

    x = tf.zeros(
        (2, 4, 4, 3),
        dtype=tf.float32,
    )

    y = tf.one_hot(
        [0, 1],
        depth=3,
        dtype=tf.float32,
    )

    params = model(
        [x, y],
        training=False,
    )

    assert tuple(params.shape) == (2, 4, 4, 9)
    assert params.dtype == tf.float32

    assert bool(
        tf.reduce_all(
            tf.math.is_finite(
                params
            )
        )
    )

    base_mu, scale, alpha = (
        split_censored_logistic_parameters(
            params
        )
    )

    assert tuple(base_mu.shape) == (2, 4, 4, 3)
    assert tuple(scale.shape) == (2, 4, 4, 3)
    assert tuple(alpha.shape) == (2, 4, 4, 3)

    assert bool(
        tf.reduce_all(
            scale > 0.0
        )
    )

    assert bool(
        tf.reduce_all(
            tf.math.abs(
                alpha
            ) <= 1.0
        )
    )


def test_revision_pixelcnn_observation_primitives():
    model = _small_model()

    x = tf.constant(
        np.linspace(
            0.0,
            1.0,
            num=2 * 4 * 4 * 3,
            dtype=np.float32,
        ).reshape(
            2,
            4,
            4,
            3,
        )
    )

    y = tf.one_hot(
        [0, 2],
        depth=3,
        dtype=tf.float32,
    )

    params = model(
        [x, y],
        training=False,
    )

    nll = coupled_rgb_censored_logistic_nll(
        x,
        params,
    )

    assert bool(
        tf.math.is_finite(
            nll
        )
    )

    sample = sample_coupled_rgb_from_parameters(
        params,
        temperature=1.0,
        seed=42,
    )

    assert tuple(sample.shape) == (2, 4, 4, 3)
    assert sample.dtype == tf.float32

    assert bool(
        tf.reduce_all(
            tf.math.is_finite(
                sample
            )
        )
    )

    assert float(
        tf.reduce_min(
            sample
        )
    ) >= 0.0

    assert float(
        tf.reduce_max(
            sample
        )
    ) <= 1.0

    deterministic_a = sample_coupled_rgb_from_parameters(
        params,
        temperature=0.0,
        seed=42,
    )

    deterministic_b = sample_coupled_rgb_from_parameters(
        params,
        temperature=0.0,
        seed=999,
    )

    tf.debugging.assert_equal(
        deterministic_a,
        deterministic_b,
    )


def test_revision_pixelcnn_spatial_causality():
    model = _small_model()

    label = tf.one_hot(
        [1],
        depth=3,
        dtype=tf.float32,
    )

    base = np.zeros(
        (1, 4, 4, 3),
        dtype=np.float32,
    )

    future = base.copy()
    future[0, 3, 3, :] = 1.0

    current = base.copy()
    current[0, 1, 1, :] = 1.0

    previous = base.copy()
    previous[0, 1, 0, :] = 1.0

    p_base = model(
        [tf.constant(base), label],
        training=False,
    )

    p_future = model(
        [tf.constant(future), label],
        training=False,
    )

    p_current = model(
        [tf.constant(current), label],
        training=False,
    )

    p_previous = model(
        [tf.constant(previous), label],
        training=False,
    )

    location = (1, 1)

    base_pred = p_base[
        :,
        location[0],
        location[1],
        :,
    ]

    future_pred = p_future[
        :,
        location[0],
        location[1],
        :,
    ]

    current_pred = p_current[
        :,
        location[0],
        location[1],
        :,
    ]

    previous_pred = p_previous[
        :,
        location[0],
        location[1],
        :,
    ]

    tf.debugging.assert_near(
        base_pred,
        future_pred,
        atol=1e-7,
    )

    tf.debugging.assert_near(
        base_pred,
        current_pred,
        atol=1e-7,
    )

    delta = float(
        tf.reduce_max(
            tf.abs(
                base_pred
                - previous_pred
            )
        )
    )

    assert delta > 1e-9


def test_revision_pixelcnn_first_pixel_class_conditioning():
    model = _small_model()

    blank = tf.zeros(
        (1, 4, 4, 3),
        dtype=tf.float32,
    )

    y0 = tf.one_hot(
        [0],
        depth=3,
        dtype=tf.float32,
    )

    y1 = tf.one_hot(
        [1],
        depth=3,
        dtype=tf.float32,
    )

    p0 = model(
        [blank, y0],
        training=False,
    )[:, 0, 0, :]

    p1 = model(
        [blank, y1],
        training=False,
    )[:, 0, 0, :]

    delta = float(
        tf.reduce_max(
            tf.abs(p0 - p1)
        )
    )

    assert delta > 1e-9


def test_revision_pixelcnn_raster_sampler_reproducibility():
    model = _small_model()

    class_ids = np.array(
        [0, 2],
        dtype=np.int32,
    )

    a = sample_revision_pixelcnn(
        model,
        class_ids=class_ids,
        img_shape=(4, 4, 3),
        num_classes=3,
        temperature=1.0,
        seed=42,
    )

    b = sample_revision_pixelcnn(
        model,
        class_ids=class_ids,
        img_shape=(4, 4, 3),
        num_classes=3,
        temperature=1.0,
        seed=42,
    )

    c = sample_revision_pixelcnn(
        model,
        class_ids=class_ids,
        img_shape=(4, 4, 3),
        num_classes=3,
        temperature=1.0,
        seed=43,
    )

    assert a.shape == (2, 4, 4, 3)
    assert a.dtype == np.float32
    assert np.isfinite(a).all()
    assert a.min() >= 0.0
    assert a.max() <= 1.0

    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)

    t0_a = sample_revision_pixelcnn(
        model,
        class_ids=class_ids,
        img_shape=(4, 4, 3),
        num_classes=3,
        temperature=0.0,
        seed=42,
    )

    t0_b = sample_revision_pixelcnn(
        model,
        class_ids=class_ids,
        img_shape=(4, 4, 3),
        num_classes=3,
        temperature=0.0,
        seed=999,
    )

    assert np.array_equal(
        t0_a,
        t0_b,
    )


def test_revision_pixelcnn_rejects_invalid_class():
    model = _small_model()

    try:
        sample_revision_pixelcnn(
            model,
            class_ids=np.array(
                [3],
                dtype=np.int32,
            ),
            img_shape=(4, 4, 3),
            num_classes=3,
            seed=42,
        )
    except ValueError:
        return

    raise AssertionError(
        "out-of-range class identifier was not rejected"
    )


def test_revision_pixelcnn_canonical_40x40x3_compatibility():
    model = build_revision_pixelcnn(
        (40, 40, 3),
        num_classes=3,
        filters=8,
        masked_layers=1,
    )

    x = tf.zeros(
        (
            1,
            40,
            40,
            3,
        ),
        dtype=tf.float32,
    )

    y = tf.one_hot(
        [1],
        depth=3,
        dtype=tf.float32,
    )

    params = model(
        [x, y],
        training=False,
    )

    assert tuple(params.shape) == (
        1,
        40,
        40,
        9,
    )

    assert params.dtype == tf.float32

    assert bool(
        tf.reduce_all(
            tf.math.is_finite(
                params
            )
        )
    )


def test_revision_pixelcnn_training_step_has_finite_update():
    from trustforge.models.paper01_revision_2026.autoregressive_training import (
        AutoregressiveTrainingConfig,
        train_step,
    )

    tf.random.set_seed(
        42
    )

    model = _small_model()

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-3
    )

    config = AutoregressiveTrainingConfig(
        epochs=1,
        patience=1,
        save_every=1,
        min_log_scale=-7.0,
        max_log_scale=3.0,
    )

    x = tf.constant(
        np.linspace(
            0.0,
            1.0,
            num=2 * 4 * 4 * 3,
            dtype=np.float32,
        ).reshape(
            2,
            4,
            4,
            3,
        )
    )

    y = tf.one_hot(
        [0, 2],
        depth=3,
        dtype=tf.float32,
    )

    before = [
        variable.numpy().copy()
        for variable in model.trainable_variables
    ]

    loss = train_step(
        model,
        optimizer,
        x,
        y,
        config=config,
    )

    after = [
        variable.numpy().copy()
        for variable in model.trainable_variables
    ]

    assert bool(
        tf.math.is_finite(
            loss
        )
    )

    assert any(
        not np.array_equal(
            old,
            new,
        )
        for old, new in zip(
            before,
            after,
        )
    )

    for variable in model.trainable_variables:
        assert np.isfinite(
            variable.numpy()
        ).all()
