import numpy as np
import tensorflow as tf

from trustforge.models.paper01_revision_2026.masked_autoflow_blocks import (
    BLOCK_DIM,
    NUM_BLOCKS,
    blocks_to_images,
)
from trustforge.models.paper01_revision_2026.masked_autoflow_model import (
    DEFAULT_PERMUTATION_SEEDS,
    MAX_TOTAL_INVERSE_BLOCK_STAGES,
    NUM_FLOWS,
    ConditionalBlockMAF,
)


def _model():
    tf.keras.utils.set_random_seed(
        42
    )

    model = ConditionalBlockMAF(
        num_classes=9,
    )

    # Deterministically build all five stages.
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


def test_five_flow_configuration_and_permutations():
    model = _model()

    assert NUM_FLOWS == 5
    assert len(
        model.flow_stages
    ) == 5

    assert tuple(
        model.permutation_seeds
    ) == DEFAULT_PERMUTATION_SEEDS

    permutations = [
        tuple(
            stage.permutation.numpy().tolist()
        )
        for stage in model.flow_stages
    ]

    assert len(
        set(permutations)
    ) == 5


def test_conditional_log_prob_is_finite_and_class_sensitive():
    model = _model()

    images = tf.random.stateless_uniform(
        (1, 40, 40, 3),
        seed=(300, 301),
        minval=0.05,
        maxval=0.95,
        dtype=tf.float32,
    )

    log_prob_a = model.log_prob(
        images,
        tf.constant(
            [1],
            dtype=tf.int32,
        ),
        training=False,
    )

    log_prob_b = model.log_prob(
        images,
        tf.constant(
            [7],
            dtype=tf.int32,
        ),
        training=False,
    )

    assert tuple(
        log_prob_a.shape
    ) == (1,)

    assert tuple(
        log_prob_b.shape
    ) == (1,)

    assert bool(
        tf.reduce_all(
            tf.math.is_finite(
                log_prob_a
            )
        ).numpy()
    )

    assert bool(
        tf.reduce_all(
            tf.math.is_finite(
                log_prob_b
            )
        ).numpy()
    )

    delta = float(
        tf.reduce_max(
            tf.abs(
                log_prob_a
                - log_prob_b
            )
        ).numpy()
    )

    assert delta > 0.0


def test_five_flow_forward_inverse_round_trip_and_125_stage_bound():
    model = _model()

    latent_seed_blocks = (
        tf.random.stateless_uniform(
            (
                1,
                NUM_BLOCKS,
                BLOCK_DIM,
            ),
            seed=(400, 401),
            minval=0.05,
            maxval=0.95,
            dtype=tf.float32,
        )
    )

    labels = tf.constant(
        [3],
        dtype=tf.int32,
    )

    (
        transformed,
        forward_log_det,
    ) = model.forward_blocks(
        latent_seed_blocks,
        labels,
        training=False,
    )

    (
        recovered,
        inverse_log_det,
        diagnostics,
    ) = model.inverse_blocks(
        transformed,
        labels,
        training=False,
        return_diagnostics=True,
    )

    max_error = float(
        tf.reduce_max(
            tf.abs(
                recovered
                - latent_seed_blocks
            )
        ).numpy()
    )

    logdet_error = float(
        tf.reduce_max(
            tf.abs(
                forward_log_det
                + inverse_log_det
            )
        ).numpy()
    )

    assert max_error < 2e-4
    assert logdet_error < 2e-2

    assert (
        diagnostics[
            "num_flows"
        ]
        == 5
    )

    assert (
        diagnostics[
            "inverse_block_stages"
        ]
        == 125
    )

    assert (
        diagnostics[
            "maximum_inverse_block_stages"
        ]
        == 125
    )

    assert (
        MAX_TOTAL_INVERSE_BLOCK_STAGES
        == 125
    )


def test_sampling_is_seed_reproducible_and_bounded():
    model = _model()

    labels = tf.constant(
        [0, 4],
        dtype=tf.int32,
    )

    samples_a = model.sample(
        labels,
        seed=42,
    )

    samples_b = model.sample(
        labels,
        seed=42,
    )

    samples_c = model.sample(
        labels,
        seed=43,
    )

    assert tuple(
        samples_a.shape
    ) == (
        2,
        40,
        40,
        3,
    )

    np.testing.assert_array_equal(
        samples_a.numpy(),
        samples_b.numpy(),
    )

    assert not np.array_equal(
        samples_a.numpy(),
        samples_c.numpy(),
    )

    assert bool(
        tf.reduce_all(
            tf.math.is_finite(
                samples_a
            )
        ).numpy()
    )

    assert float(
        tf.reduce_min(
            samples_a
        ).numpy()
    ) >= 0.0

    assert float(
        tf.reduce_max(
            samples_a
        ).numpy()
    ) <= 1.0


def test_image_api_preserves_exact_canonical_shape():
    model = _model()

    blocks = tf.random.stateless_uniform(
        (
            1,
            NUM_BLOCKS,
            BLOCK_DIM,
        ),
        seed=(500, 501),
        minval=0.0,
        maxval=1.0,
        dtype=tf.float32,
    )

    images = blocks_to_images(
        blocks
    )

    labels = tf.constant(
        [2],
        dtype=tf.int32,
    )

    log_prob = model.log_prob(
        images,
        labels,
        training=False,
    )

    assert tuple(
        images.shape
    ) == (
        1,
        40,
        40,
        3,
    )

    assert tuple(
        log_prob.shape
    ) == (1,)

    assert bool(
        tf.reduce_all(
            tf.math.is_finite(
                log_prob
            )
        ).numpy()
    )
