import numpy as np
import tensorflow as tf

from trustforge.models.paper01_revision_2026.masked_autoflow_blocks import (
    BLOCK_DIM,
    NUM_BLOCKS,
)
from trustforge.models.paper01_revision_2026.masked_autoflow_flow import (
    MAX_INVERSE_BLOCK_STAGES,
    BlockMAFFlowStage,
)


def _inputs(
    batch=2,
):
    blocks = tf.random.stateless_uniform(
        (
            batch,
            NUM_BLOCKS,
            BLOCK_DIM,
        ),
        seed=(100, 101),
        minval=0.0,
        maxval=1.0,
        dtype=tf.float32,
    )

    labels = tf.range(
        batch,
        dtype=tf.int32,
    )

    return blocks, labels


def _stage(
    *,
    permutation_seed=None,
):
    tf.keras.utils.set_random_seed(
        42
    )

    stage = BlockMAFFlowStage(
        num_classes=9,
        permutation_seed=permutation_seed,
    )

    blocks, labels = _inputs()

    # Build deterministically.
    stage.forward(
        blocks,
        labels,
        training=False,
    )

    return stage


def test_forward_shape_range_and_logdet_contract():
    stage = _stage()

    blocks, labels = _inputs(
        batch=3
    )

    transformed, log_det = (
        stage.forward(
            blocks,
            labels,
            training=False,
        )
    )

    assert tuple(
        transformed.shape
    ) == (
        3,
        NUM_BLOCKS,
        BLOCK_DIM,
    )

    assert tuple(
        log_det.shape
    ) == (3,)

    assert float(
        tf.reduce_min(
            transformed
        ).numpy()
    ) >= 0.0

    assert float(
        tf.reduce_max(
            transformed
        ).numpy()
    ) <= 1.0

    assert bool(
        tf.reduce_all(
            tf.math.is_finite(
                transformed
            )
        ).numpy()
    )

    assert bool(
        tf.reduce_all(
            tf.math.is_finite(
                log_det
            )
        ).numpy()
    )


def test_single_stage_forward_inverse_round_trip():
    stage = _stage()

    blocks, labels = _inputs()

    transformed, forward_log_det = (
        stage.forward(
            blocks,
            labels,
            training=False,
        )
    )

    recovered, inverse_log_det = (
        stage.inverse(
            transformed,
            labels,
            training=False,
        )
    )

    max_error = float(
        tf.reduce_max(
            tf.abs(
                recovered
                - blocks
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

    assert max_error < 5e-5
    assert logdet_error < 5e-3


def test_permuted_stage_round_trip_and_inverse_permutation():
    stage = _stage(
        permutation_seed=123
    )

    blocks, labels = _inputs()

    transformed, forward_log_det = (
        stage.forward(
            blocks,
            labels,
            training=False,
        )
    )

    recovered, inverse_log_det = (
        stage.inverse(
            transformed,
            labels,
            training=False,
        )
    )

    np.testing.assert_allclose(
        recovered.numpy(),
        blocks.numpy(),
        atol=5e-5,
        rtol=0.0,
    )

    np.testing.assert_allclose(
        (
            forward_log_det
            + inverse_log_det
        ).numpy(),
        0.0,
        atol=5e-3,
        rtol=0.0,
    )


def test_inverse_uses_exactly_25_block_stages():
    stage = _stage(
        permutation_seed=7
    )

    blocks, labels = _inputs(
        batch=1
    )

    transformed, _ = stage.forward(
        blocks,
        labels,
        training=False,
    )

    (
        recovered,
        _,
        diagnostics,
    ) = stage.inverse(
        transformed,
        labels,
        training=False,
        return_diagnostics=True,
    )

    assert tuple(
        recovered.shape
    ) == (
        1,
        NUM_BLOCKS,
        BLOCK_DIM,
    )

    assert (
        diagnostics[
            "inverse_block_stages"
        ]
        == 25
    )

    assert (
        diagnostics[
            "maximum_inverse_block_stages"
        ]
        == 25
    )

    assert (
        MAX_INVERSE_BLOCK_STAGES
        == 25
    )


def test_class_condition_participates_in_actual_flow():
    stage = _stage()

    blocks = tf.random.stateless_uniform(
        (
            1,
            NUM_BLOCKS,
            BLOCK_DIM,
        ),
        seed=(200, 201),
        minval=0.0,
        maxval=1.0,
        dtype=tf.float32,
    )

    transformed_a, log_det_a = (
        stage.forward(
            blocks,
            tf.constant(
                [1],
                dtype=tf.int32,
            ),
            training=False,
        )
    )

    transformed_b, log_det_b = (
        stage.forward(
            blocks,
            tf.constant(
                [7],
                dtype=tf.int32,
            ),
            training=False,
        )
    )

    transformed_delta = float(
        tf.reduce_max(
            tf.abs(
                transformed_a
                - transformed_b
            )
        ).numpy()
    )

    logdet_delta = float(
        tf.reduce_max(
            tf.abs(
                log_det_a
                - log_det_b
            )
        ).numpy()
    )

    assert (
        transformed_delta > 0.0
        or logdet_delta > 0.0
    )


def test_exact_boundaries_survive_full_flow_round_trip():
    stage = _stage(
        permutation_seed=42
    )

    blocks = tf.zeros(
        (
            1,
            NUM_BLOCKS,
            BLOCK_DIM,
        ),
        dtype=tf.float32,
    )

    # Mix exact zero and exact one throughout the block tensor.
    pattern = tf.cast(
        tf.math.mod(
            tf.range(
                NUM_BLOCKS * BLOCK_DIM
            ),
            2,
        ),
        tf.float32,
    )

    blocks = tf.reshape(
        pattern,
        (
            1,
            NUM_BLOCKS,
            BLOCK_DIM,
        ),
    )

    labels = tf.constant(
        [3],
        dtype=tf.int32,
    )

    transformed, _ = stage.forward(
        blocks,
        labels,
        training=False,
    )

    recovered, _ = stage.inverse(
        transformed,
        labels,
        training=False,
    )

    np.testing.assert_allclose(
        recovered.numpy(),
        blocks.numpy(),
        atol=5e-5,
        rtol=0.0,
    )

    assert float(
        tf.reduce_min(
            transformed
        ).numpy()
    ) >= 0.0

    assert float(
        tf.reduce_max(
            transformed
        ).numpy()
    ) <= 1.0
