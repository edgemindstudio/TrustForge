import numpy as np
import tensorflow as tf

from trustforge.models.paper01_revision_2026.masked_autoflow_blocks import (
    BLOCK_DIM,
    NUM_BLOCKS,
)
from trustforge.models.paper01_revision_2026.masked_autoflow_conditioner import (
    DECODER_BOTTLENECK,
    DECODER_OUTPUT_DIM,
    BlockConditionalSplineConditioner,
    strict_block_dependency_mask,
)
from trustforge.models.paper01_revision_2026.masked_autoflow_spline import (
    NUM_BINS,
)


def _model():
    tf.keras.utils.set_random_seed(
        42
    )

    model = BlockConditionalSplineConditioner(
        num_classes=9,
    )

    # Build all trainable layers deterministically.
    blocks = tf.zeros(
        (
            2,
            NUM_BLOCKS,
            BLOCK_DIM,
        ),
        dtype=tf.float32,
    )

    labels = tf.constant(
        [0, 1],
        dtype=tf.int32,
    )

    model(
        blocks,
        labels,
        training=False,
    )

    return model


def test_strict_dependency_mask_has_exact_structure():
    mask = strict_block_dependency_mask().numpy()

    assert mask.shape == (
        NUM_BLOCKS,
        NUM_BLOCKS,
    )

    for i in range(NUM_BLOCKS):
        for j in range(NUM_BLOCKS):
            expected = (
                1.0
                if j < i
                else 0.0
            )

            assert mask[i, j] == expected


def test_conditioner_output_shapes_and_shared_decoder_contract():
    model = _model()

    blocks = tf.zeros(
        (
            3,
            NUM_BLOCKS,
            BLOCK_DIM,
        ),
        dtype=tf.float32,
    )

    labels = tf.constant(
        [0, 1, 2],
        dtype=tf.int32,
    )

    raw_widths, raw_slope_controls = model(
        blocks,
        labels,
        training=False,
    )

    assert tuple(
        raw_widths.shape
    ) == (
        3,
        NUM_BLOCKS,
        BLOCK_DIM,
        NUM_BINS,
    )

    assert tuple(
        raw_slope_controls.shape
    ) == (
        3,
        NUM_BLOCKS,
        BLOCK_DIM,
        NUM_BINS,
    )

    assert (
        model.decoder_bottleneck_layer.units
        == DECODER_BOTTLENECK
    )

    assert (
        model.decoder_output_layer.units
        == DECODER_OUTPUT_DIM
    )


def test_current_block_cannot_influence_its_own_parameters():
    model = _model()

    baseline = tf.zeros(
        (
            1,
            NUM_BLOCKS,
            BLOCK_DIM,
        ),
        dtype=tf.float32,
    )

    labels = tf.constant(
        [3],
        dtype=tf.int32,
    )

    target_block = 12

    changed = tf.tensor_scatter_nd_update(
        baseline,
        indices=[
            [0, target_block, 0],
            [0, target_block, 50],
            [0, target_block, 191],
        ],
        updates=[
            0.25,
            0.50,
            0.75,
        ],
    )

    widths_a, controls_a = model(
        baseline,
        labels,
        training=False,
    )

    widths_b, controls_b = model(
        changed,
        labels,
        training=False,
    )

    np.testing.assert_array_equal(
        widths_a[
            :,
            target_block,
            :,
            :,
        ].numpy(),
        widths_b[
            :,
            target_block,
            :,
            :,
        ].numpy(),
    )

    np.testing.assert_array_equal(
        controls_a[
            :,
            target_block,
            :,
            :,
        ].numpy(),
        controls_b[
            :,
            target_block,
            :,
            :,
        ].numpy(),
    )


def test_future_blocks_cannot_influence_earlier_parameters():
    model = _model()

    baseline = tf.zeros(
        (
            1,
            NUM_BLOCKS,
            BLOCK_DIM,
        ),
        dtype=tf.float32,
    )

    labels = tf.constant(
        [4],
        dtype=tf.int32,
    )

    earlier_block = 5
    future_block = 20

    changed = tf.tensor_scatter_nd_update(
        baseline,
        indices=[
            [0, future_block, 0],
            [0, future_block, 100],
            [0, future_block, 191],
        ],
        updates=[
            1.0,
            0.5,
            0.25,
        ],
    )

    widths_a, controls_a = model(
        baseline,
        labels,
        training=False,
    )

    widths_b, controls_b = model(
        changed,
        labels,
        training=False,
    )

    np.testing.assert_array_equal(
        widths_a[
            :,
            earlier_block,
            :,
            :,
        ].numpy(),
        widths_b[
            :,
            earlier_block,
            :,
            :,
        ].numpy(),
    )

    np.testing.assert_array_equal(
        controls_a[
            :,
            earlier_block,
            :,
            :,
        ].numpy(),
        controls_b[
            :,
            earlier_block,
            :,
            :,
        ].numpy(),
    )


def test_earlier_block_can_influence_later_parameters():
    model = _model()

    baseline = tf.zeros(
        (
            1,
            NUM_BLOCKS,
            BLOCK_DIM,
        ),
        dtype=tf.float32,
    )

    labels = tf.constant(
        [5],
        dtype=tf.int32,
    )

    earlier_block = 2
    later_block = 18

    changed = tf.tensor_scatter_nd_update(
        baseline,
        indices=[
            [0, earlier_block, 0],
            [0, earlier_block, 80],
            [0, earlier_block, 191],
        ],
        updates=[
            1.0,
            0.5,
            0.25,
        ],
    )

    widths_a, controls_a = model(
        baseline,
        labels,
        training=False,
    )

    widths_b, controls_b = model(
        changed,
        labels,
        training=False,
    )

    width_delta = float(
        tf.reduce_max(
            tf.abs(
                widths_a[
                    :,
                    later_block,
                    :,
                    :,
                ]
                - widths_b[
                    :,
                    later_block,
                    :,
                    :,
                ]
            )
        ).numpy()
    )

    control_delta = float(
        tf.reduce_max(
            tf.abs(
                controls_a[
                    :,
                    later_block,
                    :,
                    :,
                ]
                - controls_b[
                    :,
                    later_block,
                    :,
                    :,
                ]
            )
        ).numpy()
    )

    assert (
        width_delta > 0.0
        or control_delta > 0.0
    )


def test_class_condition_changes_parameters():
    model = _model()

    blocks = tf.zeros(
        (
            1,
            NUM_BLOCKS,
            BLOCK_DIM,
        ),
        dtype=tf.float32,
    )

    widths_a, controls_a = model(
        blocks,
        tf.constant(
            [1],
            dtype=tf.int32,
        ),
        training=False,
    )

    widths_b, controls_b = model(
        blocks,
        tf.constant(
            [7],
            dtype=tf.int32,
        ),
        training=False,
    )

    width_delta = float(
        tf.reduce_max(
            tf.abs(
                widths_a
                - widths_b
            )
        ).numpy()
    )

    control_delta = float(
        tf.reduce_max(
            tf.abs(
                controls_a
                - controls_b
            )
        ).numpy()
    )

    assert (
        width_delta > 0.0
        or control_delta > 0.0
    )


def test_shared_decoder_can_emit_different_parameters_by_block():
    model = _model()

    blocks = tf.zeros(
        (
            1,
            NUM_BLOCKS,
            BLOCK_DIM,
        ),
        dtype=tf.float32,
    )

    labels = tf.constant(
        [2],
        dtype=tf.int32,
    )

    raw_widths, raw_slope_controls = model(
        blocks,
        labels,
        training=False,
    )

    first_widths = raw_widths[
        0,
        0,
    ]

    later_widths = raw_widths[
        0,
        10,
    ]

    first_controls = raw_slope_controls[
        0,
        0,
    ]

    later_controls = raw_slope_controls[
        0,
        10,
    ]

    width_delta = float(
        tf.reduce_max(
            tf.abs(
                first_widths
                - later_widths
            )
        ).numpy()
    )

    control_delta = float(
        tf.reduce_max(
            tf.abs(
                first_controls
                - later_controls
            )
        ).numpy()
    )

    assert (
        width_delta > 0.0
        or control_delta > 0.0
    )
