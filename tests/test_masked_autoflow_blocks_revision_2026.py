import numpy as np
import tensorflow as tf

from trustforge.models.paper01_revision_2026.masked_autoflow_blocks import (
    BLOCK_DIM,
    CANONICAL_DIM,
    NUM_BLOCKS,
    blocks_to_images,
    images_to_blocks,
    make_block_permutation,
    permute_blocks,
)


def test_block_shape_contract():
    x = tf.zeros(
        (3, 40, 40, 3),
        dtype=tf.float32,
    )

    blocks = images_to_blocks(x)

    assert tuple(blocks.shape) == (
        3,
        25,
        192,
    )

    assert NUM_BLOCKS == 25
    assert BLOCK_DIM == 192
    assert CANONICAL_DIM == 4800


def test_image_block_round_trip_is_exact():
    x = tf.reshape(
        tf.range(
            2 * 40 * 40 * 3,
            dtype=tf.float32,
        ),
        (2, 40, 40, 3),
    )

    blocks = images_to_blocks(x)
    recovered = blocks_to_images(blocks)

    tf.debugging.assert_equal(
        recovered,
        x,
    )


def test_every_scalar_position_appears_once():
    x = np.arange(
        CANONICAL_DIM,
        dtype=np.float32,
    ).reshape(
        1,
        40,
        40,
        3,
    )

    blocks = images_to_blocks(
        tf.constant(x)
    ).numpy()

    flattened = blocks.reshape(-1)

    assert flattened.size == CANONICAL_DIM

    np.testing.assert_array_equal(
        np.sort(flattened),
        np.arange(
            CANONICAL_DIM,
            dtype=np.float32,
        ),
    )


def test_raster_block_order_is_deterministic():
    x = np.zeros(
        (1, 40, 40, 3),
        dtype=np.float32,
    )

    block_id = 0

    for block_row in range(5):
        for block_col in range(5):
            r0 = block_row * 8
            r1 = r0 + 8
            c0 = block_col * 8
            c1 = c0 + 8

            x[
                0,
                r0:r1,
                c0:c1,
                :,
            ] = float(block_id)

            block_id += 1

    blocks = images_to_blocks(
        tf.constant(x)
    ).numpy()[0]

    for index in range(NUM_BLOCKS):
        np.testing.assert_array_equal(
            blocks[index],
            np.full(
                (BLOCK_DIM,),
                float(index),
                dtype=np.float32,
            ),
        )


def test_seeded_block_permutation_is_reproducible():
    permutation_a, inverse_a = (
        make_block_permutation(42)
    )

    permutation_b, inverse_b = (
        make_block_permutation(42)
    )

    permutation_c, _ = (
        make_block_permutation(43)
    )

    np.testing.assert_array_equal(
        permutation_a,
        permutation_b,
    )

    np.testing.assert_array_equal(
        inverse_a,
        inverse_b,
    )

    assert not np.array_equal(
        permutation_a,
        permutation_c,
    )


def test_inverse_permutation_recovers_blocks():
    permutation, inverse = (
        make_block_permutation(42)
    )

    blocks = tf.reshape(
        tf.range(
            2 * NUM_BLOCKS * BLOCK_DIM,
            dtype=tf.float32,
        ),
        (
            2,
            NUM_BLOCKS,
            BLOCK_DIM,
        ),
    )

    permuted = permute_blocks(
        blocks,
        permutation,
    )

    recovered = permute_blocks(
        permuted,
        inverse,
    )

    tf.debugging.assert_equal(
        recovered,
        blocks,
    )
