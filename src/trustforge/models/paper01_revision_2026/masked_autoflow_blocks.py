"""Block-layout primitives for the Paper 1 revision-2026 Masked AutoFlow.

The governed canonical representation is ``40 x 40 x 3`` and is partitioned
into a raster-ordered ``5 x 5`` grid of non-overlapping ``8 x 8 x 3`` blocks.

The model-facing block representation is therefore::

    (B, 40, 40, 3) <-> (B, 25, 192)

Only whole blocks may be permuted. Scalar positions inside a block are never
permuted by this module.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf


IMAGE_HEIGHT = 40
IMAGE_WIDTH = 40
IMAGE_CHANNELS = 3

BLOCK_HEIGHT = 8
BLOCK_WIDTH = 8

BLOCK_ROWS = IMAGE_HEIGHT // BLOCK_HEIGHT
BLOCK_COLS = IMAGE_WIDTH // BLOCK_WIDTH

NUM_BLOCKS = BLOCK_ROWS * BLOCK_COLS
BLOCK_DIM = BLOCK_HEIGHT * BLOCK_WIDTH * IMAGE_CHANNELS
CANONICAL_DIM = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS

IMAGE_SHAPE: Tuple[int, int, int] = (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    IMAGE_CHANNELS,
)


def images_to_blocks(images: tf.Tensor) -> tf.Tensor:
    """Convert canonical images to raster-ordered block vectors.

    Parameters
    ----------
    images:
        Tensor with shape ``(B, 40, 40, 3)``.

    Returns
    -------
    tf.Tensor
        Tensor with shape ``(B, 25, 192)``.

    Notes
    -----
    Block order is raster order over the 5 x 5 spatial block grid.
    Scalar order inside each block is row-major HWC order.
    """
    x = tf.convert_to_tensor(
        images,
        dtype=tf.float32,
    )

    tf.debugging.assert_rank(
        x,
        4,
        message="images must have rank 4",
    )

    tf.debugging.assert_equal(
        tf.shape(x)[1:],
        tf.constant(
            IMAGE_SHAPE,
            dtype=tf.int32,
        ),
        message="images must have shape (B,40,40,3)",
    )

    batch = tf.shape(x)[0]

    # (B, 5, 8, 5, 8, 3)
    x = tf.reshape(
        x,
        (
            batch,
            BLOCK_ROWS,
            BLOCK_HEIGHT,
            BLOCK_COLS,
            BLOCK_WIDTH,
            IMAGE_CHANNELS,
        ),
    )

    # Move block-grid axes together:
    # (B, block_row, block_col, local_row, local_col, channel)
    x = tf.transpose(
        x,
        perm=(0, 1, 3, 2, 4, 5),
    )

    return tf.reshape(
        x,
        (
            batch,
            NUM_BLOCKS,
            BLOCK_DIM,
        ),
    )


def blocks_to_images(blocks: tf.Tensor) -> tf.Tensor:
    """Invert :func:`images_to_blocks` exactly."""
    x = tf.convert_to_tensor(
        blocks,
        dtype=tf.float32,
    )

    tf.debugging.assert_rank(
        x,
        3,
        message="blocks must have rank 3",
    )

    tf.debugging.assert_equal(
        tf.shape(x)[1:],
        tf.constant(
            (
                NUM_BLOCKS,
                BLOCK_DIM,
            ),
            dtype=tf.int32,
        ),
        message="blocks must have shape (B,25,192)",
    )

    batch = tf.shape(x)[0]

    # (B, block_row, block_col, local_row, local_col, channel)
    x = tf.reshape(
        x,
        (
            batch,
            BLOCK_ROWS,
            BLOCK_COLS,
            BLOCK_HEIGHT,
            BLOCK_WIDTH,
            IMAGE_CHANNELS,
        ),
    )

    # Restore image-axis ordering:
    # (B, block_row, local_row, block_col, local_col, channel)
    x = tf.transpose(
        x,
        perm=(0, 1, 3, 2, 4, 5),
    )

    return tf.reshape(
        x,
        (
            batch,
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            IMAGE_CHANNELS,
        ),
    )


def make_block_permutation(
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a deterministic whole-block permutation and its inverse."""
    rng = np.random.default_rng(
        int(seed)
    )

    permutation = rng.permutation(
        NUM_BLOCKS
    ).astype(
        np.int32
    )

    inverse = np.argsort(
        permutation
    ).astype(
        np.int32
    )

    return permutation, inverse


def permute_blocks(
    blocks: tf.Tensor,
    permutation,
) -> tf.Tensor:
    """Apply a whole-block permutation along the block axis."""
    x = tf.convert_to_tensor(
        blocks,
        dtype=tf.float32,
    )

    permutation = tf.convert_to_tensor(
        permutation,
        dtype=tf.int32,
    )

    tf.debugging.assert_equal(
        tf.shape(permutation),
        tf.constant(
            (NUM_BLOCKS,),
            dtype=tf.int32,
        ),
        message="permutation must contain exactly 25 block indices",
    )

    tf.debugging.assert_equal(
        tf.sort(permutation),
        tf.range(
            NUM_BLOCKS,
            dtype=tf.int32,
        ),
        message="permutation must contain every block index exactly once",
    )

    return tf.gather(
        x,
        permutation,
        axis=1,
    )


__all__ = [
    "BLOCK_COLS",
    "BLOCK_DIM",
    "BLOCK_HEIGHT",
    "BLOCK_ROWS",
    "BLOCK_WIDTH",
    "CANONICAL_DIM",
    "IMAGE_CHANNELS",
    "IMAGE_HEIGHT",
    "IMAGE_SHAPE",
    "IMAGE_WIDTH",
    "NUM_BLOCKS",
    "blocks_to_images",
    "images_to_blocks",
    "make_block_permutation",
    "permute_blocks",
]
