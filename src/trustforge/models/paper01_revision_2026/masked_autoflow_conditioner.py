"""Strict block-autoregressive conditioner for Paper 1 revision-2026 MAF.

For block ``i``, predicted transform parameters may depend only on:

- blocks strictly earlier than ``i`` in the current governed ordering;
- the requested class condition;
- the current block position.

They must not depend on the current block or any future block.

The conditioner emits one independent eight-bin width parameter vector and one
independent eight-bin relative-slope control vector for every scalar in every block.
"""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf

from trustforge.models.paper01_revision_2026.masked_autoflow_blocks import (
    BLOCK_DIM,
    NUM_BLOCKS,
)
from trustforge.models.paper01_revision_2026.masked_autoflow_spline import (
    NUM_BINS,
)


HIDDEN_WIDTH = 256
HIDDEN_DEPTH = 2
CLASS_EMBED_DIM = 64
DECODER_BOTTLENECK = 64

VALUES_PER_SCALAR = 2 * NUM_BINS
DECODER_OUTPUT_DIM = BLOCK_DIM * VALUES_PER_SCALAR


def strict_block_dependency_mask() -> tf.Tensor:
    """Return the governed strict lower-triangular block dependency mask.

    Entry ``[i, j]`` is one exactly when output block ``i`` is permitted to
    depend on input block ``j``.

    Therefore only ``j < i`` entries are enabled.
    """
    return tf.linalg.band_part(
        tf.ones(
            (NUM_BLOCKS, NUM_BLOCKS),
            dtype=tf.float32,
        ),
        -1,
        0,
    ) - tf.eye(
        NUM_BLOCKS,
        dtype=tf.float32,
    )


class BlockConditionalSplineConditioner(tf.keras.Model):
    """Strict class-conditional block-autoregressive spline conditioner."""

    def __init__(
        self,
        num_classes: int,
        *,
        hidden_width: int = HIDDEN_WIDTH,
        hidden_depth: int = HIDDEN_DEPTH,
        class_embed_dim: int = CLASS_EMBED_DIM,
        decoder_bottleneck: int = DECODER_BOTTLENECK,
        name: str = "block_conditional_spline_conditioner",
    ):
        super().__init__(
            name=name
        )

        if num_classes < 2:
            raise ValueError(
                "num_classes must be at least 2"
            )

        if hidden_width <= 0:
            raise ValueError(
                "hidden_width must be positive"
            )

        if hidden_depth <= 0:
            raise ValueError(
                "hidden_depth must be positive"
            )

        if class_embed_dim <= 0:
            raise ValueError(
                "class_embed_dim must be positive"
            )

        if decoder_bottleneck <= 0:
            raise ValueError(
                "decoder_bottleneck must be positive"
            )

        self.num_classes = int(
            num_classes
        )

        self.hidden_width = int(
            hidden_width
        )

        self.hidden_depth = int(
            hidden_depth
        )

        self.class_embed_dim = int(
            class_embed_dim
        )

        self.decoder_bottleneck = int(
            decoder_bottleneck
        )

        self.block_projection = tf.keras.layers.Dense(
            self.hidden_width,
            activation=None,
            name="block_projection",
        )

        self.class_embedding = tf.keras.layers.Embedding(
            input_dim=self.num_classes,
            output_dim=self.class_embed_dim,
            name="class_embedding",
        )

        self.class_projection = tf.keras.layers.Dense(
            self.hidden_width,
            activation=None,
            name="class_projection",
        )

        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=NUM_BLOCKS,
            output_dim=self.hidden_width,
            name="position_embedding",
        )

        self.hidden_layers = [
            tf.keras.layers.Dense(
                self.hidden_width,
                activation="relu",
                name=f"hidden_{index}",
            )
            for index in range(
                self.hidden_depth
            )
        ]

        # Shared across all 25 block positions.
        self.decoder_bottleneck_layer = (
            tf.keras.layers.Dense(
                self.decoder_bottleneck,
                activation="relu",
                name="shared_decoder_bottleneck",
            )
        )

        self.decoder_output_layer = (
            tf.keras.layers.Dense(
                DECODER_OUTPUT_DIM,
                activation=None,
                name="shared_decoder_output",
            )
        )

    def _validate_inputs(
        self,
        blocks: tf.Tensor,
        labels: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        blocks = tf.convert_to_tensor(
            blocks,
            dtype=tf.float32,
        )

        labels = tf.convert_to_tensor(
            labels,
            dtype=tf.int32,
        )

        tf.debugging.assert_rank(
            blocks,
            3,
            message="blocks must have shape (B,25,192)",
        )

        tf.debugging.assert_equal(
            tf.shape(blocks)[1:],
            tf.constant(
                (
                    NUM_BLOCKS,
                    BLOCK_DIM,
                ),
                dtype=tf.int32,
            ),
            message="blocks must have shape (B,25,192)",
        )

        tf.debugging.assert_rank(
            labels,
            1,
            message="labels must have shape (B,)",
        )

        tf.debugging.assert_equal(
            tf.shape(labels)[0],
            tf.shape(blocks)[0],
            message="label batch size must match block batch size",
        )

        tf.debugging.assert_greater_equal(
            labels,
            tf.constant(
                0,
                dtype=tf.int32,
            ),
            message="class labels must be non-negative",
        )

        tf.debugging.assert_less(
            labels,
            tf.constant(
                self.num_classes,
                dtype=tf.int32,
            ),
            message="class label exceeds configured class count",
        )

        return blocks, labels

    def block_hidden(
        self,
        blocks: tf.Tensor,
        labels: tf.Tensor,
        *,
        training: bool = False,
    ) -> tf.Tensor:
        """Compute one strict autoregressive hidden representation per block."""
        blocks, labels = (
            self._validate_inputs(
                blocks,
                labels,
            )
        )

        projected_blocks = (
            self.block_projection(
                blocks,
                training=training,
            )
        )

        mask = strict_block_dependency_mask()

        # For output block i:
        #
        # context_i = sum_{j < i} projected_block_j
        #
        # The strict lower-triangular mask makes current-block and future-block
        # dependence structurally impossible.
        context = tf.einsum(
            "ij,bjh->bih",
            mask,
            projected_blocks,
        )

        class_features = (
            self.class_projection(
                self.class_embedding(
                    labels
                ),
                training=training,
            )
        )

        class_features = (
            class_features[:, None, :]
        )

        positions = tf.range(
            NUM_BLOCKS,
            dtype=tf.int32,
        )

        position_features = (
            self.position_embedding(
                positions
            )[None, :, :]
        )

        hidden = (
            context
            + class_features
            + position_features
        )

        for layer in self.hidden_layers:
            hidden = layer(
                hidden,
                training=training,
            )

        return hidden

    def call(
        self,
        blocks: tf.Tensor,
        labels: tf.Tensor,
        *,
        training: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Predict raw spline widths and heights.

        Returns
        -------
        raw_widths, raw_slope_controls:
            Each tensor has shape ``(B,25,192,8)``.
        """
        hidden = self.block_hidden(
            blocks,
            labels,
            training=training,
        )

        decoded = (
            self.decoder_bottleneck_layer(
                hidden,
                training=training,
            )
        )

        decoded = (
            self.decoder_output_layer(
                decoded,
                training=training,
            )
        )

        batch = tf.shape(
            blocks
        )[0]

        decoded = tf.reshape(
            decoded,
            (
                batch,
                NUM_BLOCKS,
                BLOCK_DIM,
                VALUES_PER_SCALAR,
            ),
        )

        raw_widths = decoded[
            ...,
            :NUM_BINS,
        ]

        raw_slope_controls = decoded[
            ...,
            NUM_BINS:,
        ]

        return (
            raw_widths,
            raw_slope_controls,
        )


__all__ = [
    "CLASS_EMBED_DIM",
    "DECODER_BOTTLENECK",
    "DECODER_OUTPUT_DIM",
    "HIDDEN_DEPTH",
    "HIDDEN_WIDTH",
    "VALUES_PER_SCALAR",
    "BlockConditionalSplineConditioner",
    "strict_block_dependency_mask",
]
