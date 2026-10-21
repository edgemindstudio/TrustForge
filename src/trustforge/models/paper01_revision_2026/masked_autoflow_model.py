"""Five-stage conditional Masked AutoFlow for Paper 1 revision-2026.

The model operates directly on canonical ``40 x 40 x 3`` float32 images in
``[0,1]``.

Density evaluation:

    image -> blocks -> five forward flow stages -> Uniform(0,1) base

Because the bounded base density is constant on the governed domain, sample
log probability is the accumulated forward log-Jacobian.

Sampling:

    Uniform(0,1) latent blocks -> inverse stages in reverse order -> image

No logit transform, endpoint epsilon, universal dequantization, or post-hoc
clipping is used.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import tensorflow as tf

from trustforge.models.paper01_revision_2026.masked_autoflow_blocks import (
    BLOCK_DIM,
    NUM_BLOCKS,
    blocks_to_images,
    images_to_blocks,
)
from trustforge.models.paper01_revision_2026.masked_autoflow_flow import (
    MAX_INVERSE_BLOCK_STAGES,
    BlockMAFFlowStage,
)


NUM_FLOWS = 5
MAX_TOTAL_INVERSE_BLOCK_STAGES = (
    NUM_FLOWS
    * MAX_INVERSE_BLOCK_STAGES
)

DEFAULT_PERMUTATION_SEEDS: Tuple[int, ...] = (
    42,
    43,
    44,
    45,
    46,
)


class ConditionalBlockMAF(tf.keras.Model):
    """Governed five-flow conditional block Masked AutoFlow."""

    def __init__(
        self,
        num_classes: int,
        *,
        permutation_seeds: Sequence[int] = DEFAULT_PERMUTATION_SEEDS,
        name: str = "conditional_block_maf_revision_2026",
    ):
        super().__init__(
            name=name
        )

        if num_classes < 2:
            raise ValueError(
                "num_classes must be at least 2"
            )

        permutation_seeds = tuple(
            int(seed)
            for seed in permutation_seeds
        )

        if len(permutation_seeds) != NUM_FLOWS:
            raise ValueError(
                "exactly five permutation seeds are required"
            )

        if len(set(permutation_seeds)) != NUM_FLOWS:
            raise ValueError(
                "five distinct permutation seeds are required"
            )

        self.num_classes = int(
            num_classes
        )

        self.permutation_seeds = (
            permutation_seeds
        )

        self.flow_stages = [
            BlockMAFFlowStage(
                num_classes=self.num_classes,
                permutation_seed=seed,
                name=f"block_maf_flow_{index}",
            )
            for index, seed in enumerate(
                self.permutation_seeds
            )
        ]

    def forward_blocks(
        self,
        blocks: tf.Tensor,
        labels: tf.Tensor,
        *,
        training: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Apply all five flow stages in density-evaluation order."""
        current = tf.convert_to_tensor(
            blocks,
            dtype=tf.float32,
        )

        batch = tf.shape(
            current
        )[0]

        total_log_det = tf.zeros(
            (batch,),
            dtype=tf.float32,
        )

        for stage in self.flow_stages:
            (
                current,
                stage_log_det,
            ) = stage.forward(
                current,
                labels,
                training=training,
            )

            total_log_det = (
                total_log_det
                + stage_log_det
            )

        tf.debugging.assert_all_finite(
            current,
            "non-finite five-flow forward output",
        )

        tf.debugging.assert_all_finite(
            total_log_det,
            "non-finite five-flow log determinant",
        )

        return (
            current,
            total_log_det,
        )

    def inverse_blocks(
        self,
        latent_blocks: tf.Tensor,
        labels: tf.Tensor,
        *,
        training: bool = False,
        return_diagnostics: bool = False,
    ):
        """Invert all five stages in reverse order."""
        current = tf.convert_to_tensor(
            latent_blocks,
            dtype=tf.float32,
        )

        batch = tf.shape(
            current
        )[0]

        total_inverse_log_det = tf.zeros(
            (batch,),
            dtype=tf.float32,
        )

        total_block_stages = 0

        for stage in reversed(
            self.flow_stages
        ):
            (
                current,
                stage_inverse_log_det,
                diagnostics,
            ) = stage.inverse(
                current,
                labels,
                training=training,
                return_diagnostics=True,
            )

            total_inverse_log_det = (
                total_inverse_log_det
                + stage_inverse_log_det
            )

            total_block_stages += int(
                diagnostics[
                    "inverse_block_stages"
                ]
            )

        tf.debugging.assert_all_finite(
            current,
            "non-finite five-flow inverse output",
        )

        tf.debugging.assert_all_finite(
            total_inverse_log_det,
            "non-finite five-flow inverse log determinant",
        )

        if return_diagnostics:
            diagnostics: Dict[str, int] = {
                "num_flows": NUM_FLOWS,
                "inverse_block_stages": (
                    total_block_stages
                ),
                "maximum_inverse_block_stages": (
                    MAX_TOTAL_INVERSE_BLOCK_STAGES
                ),
            }

            return (
                current,
                total_inverse_log_det,
                diagnostics,
            )

        return (
            current,
            total_inverse_log_det,
        )

    def log_prob(
        self,
        images: tf.Tensor,
        labels: tf.Tensor,
        *,
        training: bool = False,
    ) -> tf.Tensor:
        """Return one conditional log probability per canonical image.

        The final latent coordinates remain in ``[0,1]`` and use a factorized
        Uniform(0,1) base.

        Therefore ``log p_base(z) = 0`` on the governed support and:

            log p(x | y) = accumulated forward log |det J|.
        """
        blocks = images_to_blocks(
            images
        )

        (
            latent_blocks,
            total_log_det,
        ) = self.forward_blocks(
            blocks,
            labels,
            training=training,
        )

        tf.debugging.assert_greater_equal(
            latent_blocks,
            tf.constant(
                0.0,
                dtype=tf.float32,
            ),
            message="latent value below Uniform base support",
        )

        tf.debugging.assert_less_equal(
            latent_blocks,
            tf.constant(
                1.0,
                dtype=tf.float32,
            ),
            message="latent value above Uniform base support",
        )

        tf.debugging.assert_all_finite(
            total_log_det,
            "non-finite conditional log probability",
        )

        return total_log_det

    def sample(
        self,
        labels: tf.Tensor,
        *,
        seed: int,
    ) -> tf.Tensor:
        """Generate canonical images for the requested class conditions."""
        labels = tf.convert_to_tensor(
            labels,
            dtype=tf.int32,
        )

        tf.debugging.assert_rank(
            labels,
            1,
            message="labels must have shape (B,)",
        )

        batch = tf.shape(
            labels
        )[0]

        latent_blocks = (
            tf.random.stateless_uniform(
                (
                    batch,
                    NUM_BLOCKS,
                    BLOCK_DIM,
                ),
                seed=(
                    int(seed),
                    0,
                ),
                minval=0.0,
                maxval=1.0,
                dtype=tf.float32,
            )
        )

        blocks, _ = self.inverse_blocks(
            latent_blocks,
            labels,
            training=False,
        )

        images = blocks_to_images(
            blocks
        )

        tf.debugging.assert_all_finite(
            images,
            "non-finite generated image",
        )

        tf.debugging.assert_greater_equal(
            images,
            tf.constant(
                0.0,
                dtype=tf.float32,
            ),
            message="generated value below canonical lower bound",
        )

        tf.debugging.assert_less_equal(
            images,
            tf.constant(
                1.0,
                dtype=tf.float32,
            ),
            message="generated value above canonical upper bound",
        )

        return images


__all__ = [
    "DEFAULT_PERMUTATION_SEEDS",
    "MAX_TOTAL_INVERSE_BLOCK_STAGES",
    "NUM_FLOWS",
    "ConditionalBlockMAF",
]
