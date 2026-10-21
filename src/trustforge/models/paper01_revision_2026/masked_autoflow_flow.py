"""Single block-autoregressive flow stage for Paper 1 revision-2026 MAF.

A stage operates on canonical block tensors shaped ``(B, 25, 192)``.

Forward density evaluation is parallel because transform parameters obey a
strict block-autoregressive mask.

Inverse synthesis reconstructs exactly 25 autoregressive blocks sequentially.
All 192 scalar dimensions inside the current block are inverted in parallel.

Optional permutations operate on whole blocks only. Stage outputs are restored
to canonical block order before being returned.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf

from trustforge.models.paper01_revision_2026.masked_autoflow_blocks import (
    BLOCK_DIM,
    NUM_BLOCKS,
    make_block_permutation,
    permute_blocks,
)
from trustforge.models.paper01_revision_2026.masked_autoflow_conditioner import (
    BlockConditionalSplineConditioner,
)
from trustforge.models.paper01_revision_2026.masked_autoflow_spline import (
    forward as spline_forward,
    inverse as spline_inverse,
)


MAX_INVERSE_BLOCK_STAGES = NUM_BLOCKS


def _identity_permutation() -> np.ndarray:
    return np.arange(
        NUM_BLOCKS,
        dtype=np.int32,
    )


class BlockMAFFlowStage(tf.keras.Model):
    """One governed block-conditional Masked AutoFlow stage."""

    def __init__(
        self,
        num_classes: int,
        *,
        permutation_seed: Optional[int] = None,
        name: str = "block_maf_flow_stage",
    ):
        super().__init__(
            name=name
        )

        self.num_classes = int(
            num_classes
        )

        self.conditioner = (
            BlockConditionalSplineConditioner(
                num_classes=self.num_classes,
            )
        )

        if permutation_seed is None:
            permutation = _identity_permutation()
            inverse_permutation = (
                _identity_permutation()
            )
        else:
            (
                permutation,
                inverse_permutation,
            ) = make_block_permutation(
                int(permutation_seed)
            )

        self.permutation_seed = (
            None
            if permutation_seed is None
            else int(permutation_seed)
        )

        self.permutation = tf.constant(
            permutation,
            dtype=tf.int32,
        )

        self.inverse_permutation = tf.constant(
            inverse_permutation,
            dtype=tf.int32,
        )

    def _validate_blocks(
        self,
        blocks: tf.Tensor,
    ) -> tf.Tensor:
        blocks = tf.convert_to_tensor(
            blocks,
            dtype=tf.float32,
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

        return blocks

    def _ordered(
        self,
        blocks: tf.Tensor,
    ) -> tf.Tensor:
        return permute_blocks(
            blocks,
            self.permutation,
        )

    def _canonical(
        self,
        ordered_blocks: tf.Tensor,
    ) -> tf.Tensor:
        return permute_blocks(
            ordered_blocks,
            self.inverse_permutation,
        )

    def forward(
        self,
        blocks: tf.Tensor,
        labels: tf.Tensor,
        *,
        training: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Apply one flow stage in the density-evaluation direction.

        Returns
        -------
        transformed_blocks:
            Canonical-order tensor shaped ``(B,25,192)``.
        log_abs_det:
            One summed forward log-Jacobian value per sample, shaped ``(B,)``.
        """
        blocks = self._validate_blocks(
            blocks
        )

        ordered = self._ordered(
            blocks
        )

        (
            raw_widths,
            raw_slope_controls,
        ) = self.conditioner(
            ordered,
            labels,
            training=training,
        )

        batch = tf.shape(
            ordered
        )[0]

        flat_ordered = tf.reshape(
            ordered,
            (
                batch,
                NUM_BLOCKS * BLOCK_DIM,
            ),
        )

        flat_widths = tf.reshape(
            raw_widths,
            (
                batch,
                NUM_BLOCKS * BLOCK_DIM,
                -1,
            ),
        )

        flat_heights = tf.reshape(
            raw_slope_controls,
            (
                batch,
                NUM_BLOCKS * BLOCK_DIM,
                -1,
            ),
        )

        (
            flat_transformed,
            flat_log_det,
        ) = spline_forward(
            flat_ordered,
            flat_widths,
            flat_heights,
        )

        transformed_ordered = tf.reshape(
            flat_transformed,
            (
                batch,
                NUM_BLOCKS,
                BLOCK_DIM,
            ),
        )

        transformed = self._canonical(
            transformed_ordered
        )

        log_abs_det = tf.reduce_sum(
            flat_log_det,
            axis=-1,
        )

        tf.debugging.assert_all_finite(
            transformed,
            "non-finite flow-stage forward output",
        )

        tf.debugging.assert_all_finite(
            log_abs_det,
            "non-finite flow-stage forward log determinant",
        )

        return (
            transformed,
            log_abs_det,
        )

    def inverse(
        self,
        transformed_blocks: tf.Tensor,
        labels: tf.Tensor,
        *,
        training: bool = False,
        return_diagnostics: bool = False,
    ):
        """Invert one flow stage using 25 sequential block reconstructions.

        The conditioner is re-evaluated once per block using only already
        recovered earlier blocks plus the class condition. Strict masking makes
        unrecovered current/future placeholder values irrelevant.
        """
        transformed_blocks = (
            self._validate_blocks(
                transformed_blocks
            )
        )

        transformed_ordered = (
            self._ordered(
                transformed_blocks
            )
        )

        batch = tf.shape(
            transformed_ordered
        )[0]

        recovered_ordered = tf.zeros(
            (
                batch,
                NUM_BLOCKS,
                BLOCK_DIM,
            ),
            dtype=tf.float32,
        )

        inverse_log_abs_det = tf.zeros(
            (batch,),
            dtype=tf.float32,
        )

        executed_block_stages = 0

        for block_index in range(
            NUM_BLOCKS
        ):
            (
                raw_widths,
                raw_slope_controls,
            ) = self.conditioner(
                recovered_ordered,
                labels,
                training=training,
            )

            current_y = transformed_ordered[
                :,
                block_index,
                :,
            ]

            current_widths = raw_widths[
                :,
                block_index,
                :,
                :,
            ]

            current_slope_controls = raw_slope_controls[
                :,
                block_index,
                :,
                :,
            ]

            (
                current_x,
                current_inverse_log_det,
            ) = spline_inverse(
                current_y,
                current_widths,
                current_slope_controls,
            )

            before = recovered_ordered[
                :,
                :block_index,
                :,
            ]

            after = recovered_ordered[
                :,
                block_index + 1:,
                :,
            ]

            recovered_ordered = tf.concat(
                [
                    before,
                    current_x[:, None, :],
                    after,
                ],
                axis=1,
            )

            inverse_log_abs_det = (
                inverse_log_abs_det
                + tf.reduce_sum(
                    current_inverse_log_det,
                    axis=-1,
                )
            )

            executed_block_stages += 1

        recovered = self._canonical(
            recovered_ordered
        )

        tf.debugging.assert_all_finite(
            recovered,
            "non-finite flow-stage inverse output",
        )

        tf.debugging.assert_all_finite(
            inverse_log_abs_det,
            "non-finite flow-stage inverse log determinant",
        )

        if return_diagnostics:
            diagnostics: Dict[str, int] = {
                "inverse_block_stages": (
                    executed_block_stages
                ),
                "maximum_inverse_block_stages": (
                    MAX_INVERSE_BLOCK_STAGES
                ),
            }

            return (
                recovered,
                inverse_log_abs_det,
                diagnostics,
            )

        return (
            recovered,
            inverse_log_abs_det,
        )


__all__ = [
    "MAX_INVERSE_BLOCK_STAGES",
    "BlockMAFFlowStage",
]
