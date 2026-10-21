"""Trainable class-model state for the governed Paper 1 revision-2026 GMM.

Each instance represents exactly one class-specific censored diagonal
Gaussian mixture.

Canonical benchmark images enter as float32 ``(B,40,40,3)`` tensors in
``[0,1]`` and are cast batchwise to float64 before likelihood evaluation.

Historical Gaussian-mixture implementations remain unchanged.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import tensorflow as tf

from trustforge.models.paper01_revision_2026.gaussian_mixture_observation import (
    CANONICAL_DIM,
    DTYPE,
    NUM_COMPONENTS,
    mean_censored_mixture_nll,
    mixture_log_prob,
    transformed_scale,
)


CANONICAL_HEIGHT = 40
CANONICAL_WIDTH = 40
CANONICAL_CHANNELS = 3


def canonical_images_to_vectors(
    images: tf.Tensor,
) -> tf.Tensor:
    """Validate canonical images and return float64 flattened vectors."""

    images = tf.convert_to_tensor(
        images
    )

    tf.debugging.assert_rank(
        images,
        4,
        message=(
            "GMM canonical input must have shape (B,40,40,3)"
        ),
    )

    tf.debugging.assert_type(
        images,
        tf.float32,
        message=(
            "GMM canonical input must be float32 before governed "
            "batchwise float64 conversion"
        ),
    )

    tf.debugging.assert_equal(
        tf.shape(images)[1],
        CANONICAL_HEIGHT,
        message="GMM canonical image height must be 40",
    )

    tf.debugging.assert_equal(
        tf.shape(images)[2],
        CANONICAL_WIDTH,
        message="GMM canonical image width must be 40",
    )

    tf.debugging.assert_equal(
        tf.shape(images)[3],
        CANONICAL_CHANNELS,
        message="GMM canonical image channels must be 3",
    )

    tf.debugging.assert_all_finite(
        images,
        "GMM canonical images contain non-finite values",
    )

    tf.debugging.assert_greater_equal(
        images,
        tf.constant(
            0.0,
            dtype=tf.float32,
        ),
        message="GMM canonical images contain values below zero",
    )

    tf.debugging.assert_less_equal(
        images,
        tf.constant(
            1.0,
            dtype=tf.float32,
        ),
        message="GMM canonical images contain values above one",
    )

    batch_size = tf.shape(
        images
    )[0]

    vectors = tf.reshape(
        tf.cast(
            images,
            DTYPE,
        ),
        (
            batch_size,
            CANONICAL_DIM,
        ),
    )

    tf.debugging.assert_all_finite(
        vectors,
        "GMM float64 canonical vectors contain non-finite values",
    )

    return vectors


class CensoredDiagonalGaussianMixture(tf.Module):
    """One trainable class-specific revision-2026 GMM."""

    def __init__(
        self,
        *,
        logits: np.ndarray | tf.Tensor,
        means: np.ndarray | tf.Tensor,
        raw_scales: np.ndarray | tf.Tensor,
        name: str | None = None,
    ) -> None:
        super().__init__(
            name=name
            or "censored_diagonal_gaussian_mixture_revision_2026"
        )

        logits_tensor = tf.convert_to_tensor(
            logits,
            dtype=DTYPE,
        )

        means_tensor = tf.convert_to_tensor(
            means,
            dtype=DTYPE,
        )

        raw_scales_tensor = tf.convert_to_tensor(
            raw_scales,
            dtype=DTYPE,
        )

        self._validate_initial_state(
            logits_tensor,
            means_tensor,
            raw_scales_tensor,
        )

        self.logits = tf.Variable(
            logits_tensor,
            trainable=True,
            dtype=DTYPE,
            name="logits",
        )

        self.means = tf.Variable(
            means_tensor,
            trainable=True,
            dtype=DTYPE,
            name="means",
        )

        self.raw_scales = tf.Variable(
            raw_scales_tensor,
            trainable=True,
            dtype=DTYPE,
            name="raw_scales",
        )

    @staticmethod
    def _validate_initial_state(
        logits: tf.Tensor,
        means: tf.Tensor,
        raw_scales: tf.Tensor,
    ) -> None:
        """Fail closed on malformed or non-finite model state."""

        tf.debugging.assert_rank(
            logits,
            1,
            message="GMM logits must have shape (5,)",
        )

        tf.debugging.assert_rank(
            means,
            2,
            message="GMM means must have shape (5,4800)",
        )

        tf.debugging.assert_rank(
            raw_scales,
            2,
            message="GMM raw scales must have shape (5,4800)",
        )

        tf.debugging.assert_equal(
            tf.shape(logits),
            tf.constant(
                [NUM_COMPONENTS],
                dtype=tf.int32,
            ),
            message="GMM logits must have exact shape (5,)",
        )

        expected_matrix_shape = tf.constant(
            [
                NUM_COMPONENTS,
                CANONICAL_DIM,
            ],
            dtype=tf.int32,
        )

        tf.debugging.assert_equal(
            tf.shape(means),
            expected_matrix_shape,
            message="GMM means must have exact shape (5,4800)",
        )

        tf.debugging.assert_equal(
            tf.shape(raw_scales),
            expected_matrix_shape,
            message="GMM raw scales must have exact shape (5,4800)",
        )

        tf.debugging.assert_all_finite(
            logits,
            "GMM initial logits contain non-finite values",
        )

        tf.debugging.assert_all_finite(
            means,
            "GMM initial means contain non-finite values",
        )

        tf.debugging.assert_all_finite(
            raw_scales,
            "GMM initial raw scales contain non-finite values",
        )

        # Validate the transformed representation immediately.
        scales = transformed_scale(
            raw_scales
        )

        tf.debugging.assert_all_finite(
            scales,
            "GMM initial transformed scales contain non-finite values",
        )

        weights = tf.nn.softmax(
            logits
        )

        tf.debugging.assert_all_finite(
            weights,
            "GMM initial mixture weights contain non-finite values",
        )

        tf.debugging.assert_positive(
            weights,
            message="GMM initial mixture weights must be positive",
        )

        tf.debugging.assert_near(
            tf.reduce_sum(
                weights
            ),
            tf.constant(
                1.0,
                dtype=DTYPE,
            ),
            atol=tf.constant(
                1e-12,
                dtype=DTYPE,
            ),
            rtol=tf.constant(
                0.0,
                dtype=DTYPE,
            ),
            message="GMM initial mixture weights must sum to one",
        )

    @property
    def trainable_parameters(self) -> tuple[tf.Variable, ...]:
        """Return the governed trainable parameter order."""

        return (
            self.logits,
            self.means,
            self.raw_scales,
        )

    def mixture_weights(
        self,
    ) -> tf.Tensor:
        """Return normalized float64 component weights."""

        weights = tf.nn.softmax(
            self.logits
        )

        tf.debugging.assert_all_finite(
            weights,
            "GMM mixture weights contain non-finite values",
        )

        tf.debugging.assert_positive(
            weights,
            message="GMM mixture weights must remain positive",
        )

        return weights

    def scales(
        self,
    ) -> tf.Tensor:
        """Return governed positive component scales."""

        return transformed_scale(
            self.raw_scales
        )

    def log_prob_vectors(
        self,
        vectors: tf.Tensor,
    ) -> tf.Tensor:
        """Return one log probability per already-flattened vector."""

        vectors = tf.convert_to_tensor(
            vectors,
            dtype=DTYPE,
        )

        tf.debugging.assert_rank(
            vectors,
            2,
            message="GMM vectors must have shape (B,4800)",
        )

        tf.debugging.assert_equal(
            tf.shape(vectors)[1],
            CANONICAL_DIM,
            message="GMM vectors must have dimension 4800",
        )

        return mixture_log_prob(
            vectors,
            self.logits,
            self.means,
            self.raw_scales,
        )

    def log_prob(
        self,
        images: tf.Tensor,
    ) -> tf.Tensor:
        """Return one class-model log probability per canonical image."""

        vectors = canonical_images_to_vectors(
            images
        )

        return self.log_prob_vectors(
            vectors
        )

    def nll(
        self,
        images: tf.Tensor,
    ) -> tf.Tensor:
        """Return mean censored-mixture NLL for canonical images."""

        vectors = canonical_images_to_vectors(
            images
        )

        return mean_censored_mixture_nll(
            vectors,
            self.logits,
            self.means,
            self.raw_scales,
        )

    def state_arrays(
        self,
    ) -> Dict[str, np.ndarray]:
        """Return an exact detached float64 snapshot of authoritative state."""

        state = {
            "logits": self.logits.numpy().copy(),
            "means": self.means.numpy().copy(),
            "raw_scales": self.raw_scales.numpy().copy(),
        }

        for key, value in state.items():
            if value.dtype != np.float64:
                raise RuntimeError(
                    f"{key} snapshot is not float64"
                )

            if not np.isfinite(
                value
            ).all():
                raise RuntimeError(
                    f"{key} snapshot contains non-finite values"
                )

        return state


__all__ = [
    "CANONICAL_CHANNELS",
    "CANONICAL_HEIGHT",
    "CANONICAL_WIDTH",
    "CensoredDiagonalGaussianMixture",
    "canonical_images_to_vectors",
]
