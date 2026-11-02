"""Governed shared class-conditional RBM for Paper 1 revision 2026.

The model uses:

    E(v,h|y) = -a^T v - b^T h - v^T W h - U[y,:]^T h

with:

    v in [0,1]^4800
    h in {0,1}^128

and one shared model per dataset / experimental seed.

Class identity affects the hidden conditional through U.  The visible
conditional remains class-independent given h and uses the governed bounded
continuous-visible observation model.
"""

from __future__ import annotations

import operator

import numpy as np
import tensorflow as tf

from trustforge.models.paper01_revision_2026.restricted_boltzmann_observation import (
    DTYPE,
    mean as visible_conditional_mean,
    sample_from_uniform as sample_visible_from_uniform,
    validate_visible_values,
)


CANONICAL_HEIGHT = 40
CANONICAL_WIDTH = 40
CANONICAL_CHANNELS = 3

VISIBLE_DIM = 4800
HIDDEN_UNITS = 128

STATE_DTYPE = DTYPE


def _exact_integer(
    value: int,
    *,
    name: str,
    minimum: int = 0,
) -> int:
    """Return one governed exact integer."""

    if isinstance(value, bool):
        raise TypeError(
            f"{name} must be an exact integer, not bool"
        )

    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(
            f"{name} must be an exact integer"
        ) from exc

    if result < minimum:
        raise ValueError(
            f"{name} must be >= {minimum}"
        )

    return result


def canonical_images_to_vectors(
    images: np.ndarray | tf.Tensor,
) -> tf.Tensor:
    """Validate canonical float32 images and return float64 visible vectors."""

    tensor = tf.convert_to_tensor(images)

    tf.debugging.assert_rank(
        tensor,
        4,
        message="RBM canonical input must have shape (B,40,40,3)",
    )
    tf.debugging.assert_type(
        tensor,
        tf.float32,
        message="RBM canonical input must be float32",
    )

    tf.debugging.assert_equal(
        tf.shape(tensor)[1],
        CANONICAL_HEIGHT,
        message="RBM canonical image height must be 40",
    )
    tf.debugging.assert_equal(
        tf.shape(tensor)[2],
        CANONICAL_WIDTH,
        message="RBM canonical image width must be 40",
    )
    tf.debugging.assert_equal(
        tf.shape(tensor)[3],
        CANONICAL_CHANNELS,
        message="RBM canonical image channels must be 3",
    )

    tf.debugging.assert_all_finite(
        tensor,
        "RBM canonical input contains non-finite values",
    )
    tf.debugging.assert_greater_equal(
        tensor,
        tf.constant(0.0, dtype=tf.float32),
        message="RBM canonical input contains values below zero",
    )
    tf.debugging.assert_less_equal(
        tensor,
        tf.constant(1.0, dtype=tf.float32),
        message="RBM canonical input contains values above one",
    )

    batch_size = tf.shape(tensor)[0]

    vectors = tf.reshape(
        tf.cast(
            tensor,
            STATE_DTYPE,
        ),
        (
            batch_size,
            VISIBLE_DIM,
        ),
    )

    return validate_visible_values(vectors)


def validate_class_ids(
    class_ids: np.ndarray | tf.Tensor,
    *,
    class_count: int,
) -> tf.Tensor:
    """Validate one class-id vector and return int64 tensor."""

    governed_class_count = _exact_integer(
        class_count,
        name="class_count",
        minimum=1,
    )

    labels = tf.convert_to_tensor(class_ids)

    tf.debugging.assert_rank(
        labels,
        1,
        message="RBM class_ids must be rank one",
    )

    if labels.dtype not in (
        tf.int32,
        tf.int64,
    ):
        raise TypeError(
            "RBM class_ids must have dtype int32 or int64"
        )

    labels = tf.cast(
        labels,
        tf.int64,
    )

    tf.debugging.assert_greater_equal(
        labels,
        tf.constant(0, dtype=tf.int64),
        message="RBM class identity must be non-negative",
    )
    tf.debugging.assert_less(
        labels,
        tf.constant(governed_class_count, dtype=tf.int64),
        message="RBM class identity exceeds governed class universe",
    )

    return labels


class SharedConditionalRBM(tf.Module):
    """One governed shared class-conditional revision-2026 RBM."""

    def __init__(
        self,
        *,
        W: np.ndarray | tf.Tensor,
        a: np.ndarray | tf.Tensor,
        b: np.ndarray | tf.Tensor,
        U: np.ndarray | tf.Tensor,
        name: str | None = None,
    ) -> None:
        super().__init__(
            name=name
            or "shared_conditional_rbm_revision_2026"
        )

        W_tensor = tf.convert_to_tensor(
            W,
            dtype=STATE_DTYPE,
        )
        a_tensor = tf.convert_to_tensor(
            a,
            dtype=STATE_DTYPE,
        )
        b_tensor = tf.convert_to_tensor(
            b,
            dtype=STATE_DTYPE,
        )
        U_tensor = tf.convert_to_tensor(
            U,
            dtype=STATE_DTYPE,
        )

        self._validate_initial_state(
            W_tensor,
            a_tensor,
            b_tensor,
            U_tensor,
        )

        self.W = tf.Variable(
            W_tensor,
            trainable=True,
            dtype=STATE_DTYPE,
            name="W",
        )
        self.a = tf.Variable(
            a_tensor,
            trainable=True,
            dtype=STATE_DTYPE,
            name="a",
        )
        self.b = tf.Variable(
            b_tensor,
            trainable=True,
            dtype=STATE_DTYPE,
            name="b",
        )
        self.U = tf.Variable(
            U_tensor,
            trainable=True,
            dtype=STATE_DTYPE,
            name="U",
        )

    @staticmethod
    def _validate_initial_state(
        W: tf.Tensor,
        a: tf.Tensor,
        b: tf.Tensor,
        U: tf.Tensor,
    ) -> None:
        """Fail closed unless state exactly matches governed architecture."""

        tf.debugging.assert_rank(
            W,
            2,
            message="RBM W must be rank two",
        )
        tf.debugging.assert_rank(
            a,
            1,
            message="RBM a must be rank one",
        )
        tf.debugging.assert_rank(
            b,
            1,
            message="RBM b must be rank one",
        )
        tf.debugging.assert_rank(
            U,
            2,
            message="RBM U must be rank two",
        )

        tf.debugging.assert_equal(
            tf.shape(W),
            tf.constant(
                [VISIBLE_DIM, HIDDEN_UNITS],
                dtype=tf.int32,
            ),
            message="RBM W has wrong governed shape",
        )
        tf.debugging.assert_equal(
            tf.shape(a),
            tf.constant(
                [VISIBLE_DIM],
                dtype=tf.int32,
            ),
            message="RBM a has wrong governed shape",
        )
        tf.debugging.assert_equal(
            tf.shape(b),
            tf.constant(
                [HIDDEN_UNITS],
                dtype=tf.int32,
            ),
            message="RBM b has wrong governed shape",
        )
        tf.debugging.assert_equal(
            tf.shape(U)[1],
            HIDDEN_UNITS,
            message="RBM U hidden width must equal 128",
        )
        tf.debugging.assert_greater_equal(
            tf.shape(U)[0],
            1,
            message="RBM U must contain at least one class row",
        )

        for name, tensor in (
            ("W", W),
            ("a", a),
            ("b", b),
            ("U", U),
        ):
            tf.debugging.assert_type(
                tensor,
                STATE_DTYPE,
                message=f"RBM {name} must be float64",
            )
            tf.debugging.assert_all_finite(
                tensor,
                f"RBM {name} contains non-finite values",
            )

    @property
    def class_count(self) -> int:
        """Return the governed number of class-conditioning rows."""

        return int(
            self.U.shape[0]
        )

    def state_dict(self) -> dict[str, np.ndarray]:
        """Return an exact float64 copy of authoritative model state."""

        return {
            "W": np.array(
                self.W.numpy(),
                dtype=np.float64,
                copy=True,
            ),
            "a": np.array(
                self.a.numpy(),
                dtype=np.float64,
                copy=True,
            ),
            "b": np.array(
                self.b.numpy(),
                dtype=np.float64,
                copy=True,
            ),
            "U": np.array(
                self.U.numpy(),
                dtype=np.float64,
                copy=True,
            ),
        }

    def hidden_logits(
        self,
        visible: np.ndarray | tf.Tensor,
        class_ids: np.ndarray | tf.Tensor,
    ) -> tf.Tensor:
        """Return b + vW + U_y for governed visible vectors."""

        v = validate_visible_values(
            visible
        )

        tf.debugging.assert_rank(
            v,
            2,
            message="RBM visible vectors must be rank two",
        )
        tf.debugging.assert_equal(
            tf.shape(v)[1],
            VISIBLE_DIM,
            message="RBM visible vectors must have dimension 4800",
        )

        y = validate_class_ids(
            class_ids,
            class_count=self.class_count,
        )

        tf.debugging.assert_equal(
            tf.shape(y)[0],
            tf.shape(v)[0],
            message="RBM class_ids batch must match visible batch",
        )

        class_term = tf.gather(
            self.U,
            y,
            axis=0,
        )

        logits = (
            tf.linalg.matmul(
                v,
                self.W,
            )
            + self.b
            + class_term
        )

        tf.debugging.assert_all_finite(
            logits,
            "RBM hidden logits are non-finite",
        )

        return logits

    def hidden_probabilities(
        self,
        visible: np.ndarray | tf.Tensor,
        class_ids: np.ndarray | tf.Tensor,
    ) -> tf.Tensor:
        """Return p(h=1 | v,y)."""

        probabilities = tf.math.sigmoid(
            self.hidden_logits(
                visible,
                class_ids,
            )
        )

        tf.debugging.assert_all_finite(
            probabilities,
            "RBM hidden probabilities are non-finite",
        )
        tf.debugging.assert_greater_equal(
            probabilities,
            tf.constant(0.0, dtype=STATE_DTYPE),
            message="RBM hidden probability lies below zero",
        )
        tf.debugging.assert_less_equal(
            probabilities,
            tf.constant(1.0, dtype=STATE_DTYPE),
            message="RBM hidden probability lies above one",
        )

        return probabilities

    def sample_hidden_from_uniform(
        self,
        visible: np.ndarray | tf.Tensor,
        class_ids: np.ndarray | tf.Tensor,
        uniform: np.ndarray | tf.Tensor,
    ) -> tf.Tensor:
        """Sample Bernoulli hidden states using caller-governed U[0,1] draws."""

        probabilities = self.hidden_probabilities(
            visible,
            class_ids,
        )

        draws = tf.convert_to_tensor(
            uniform,
            dtype=STATE_DTYPE,
        )

        tf.debugging.assert_equal(
            tf.shape(draws),
            tf.shape(probabilities),
            message="RBM hidden uniform draws must match hidden-probability shape",
        )
        tf.debugging.assert_all_finite(
            draws,
            "RBM hidden uniform draws contain non-finite values",
        )
        tf.debugging.assert_greater_equal(
            draws,
            tf.constant(0.0, dtype=STATE_DTYPE),
            message="RBM hidden uniform draw lies below zero",
        )
        tf.debugging.assert_less_equal(
            draws,
            tf.constant(1.0, dtype=STATE_DTYPE),
            message="RBM hidden uniform draw lies above one",
        )

        return tf.cast(
            draws < probabilities,
            STATE_DTYPE,
        )

    def visible_natural_parameter(
        self,
        hidden: np.ndarray | tf.Tensor,
    ) -> tf.Tensor:
        """Return theta = a + h W^T."""

        h = tf.convert_to_tensor(
            hidden,
            dtype=STATE_DTYPE,
        )

        tf.debugging.assert_rank(
            h,
            2,
            message="RBM hidden state must be rank two",
        )
        tf.debugging.assert_equal(
            tf.shape(h)[1],
            HIDDEN_UNITS,
            message="RBM hidden state must have width 128",
        )
        tf.debugging.assert_all_finite(
            h,
            "RBM hidden state contains non-finite values",
        )
        tf.debugging.assert_greater_equal(
            h,
            tf.constant(0.0, dtype=STATE_DTYPE),
            message="RBM hidden state contains values below zero",
        )
        tf.debugging.assert_less_equal(
            h,
            tf.constant(1.0, dtype=STATE_DTYPE),
            message="RBM hidden state contains values above one",
        )

        theta = (
            tf.linalg.matmul(
                h,
                self.W,
                transpose_b=True,
            )
            + self.a
        )

        tf.debugging.assert_all_finite(
            theta,
            "RBM visible natural parameter is non-finite",
        )

        return theta

    def visible_mean_given_hidden(
        self,
        hidden: np.ndarray | tf.Tensor,
    ) -> tf.Tensor:
        """Return analytic E[v|h] under the governed visible conditional."""

        return visible_conditional_mean(
            self.visible_natural_parameter(
                hidden
            )
        )

    def sample_visible_from_uniform(
        self,
        hidden: np.ndarray | tf.Tensor,
        uniform: np.ndarray | tf.Tensor,
    ) -> tf.Tensor:
        """Sample visible state using caller-governed U[0,1] draws."""

        theta = self.visible_natural_parameter(
            hidden
        )

        draws = tf.convert_to_tensor(
            uniform,
            dtype=STATE_DTYPE,
        )

        tf.debugging.assert_equal(
            tf.shape(draws),
            tf.shape(theta),
            message="RBM visible uniform draws must match visible-state shape",
        )

        return sample_visible_from_uniform(
            theta,
            draws,
        )


__all__ = [
    "CANONICAL_CHANNELS",
    "CANONICAL_HEIGHT",
    "CANONICAL_WIDTH",
    "HIDDEN_UNITS",
    "STATE_DTYPE",
    "SharedConditionalRBM",
    "VISIBLE_DIM",
    "canonical_images_to_vectors",
    "validate_class_ids",
]
