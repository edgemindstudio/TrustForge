"""Governed CD-1 estimator for the Paper 1 revision-2026 RBM.

Selected semantics:

- data-initialized CD-1;
- positive hidden sufficient statistics use conditional probabilities;
- one Bernoulli hidden transition is sampled;
- one bounded-exponential visible transition is sampled;
- class labels remain clamped throughout the negative chain;
- terminal negative hidden sufficient statistics use conditional probabilities;
- update statistics are positive-minus-negative batch means;
- hidden and visible random streams are separately and deterministically derived;
- no persistent chain state exists in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import operator

import numpy as np
import tensorflow as tf

from trustforge.models.paper01_revision_2026.restricted_boltzmann_model import (
    HIDDEN_UNITS,
    STATE_DTYPE,
    VISIBLE_DIM,
    SharedConditionalRBM,
    validate_class_ids,
)
from trustforge.models.paper01_revision_2026.restricted_boltzmann_observation import (
    validate_visible_values,
)


CD_K = 1

_HIDDEN_STREAM_TAG = 0x48494444
_VISIBLE_STREAM_TAG = 0x56495349


@dataclass(frozen=True)
class RBMCD1Result:
    """One governed CD-1 stochastic transition and its update statistics."""

    positive_hidden_probabilities: np.ndarray
    sampled_hidden_state: np.ndarray
    negative_visible_state: np.ndarray
    negative_hidden_probabilities: np.ndarray
    delta_W: np.ndarray
    delta_a: np.ndarray
    delta_b: np.ndarray
    delta_U: np.ndarray


def _exact_nonnegative_integer(
    value: int,
    *,
    name: str,
) -> int:
    if isinstance(
        value,
        bool,
    ):
        raise TypeError(
            f"{name} must be an exact integer, not bool"
        )

    try:
        result = operator.index(
            value
        )
    except TypeError as exc:
        raise TypeError(
            f"{name} must be an exact integer"
        ) from exc

    if result < 0:
        raise ValueError(
            f"{name} must be non-negative"
        )

    return result


def _validate_seed(
    experimental_seed: int,
) -> int:
    seed = _exact_nonnegative_integer(
        experimental_seed,
        name="experimental_seed",
    )

    if seed > np.iinfo(
        np.uint32
    ).max:
        raise ValueError(
            "experimental_seed must lie in [0, 2**32 - 1]"
        )

    return seed


def _validate_dataset_id(
    dataset_id: str,
) -> str:
    if not isinstance(
        dataset_id,
        str,
    ):
        raise TypeError(
            "dataset_id must be a string"
        )

    normalized = dataset_id.strip()

    if not normalized:
        raise ValueError(
            "dataset_id must not be empty"
        )

    return normalized


def _dataset_entropy_words(
    dataset_id: str,
) -> tuple[
    int,
    int,
    int,
    int,
]:
    digest = hashlib.blake2b(
        dataset_id.encode(
            "utf-8"
        ),
        digest_size=16,
        person=b"TF-RBM-CD1",
    ).digest()

    words = np.frombuffer(
        digest,
        dtype="<u4",
    )

    return tuple(
        int(word)
        for word in words
    )


def _uniform_stream(
    *,
    dataset_id: str,
    experimental_seed: int,
    update_index: int,
    stream_tag: int,
    shape: tuple[int, int],
) -> np.ndarray:
    """Return one deterministic float64 U[0,1) stream."""

    low = update_index & 0xFFFFFFFF
    high = (
        update_index >> 32
    ) & 0xFFFFFFFF

    seed_sequence = np.random.SeedSequence(
        [
            *_dataset_entropy_words(
                dataset_id
            ),
            experimental_seed,
            low,
            high,
            stream_tag,
        ]
    )

    rng = np.random.default_rng(
        seed_sequence
    )

    draws = rng.random(
        shape,
        dtype=np.float64,
    )

    if draws.dtype != np.float64:
        raise RuntimeError(
            "CD random stream is not float64"
        )

    if not np.all(
        np.isfinite(draws)
    ):
        raise RuntimeError(
            "CD random stream contains non-finite values"
        )

    if np.any(
        draws < 0.0
    ) or np.any(
        draws >= 1.0
    ):
        raise RuntimeError(
            "CD random stream lies outside [0,1)"
        )

    return draws


def _validate_visible_batch(
    visible_batch: np.ndarray | tf.Tensor,
) -> tf.Tensor:
    visible = validate_visible_values(
        visible_batch
    )

    tf.debugging.assert_rank(
        visible,
        2,
        message="RBM CD visible batch must be rank two",
    )

    tf.debugging.assert_equal(
        tf.shape(visible)[1],
        VISIBLE_DIM,
        message="RBM CD visible batch width must equal 4800",
    )

    tf.debugging.assert_greater_equal(
        tf.shape(visible)[0],
        1,
        message="RBM CD visible batch must not be empty",
    )

    return visible


def _class_interaction_statistics(
    hidden_probabilities: tf.Tensor,
    class_ids: tf.Tensor,
    *,
    class_count: int,
) -> tf.Tensor:
    """Return per-class summed hidden probabilities."""

    one_hot = tf.one_hot(
        class_ids,
        depth=class_count,
        dtype=STATE_DTYPE,
    )

    statistics = tf.linalg.matmul(
        one_hot,
        hidden_probabilities,
        transpose_a=True,
    )

    tf.debugging.assert_equal(
        tf.shape(statistics),
        tf.constant(
            [
                class_count,
                HIDDEN_UNITS,
            ],
            dtype=tf.int32,
        ),
        message="RBM CD class statistics have wrong shape",
    )

    tf.debugging.assert_all_finite(
        statistics,
        "RBM CD class statistics contain non-finite values",
    )

    return statistics


def cd1_update_statistics(
    *,
    model: SharedConditionalRBM,
    visible_batch: np.ndarray | tf.Tensor,
    class_ids: np.ndarray | tf.Tensor,
    dataset_id: str,
    experimental_seed: int,
    update_index: int,
) -> RBMCD1Result:
    """Run one governed CD-1 transition and return update statistics.

    This function does not mutate model parameters.
    """

    if not isinstance(
        model,
        SharedConditionalRBM,
    ):
        raise TypeError(
            "model must be a SharedConditionalRBM"
        )

    governed_dataset_id = _validate_dataset_id(
        dataset_id
    )
    governed_seed = _validate_seed(
        experimental_seed
    )
    governed_update_index = _exact_nonnegative_integer(
        update_index,
        name="update_index",
    )

    visible = _validate_visible_batch(
        visible_batch
    )

    labels = validate_class_ids(
        class_ids,
        class_count=model.class_count,
    )

    tf.debugging.assert_equal(
        tf.shape(labels)[0],
        tf.shape(visible)[0],
        message="RBM CD class batch must match visible batch",
    )

    batch_size = int(
        tf.shape(visible)[0].numpy()
    )

    positive_hidden_probabilities = model.hidden_probabilities(
        visible,
        labels,
    )

    hidden_uniform = _uniform_stream(
        dataset_id=governed_dataset_id,
        experimental_seed=governed_seed,
        update_index=governed_update_index,
        stream_tag=_HIDDEN_STREAM_TAG,
        shape=(
            batch_size,
            HIDDEN_UNITS,
        ),
    )

    sampled_hidden_state = model.sample_hidden_from_uniform(
        visible,
        labels,
        hidden_uniform,
    )

    visible_uniform = _uniform_stream(
        dataset_id=governed_dataset_id,
        experimental_seed=governed_seed,
        update_index=governed_update_index,
        stream_tag=_VISIBLE_STREAM_TAG,
        shape=(
            batch_size,
            VISIBLE_DIM,
        ),
    )

    negative_visible_state = model.sample_visible_from_uniform(
        sampled_hidden_state,
        visible_uniform,
    )

    negative_hidden_probabilities = model.hidden_probabilities(
        negative_visible_state,
        labels,
    )

    batch_scale = tf.constant(
        1.0 / float(
            batch_size
        ),
        dtype=STATE_DTYPE,
    )

    positive_W = tf.linalg.matmul(
        visible,
        positive_hidden_probabilities,
        transpose_a=True,
    )

    negative_W = tf.linalg.matmul(
        negative_visible_state,
        negative_hidden_probabilities,
        transpose_a=True,
    )

    delta_W = (
        positive_W
        - negative_W
    ) * batch_scale

    delta_a = (
        tf.reduce_sum(
            visible,
            axis=0,
        )
        - tf.reduce_sum(
            negative_visible_state,
            axis=0,
        )
    ) * batch_scale

    delta_b = (
        tf.reduce_sum(
            positive_hidden_probabilities,
            axis=0,
        )
        - tf.reduce_sum(
            negative_hidden_probabilities,
            axis=0,
        )
    ) * batch_scale

    positive_U = _class_interaction_statistics(
        positive_hidden_probabilities,
        labels,
        class_count=model.class_count,
    )

    negative_U = _class_interaction_statistics(
        negative_hidden_probabilities,
        labels,
        class_count=model.class_count,
    )

    delta_U = (
        positive_U
        - negative_U
    ) * batch_scale

    for name, tensor, expected_shape in (
        (
            "delta_W",
            delta_W,
            (
                VISIBLE_DIM,
                HIDDEN_UNITS,
            ),
        ),
        (
            "delta_a",
            delta_a,
            (
                VISIBLE_DIM,
            ),
        ),
        (
            "delta_b",
            delta_b,
            (
                HIDDEN_UNITS,
            ),
        ),
        (
            "delta_U",
            delta_U,
            (
                model.class_count,
                HIDDEN_UNITS,
            ),
        ),
    ):
        tf.debugging.assert_equal(
            tf.shape(tensor),
            tf.constant(
                expected_shape,
                dtype=tf.int32,
            ),
            message=f"RBM CD {name} has wrong shape",
        )

        tf.debugging.assert_all_finite(
            tensor,
            f"RBM CD {name} contains non-finite values",
        )

    tf.debugging.assert_all_finite(
        positive_hidden_probabilities,
        "RBM CD positive hidden probabilities are non-finite",
    )
    tf.debugging.assert_all_finite(
        sampled_hidden_state,
        "RBM CD sampled hidden state is non-finite",
    )
    tf.debugging.assert_all_finite(
        negative_visible_state,
        "RBM CD negative visible state is non-finite",
    )
    tf.debugging.assert_all_finite(
        negative_hidden_probabilities,
        "RBM CD negative hidden probabilities are non-finite",
    )

    tf.debugging.assert_greater_equal(
        negative_visible_state,
        tf.constant(
            0.0,
            dtype=STATE_DTYPE,
        ),
        message="RBM CD negative visible sample lies below zero",
    )

    tf.debugging.assert_less_equal(
        negative_visible_state,
        tf.constant(
            1.0,
            dtype=STATE_DTYPE,
        ),
        message="RBM CD negative visible sample lies above one",
    )

    tf.debugging.assert_equal(
        sampled_hidden_state,
        tf.round(
            sampled_hidden_state
        ),
        message="RBM CD hidden sample is not binary",
    )

    return RBMCD1Result(
        positive_hidden_probabilities=np.array(
            positive_hidden_probabilities.numpy(),
            dtype=np.float64,
            copy=True,
        ),
        sampled_hidden_state=np.array(
            sampled_hidden_state.numpy(),
            dtype=np.float64,
            copy=True,
        ),
        negative_visible_state=np.array(
            negative_visible_state.numpy(),
            dtype=np.float64,
            copy=True,
        ),
        negative_hidden_probabilities=np.array(
            negative_hidden_probabilities.numpy(),
            dtype=np.float64,
            copy=True,
        ),
        delta_W=np.array(
            delta_W.numpy(),
            dtype=np.float64,
            copy=True,
        ),
        delta_a=np.array(
            delta_a.numpy(),
            dtype=np.float64,
            copy=True,
        ),
        delta_b=np.array(
            delta_b.numpy(),
            dtype=np.float64,
            copy=True,
        ),
        delta_U=np.array(
            delta_U.numpy(),
            dtype=np.float64,
            copy=True,
        ),
    )


__all__ = [
    "CD_K",
    "RBMCD1Result",
    "cd1_update_statistics",
]
