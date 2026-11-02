"""Governed sampling primitives for the Paper 1 revision-2026 RBM.

This module implements only:

- governed synthesis identity;
- model-based chain initialization;
- one fixed-class Gibbs transition.

It intentionally does not select or embed a production burn-in or total chain
length. Those remain blocked pending the governed Slurm chain-length audit.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import operator

import numpy as np
import tensorflow as tf

from trustforge.models.paper01_revision_2026.restricted_boltzmann_checkpoint import (
    RBMPersistedCheckpoint,
)
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


_INITIAL_VISIBLE_STREAM_TAG = 0x494E4954
_TRANSITION_HIDDEN_STREAM_TAG = 0x48494444
_TRANSITION_VISIBLE_STREAM_TAG = 0x56495349


@dataclass(frozen=True)
class RBMSamplingIdentity:
    """Scientific/stochastic identity for one synthetic RBM chain."""

    dataset_id: str
    experimental_seed: int
    class_id: int
    sample_index: int


@dataclass(frozen=True)
class RBMChainState:
    """One authoritative internal state of a governed RBM synthesis chain."""

    visible: np.ndarray
    hidden: np.ndarray
    class_id: int
    transition_count: int


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
            f"{name} must be an integer, not bool"
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


def build_sampling_identity(
    *,
    dataset_id: str,
    experimental_seed: int,
    class_id: int,
    sample_index: int,
    class_count: int,
) -> RBMSamplingIdentity:
    """Validate and construct one governed synthesis-chain identity."""

    governed_dataset = _validate_dataset_id(
        dataset_id
    )

    governed_seed = _validate_seed(
        experimental_seed
    )

    governed_class_count = _exact_nonnegative_integer(
        class_count,
        name="class_count",
    )

    if governed_class_count < 1:
        raise ValueError(
            "class_count must be at least one"
        )

    governed_class = _exact_nonnegative_integer(
        class_id,
        name="class_id",
    )

    if governed_class >= governed_class_count:
        raise ValueError(
            "class_id exceeds governed class universe"
        )

    governed_sample_index = _exact_nonnegative_integer(
        sample_index,
        name="sample_index",
    )

    return RBMSamplingIdentity(
        dataset_id=governed_dataset,
        experimental_seed=governed_seed,
        class_id=governed_class,
        sample_index=governed_sample_index,
    )


def _identity_entropy_words(
    identity: RBMSamplingIdentity,
) -> tuple[int, ...]:
    payload = (
        f"{identity.dataset_id}\0"
        f"{identity.experimental_seed}\0"
        f"{identity.class_id}\0"
        f"{identity.sample_index}"
    ).encode(
        "utf-8"
    )

    digest = hashlib.blake2b(
        payload,
        digest_size=24,
        person=b"TF-RBM-SAMPLE",
    ).digest()

    words = np.frombuffer(
        digest,
        dtype="<u4",
    )

    return tuple(
        int(
            word
        )
        for word in words
    )


def _uniform_stream(
    *,
    identity: RBMSamplingIdentity,
    transition_index: int,
    stream_tag: int,
    shape: tuple[int, int],
) -> np.ndarray:
    """Return one deterministic float64 U[0,1) stream."""

    governed_transition = _exact_nonnegative_integer(
        transition_index,
        name="transition_index",
    )

    low = governed_transition & 0xFFFFFFFF
    high = (
        governed_transition >> 32
    ) & 0xFFFFFFFF

    seed_sequence = np.random.SeedSequence(
        [
            *_identity_entropy_words(
                identity
            ),
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
            "RBM sampling random stream is not float64"
        )

    if not np.all(
        np.isfinite(
            draws
        )
    ):
        raise RuntimeError(
            "RBM sampling random stream contains non-finite values"
        )

    if np.any(
        draws < 0.0
    ) or np.any(
        draws >= 1.0
    ):
        raise RuntimeError(
            "RBM sampling random stream lies outside [0,1)"
        )

    return draws


def _validate_model(
    model: SharedConditionalRBM,
) -> None:
    if not isinstance(
        model,
        SharedConditionalRBM,
    ):
        raise TypeError(
            "model must be SharedConditionalRBM"
        )

    state = model.state_dict()

    if set(
        state
    ) != {
        "W",
        "a",
        "b",
        "U",
    }:
        raise ValueError(
            "RBM model state keys do not exactly match W,a,b,U"
        )

    expected_shapes = {
        "W": (
            VISIBLE_DIM,
            HIDDEN_UNITS,
        ),
        "a": (
            VISIBLE_DIM,
        ),
        "b": (
            HIDDEN_UNITS,
        ),
        "U": (
            model.class_count,
            HIDDEN_UNITS,
        ),
    }

    for name in (
        "W",
        "a",
        "b",
        "U",
    ):
        array = state[
            name
        ]

        if array.dtype != np.float64:
            raise TypeError(
                f"RBM {name} must have exact dtype float64"
            )

        if array.shape != expected_shapes[
            name
        ]:
            raise ValueError(
                f"RBM {name} has invalid shape {array.shape}"
            )

        if not np.isfinite(
            array
        ).all():
            raise ValueError(
                f"RBM {name} contains non-finite values"
            )

    if model.class_count < 1:
        raise ValueError(
            "RBM model must contain at least one class row"
        )


def _validate_checkpoint_identity(
    checkpoint: RBMPersistedCheckpoint,
    identity: RBMSamplingIdentity,
) -> SharedConditionalRBM:
    if not isinstance(
        checkpoint,
        RBMPersistedCheckpoint,
    ):
        raise TypeError(
            "checkpoint must be RBMPersistedCheckpoint"
        )

    metadata = checkpoint.metadata

    if metadata[
        "dataset_id"
    ] != identity.dataset_id:
        raise ValueError(
            "sampling dataset identity does not match checkpoint"
        )

    if metadata[
        "experimental_seed"
    ] != identity.experimental_seed:
        raise ValueError(
            "sampling experimental seed does not match checkpoint"
        )

    if metadata[
        "class_count"
    ] != checkpoint.model.class_count:
        raise ValueError(
            "checkpoint class-count metadata does not match model"
        )

    if identity.class_id >= checkpoint.model.class_count:
        raise ValueError(
            "sampling class identity exceeds checkpoint class universe"
        )

    _validate_model(
        checkpoint.model
    )

    return checkpoint.model


def initialize_chain(
    *,
    checkpoint: RBMPersistedCheckpoint,
    class_id: int,
    sample_index: int,
) -> RBMChainState:
    """Create the governed h0=0, v0~p(v|h0) synthesis state."""

    metadata = checkpoint.metadata

    identity = build_sampling_identity(
        dataset_id=metadata[
            "dataset_id"
        ],
        experimental_seed=metadata[
            "experimental_seed"
        ],
        class_id=class_id,
        sample_index=sample_index,
        class_count=metadata[
            "class_count"
        ],
    )

    model = _validate_checkpoint_identity(
        checkpoint,
        identity,
    )

    hidden = np.zeros(
        (
            1,
            HIDDEN_UNITS,
        ),
        dtype=np.float64,
    )

    visible_uniform = _uniform_stream(
        identity=identity,
        transition_index=0,
        stream_tag=_INITIAL_VISIBLE_STREAM_TAG,
        shape=(
            1,
            VISIBLE_DIM,
        ),
    )

    visible_tensor = model.sample_visible_from_uniform(
        hidden,
        visible_uniform,
    )

    visible_tensor = validate_visible_values(
        visible_tensor
    )

    tf.debugging.assert_type(
        visible_tensor,
        STATE_DTYPE,
        message="RBM initial visible state must be float64",
    )

    visible = np.array(
        visible_tensor.numpy(),
        dtype=np.float64,
        copy=True,
    )

    if visible.shape != (
        1,
        VISIBLE_DIM,
    ):
        raise RuntimeError(
            "RBM initial visible state has invalid shape"
        )

    return RBMChainState(
        visible=visible,
        hidden=hidden,
        class_id=identity.class_id,
        transition_count=0,
    )


def gibbs_transition(
    *,
    checkpoint: RBMPersistedCheckpoint,
    state: RBMChainState,
    sample_index: int,
) -> RBMChainState:
    """Advance one governed fixed-class RBM Gibbs transition."""

    if not isinstance(
        state,
        RBMChainState,
    ):
        raise TypeError(
            "state must be RBMChainState"
        )

    metadata = checkpoint.metadata

    identity = build_sampling_identity(
        dataset_id=metadata[
            "dataset_id"
        ],
        experimental_seed=metadata[
            "experimental_seed"
        ],
        class_id=state.class_id,
        sample_index=sample_index,
        class_count=metadata[
            "class_count"
        ],
    )

    model = _validate_checkpoint_identity(
        checkpoint,
        identity,
    )

    visible = validate_visible_values(
        state.visible
    )

    tf.debugging.assert_rank(
        visible,
        2,
        message="RBM chain visible state must be rank two",
    )

    tf.debugging.assert_equal(
        tf.shape(
            visible
        ),
        tf.constant(
            [
                1,
                VISIBLE_DIM,
            ],
            dtype=tf.int32,
        ),
        message="RBM chain visible state must have shape (1,4800)",
    )

    if (
        not isinstance(
            state.hidden,
            np.ndarray,
        )
        or state.hidden.dtype != np.float64
        or state.hidden.shape
        != (
            1,
            HIDDEN_UNITS,
        )
        or not np.isfinite(
            state.hidden
        ).all()
    ):
        raise ValueError(
            "RBM chain hidden state is invalid"
        )

    if np.any(
        state.hidden < 0.0
    ) or np.any(
        state.hidden > 1.0
    ):
        raise ValueError(
            "RBM chain hidden state lies outside [0,1]"
        )

    governed_transition = _exact_nonnegative_integer(
        state.transition_count,
        name="state.transition_count",
    ) + 1

    class_ids = np.array(
        [
            identity.class_id,
        ],
        dtype=np.int64,
    )

    validate_class_ids(
        class_ids,
        class_count=model.class_count,
    )

    hidden_uniform = _uniform_stream(
        identity=identity,
        transition_index=governed_transition,
        stream_tag=_TRANSITION_HIDDEN_STREAM_TAG,
        shape=(
            1,
            HIDDEN_UNITS,
        ),
    )

    hidden_tensor = model.sample_hidden_from_uniform(
        visible,
        class_ids,
        hidden_uniform,
    )

    visible_uniform = _uniform_stream(
        identity=identity,
        transition_index=governed_transition,
        stream_tag=_TRANSITION_VISIBLE_STREAM_TAG,
        shape=(
            1,
            VISIBLE_DIM,
        ),
    )

    next_visible_tensor = model.sample_visible_from_uniform(
        hidden_tensor,
        visible_uniform,
    )

    next_visible_tensor = validate_visible_values(
        next_visible_tensor
    )

    next_hidden = np.array(
        hidden_tensor.numpy(),
        dtype=np.float64,
        copy=True,
    )

    next_visible = np.array(
        next_visible_tensor.numpy(),
        dtype=np.float64,
        copy=True,
    )

    if next_hidden.shape != (
        1,
        HIDDEN_UNITS,
    ):
        raise RuntimeError(
            "RBM sampled hidden state has invalid shape"
        )

    if next_visible.shape != (
        1,
        VISIBLE_DIM,
    ):
        raise RuntimeError(
            "RBM sampled visible state has invalid shape"
        )

    if not np.isfinite(
        next_hidden
    ).all():
        raise RuntimeError(
            "RBM sampled hidden state contains non-finite values"
        )

    if not np.isfinite(
        next_visible
    ).all():
        raise RuntimeError(
            "RBM sampled visible state contains non-finite values"
        )

    if np.any(
        next_visible < 0.0
    ) or np.any(
        next_visible > 1.0
    ):
        raise RuntimeError(
            "RBM sampled visible state lies outside [0,1]"
        )

    return RBMChainState(
        visible=next_visible,
        hidden=next_hidden,
        class_id=identity.class_id,
        transition_count=governed_transition,
    )


__all__ = [
    "RBMChainState",
    "RBMSamplingIdentity",
    "build_sampling_identity",
    "gibbs_transition",
    "initialize_chain",
]
