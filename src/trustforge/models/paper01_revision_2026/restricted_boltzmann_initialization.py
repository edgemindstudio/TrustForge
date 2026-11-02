"""Governed initialization for the Paper 1 revision-2026 RBM.

Selected initialization:

- W_ij ~ Normal(0, 0.01^2), float64;
- visible bias a_i = a0 for all i;
- a0 uniquely solves E[V | theta=a0] = m_train;
- b = 0;
- U = 0.

The authoritative production m_train must come from the complete governed
training split.  Representative audit means are not production constants.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator

import numpy as np
import tensorflow as tf

from trustforge.models.paper01_revision_2026.restricted_boltzmann_model import (
    HIDDEN_UNITS,
    VISIBLE_DIM,
    SharedConditionalRBM,
)
from trustforge.models.paper01_revision_2026.restricted_boltzmann_observation import (
    mean as visible_conditional_mean,
)


INITIAL_WEIGHT_STD = 0.01

ROOT_LOWER_INITIAL = -1.0
ROOT_UPPER_INITIAL = 1.0

ROOT_EXPANSION_FACTOR = 2.0
ROOT_MAX_EXPANSIONS = 64
ROOT_MAX_BISECTION_STEPS = 256

ROOT_ABS_TOL = 1.0e-12
ROOT_MEAN_TOL = 1.0e-12


@dataclass(frozen=True)
class RBMInitializationDiagnostics:
    """Deterministic evidence describing one governed initialization."""

    class_count: int
    experimental_seed: int
    training_scalar_mean: float
    visible_bias_scalar: float
    weight_std: float


def _exact_nonnegative_integer(
    value: int,
    *,
    name: str,
) -> int:
    """Return one governed exact non-negative integer."""

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

    if result < 0:
        raise ValueError(
            f"{name} must be non-negative"
        )

    return result


def _exact_positive_integer(
    value: int,
    *,
    name: str,
) -> int:
    """Return one governed exact positive integer."""

    result = _exact_nonnegative_integer(
        value,
        name=name,
    )

    if result == 0:
        raise ValueError(
            f"{name} must be greater than zero"
        )

    return result


def _validate_training_scalar_mean(
    value: float,
) -> float:
    """Validate one authoritative full-training scalar mean."""

    result = float(value)

    if not math.isfinite(result):
        raise ValueError(
            "training_scalar_mean must be finite"
        )

    if not (
        0.0
        < result
        < 1.0
    ):
        raise ValueError(
            "training_scalar_mean must lie strictly inside (0,1)"
        )

    return result


def visible_mean_from_scalar_theta(
    theta: float,
) -> float:
    """Evaluate the governed bounded-exponential mean for one scalar theta."""

    theta_value = float(theta)

    if not math.isfinite(theta_value):
        raise ValueError(
            "theta must be finite"
        )

    result = visible_conditional_mean(
        tf.constant(
            theta_value,
            dtype=tf.float64,
        )
    )

    scalar = float(
        result.numpy()
    )

    if not math.isfinite(scalar):
        raise ValueError(
            "bounded-exponential mean evaluation is non-finite"
        )

    return scalar


def solve_visible_bias_scalar(
    training_scalar_mean: float,
) -> float:
    """Invert E[V|theta] = m_train by deterministic bracketed bisection."""

    target = _validate_training_scalar_mean(
        training_scalar_mean
    )

    if target == 0.5:
        return 0.0

    lower = ROOT_LOWER_INITIAL
    upper = ROOT_UPPER_INITIAL

    lower_mean = visible_mean_from_scalar_theta(
        lower
    )
    upper_mean = visible_mean_from_scalar_theta(
        upper
    )

    expansions = 0

    while target < lower_mean:
        upper = lower
        upper_mean = lower_mean

        lower *= ROOT_EXPANSION_FACTOR
        lower_mean = visible_mean_from_scalar_theta(
            lower
        )

        expansions += 1
        if expansions > ROOT_MAX_EXPANSIONS:
            raise RuntimeError(
                "failed to bracket visible-bias root on negative side"
            )

    while target > upper_mean:
        lower = upper
        lower_mean = upper_mean

        upper *= ROOT_EXPANSION_FACTOR
        upper_mean = visible_mean_from_scalar_theta(
            upper
        )

        expansions += 1
        if expansions > ROOT_MAX_EXPANSIONS:
            raise RuntimeError(
                "failed to bracket visible-bias root on positive side"
            )

    if not (
        lower_mean
        <= target
        <= upper_mean
    ):
        raise RuntimeError(
            "visible-bias target was not bracketed"
        )

    midpoint = 0.5 * (
        lower + upper
    )

    for _ in range(
        ROOT_MAX_BISECTION_STEPS
    ):
        midpoint = 0.5 * (
            lower + upper
        )

        midpoint_mean = visible_mean_from_scalar_theta(
            midpoint
        )

        if abs(
            midpoint_mean - target
        ) <= ROOT_MEAN_TOL:
            break

        if abs(
            upper - lower
        ) <= ROOT_ABS_TOL:
            break

        if midpoint_mean < target:
            lower = midpoint
        else:
            upper = midpoint

    recovered = visible_mean_from_scalar_theta(
        midpoint
    )

    if abs(
        recovered - target
    ) > 5.0e-11:
        raise RuntimeError(
            "visible-bias root inversion did not meet governed tolerance"
        )

    if not math.isfinite(midpoint):
        raise RuntimeError(
            "visible-bias root inversion produced non-finite result"
        )

    return float(
        midpoint
    )


def initialize_shared_conditional_rbm(
    *,
    class_count: int,
    experimental_seed: int,
    training_scalar_mean: float,
) -> tuple[
    SharedConditionalRBM,
    RBMInitializationDiagnostics,
]:
    """Construct one governed shared class-conditional RBM initialization."""

    governed_class_count = _exact_positive_integer(
        class_count,
        name="class_count",
    )
    governed_seed = _exact_nonnegative_integer(
        experimental_seed,
        name="experimental_seed",
    )
    governed_mean = _validate_training_scalar_mean(
        training_scalar_mean
    )

    if governed_seed > np.iinfo(
        np.uint32
    ).max:
        raise ValueError(
            "experimental_seed must lie in [0, 2**32 - 1]"
        )

    visible_bias_scalar = solve_visible_bias_scalar(
        governed_mean
    )

    rng = np.random.default_rng(
        governed_seed
    )

    W = rng.normal(
        loc=0.0,
        scale=INITIAL_WEIGHT_STD,
        size=(
            VISIBLE_DIM,
            HIDDEN_UNITS,
        ),
    ).astype(
        np.float64,
        copy=False,
    )

    a = np.full(
        (VISIBLE_DIM,),
        visible_bias_scalar,
        dtype=np.float64,
    )

    b = np.zeros(
        (HIDDEN_UNITS,),
        dtype=np.float64,
    )

    U = np.zeros(
        (
            governed_class_count,
            HIDDEN_UNITS,
        ),
        dtype=np.float64,
    )

    model = SharedConditionalRBM(
        W=W,
        a=a,
        b=b,
        U=U,
    )

    diagnostics = RBMInitializationDiagnostics(
        class_count=governed_class_count,
        experimental_seed=governed_seed,
        training_scalar_mean=governed_mean,
        visible_bias_scalar=visible_bias_scalar,
        weight_std=INITIAL_WEIGHT_STD,
    )

    return model, diagnostics


__all__ = [
    "INITIAL_WEIGHT_STD",
    "RBMInitializationDiagnostics",
    "full_training_scalar_mean",
    "initialize_shared_conditional_rbm",
    "solve_visible_bias_scalar",
    "visible_mean_from_scalar_theta",
]


class CanonicalTrainingAdapterProtocol:
    """Structural documentation for the required canonical training adapter.

    Compatible adapters expose:

        split_size(split: str) -> int

        canonical_batch(
            split: str,
            logical_positions: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]
    """


def full_training_scalar_mean(
    adapter,
    *,
    batch_size: int = 256,
) -> float:
    """Compute the authoritative full-training canonical scalar mean.

    The census:

    - uses only split="train";
    - visits every governed training logical position exactly once;
    - requests canonical float32 observations from the adapter;
    - accumulates scalar sum and scalar count in float64;
    - never materializes the full dataset in memory.
    """

    governed_batch_size = _exact_positive_integer(
        batch_size,
        name="batch_size",
    )

    if not hasattr(
        adapter,
        "split_size",
    ):
        raise TypeError(
            "adapter must provide split_size(split)"
        )

    if not hasattr(
        adapter,
        "canonical_batch",
    ):
        raise TypeError(
            "adapter must provide canonical_batch(split, logical_positions)"
        )

    training_count = _exact_positive_integer(
        adapter.split_size(
            "train"
        ),
        name="training split size",
    )

    scalar_sum = np.float64(
        0.0
    )
    scalar_count = 0

    for start in range(
        0,
        training_count,
        governed_batch_size,
    ):
        stop = min(
            start + governed_batch_size,
            training_count,
        )

        logical_positions = np.arange(
            start,
            stop,
            dtype=np.int64,
        )

        images, _ = adapter.canonical_batch(
            "train",
            logical_positions,
        )

        array = np.asarray(
            images
        )

        if array.dtype != np.float32:
            raise TypeError(
                "canonical training census requires float32 images"
            )

        if array.ndim != 4:
            raise ValueError(
                "canonical training census images must have rank four"
            )

        if array.shape[0] != logical_positions.size:
            raise ValueError(
                "canonical training census batch size mismatch"
            )

        if tuple(
            array.shape[1:]
        ) != (
            40,
            40,
            3,
        ):
            raise ValueError(
                "canonical training census images must have shape "
                "(B,40,40,3)"
            )

        if not np.all(
            np.isfinite(array)
        ):
            raise ValueError(
                "canonical training census contains non-finite values"
            )

        if np.any(
            array < 0.0
        ) or np.any(
            array > 1.0
        ):
            raise ValueError(
                "canonical training census contains values outside [0,1]"
            )

        batch64 = array.astype(
            np.float64,
            copy=False,
        )

        scalar_sum += np.sum(
            batch64,
            dtype=np.float64,
        )

        scalar_count += int(
            batch64.size
        )

    expected_scalar_count = (
        training_count
        * VISIBLE_DIM
    )

    if scalar_count != expected_scalar_count:
        raise RuntimeError(
            "canonical training census scalar-count mismatch"
        )

    if scalar_count <= 0:
        raise RuntimeError(
            "canonical training census produced no scalar observations"
        )

    training_mean = float(
        scalar_sum
        / np.float64(
            scalar_count
        )
    )

    return _validate_training_scalar_mean(
        training_mean
    )
