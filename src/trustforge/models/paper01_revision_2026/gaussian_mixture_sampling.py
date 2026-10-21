"""Governed sampling for the Paper 1 revision-2026 censored GMM."""

from __future__ import annotations

from dataclasses import dataclass
import operator
from pathlib import Path

import numpy as np

from trustforge.models.paper01_revision_2026.gaussian_mixture_checkpoint import (
    load_gmm_checkpoint,
)
from trustforge.models.paper01_revision_2026.gaussian_mixture_model import (
    CANONICAL_CHANNELS,
    CANONICAL_HEIGHT,
    CANONICAL_WIDTH,
)
from trustforge.models.paper01_revision_2026.gaussian_mixture_observation import (
    CANONICAL_DIM,
    NUM_COMPONENTS,
)
from trustforge.provenance import (
    canonical_json_bytes,
    sha256_bytes,
)


SAMPLING_PURPOSE = "paper01_revision2026_gmm_sampling"
PRIMARY_SYNTHETIC_SAMPLES_PER_CLASS = 2000


@dataclass(frozen=True)
class GMMSamplingResult:
    """One validated governed synthetic sampling result."""

    images: np.ndarray
    dataset_id: str
    class_id: int
    experimental_seed: int
    sampling_identity_sha256: str


def _exact_nonnegative_integer(
    value: int,
    *,
    name: str,
) -> int:
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
    result = _exact_nonnegative_integer(
        value,
        name=name,
    )

    if result == 0:
        raise ValueError(
            f"{name} must be greater than zero"
        )

    return result


def _validate_dataset_id(
    dataset_id: str,
) -> str:
    if not isinstance(dataset_id, str):
        raise TypeError(
            "dataset_id must be a string"
        )

    if not dataset_id:
        raise ValueError(
            "dataset_id must not be empty"
        )

    if dataset_id != dataset_id.strip():
        raise ValueError(
            "dataset_id must not contain surrounding whitespace"
        )

    return dataset_id


def sampling_identity_sha256(
    *,
    dataset_id: str,
    class_id: int,
    experimental_seed: int,
) -> str:
    """Return the governed SHA-256 sampling-substream identity."""

    dataset = _validate_dataset_id(
        dataset_id
    )

    governed_class = _exact_nonnegative_integer(
        class_id,
        name="class_id",
    )

    seed = _exact_nonnegative_integer(
        experimental_seed,
        name="experimental_seed",
    )

    if seed > np.iinfo(np.uint32).max:
        raise ValueError(
            "experimental_seed must lie in [0, 2**32 - 1]"
        )

    payload = {
        "purpose": SAMPLING_PURPOSE,
        "dataset_id": dataset,
        "class_id": governed_class,
        "experimental_seed": seed,
    }

    return sha256_bytes(
        canonical_json_bytes(
            payload
        )
    )


def _sampling_rng(
    *,
    dataset_id: str,
    class_id: int,
    experimental_seed: int,
) -> tuple[np.random.Generator, str]:
    """Construct the governed identity-derived NumPy PRNG."""

    digest_hex = sampling_identity_sha256(
        dataset_id=dataset_id,
        class_id=class_id,
        experimental_seed=experimental_seed,
    )

    digest = bytes.fromhex(
        digest_hex
    )

    entropy = [
        int.from_bytes(
            digest[offset:offset + 4],
            byteorder="big",
            signed=False,
        )
        for offset in range(
            0,
            len(digest),
            4,
        )
    ]

    if len(entropy) != 8:
        raise RuntimeError(
            "GMM sampling SHA-256 digest must produce eight entropy words"
        )

    sequence = np.random.SeedSequence(
        entropy
    )

    return (
        np.random.default_rng(
            sequence
        ),
        digest_hex,
    )


def _validate_weights(
    weights: np.ndarray,
) -> np.ndarray:
    weights = np.asarray(
        weights
    )

    if weights.shape != (
        NUM_COMPONENTS,
    ):
        raise ValueError(
            "GMM sampling weights must have shape (5,)"
        )

    if weights.dtype != np.float64:
        raise TypeError(
            "GMM sampling weights must be float64"
        )

    if not np.isfinite(
        weights
    ).all():
        raise ValueError(
            "GMM sampling weights contain non-finite values"
        )

    if np.any(
        weights <= 0.0
    ):
        raise ValueError(
            "GMM sampling weights must be strictly positive"
        )

    if not np.isclose(
        weights.sum(dtype=np.float64),
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "GMM sampling weights must sum to one"
        )

    return weights


def _validate_sampling_state(
    *,
    means: np.ndarray,
    scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    means = np.asarray(
        means
    )

    scales = np.asarray(
        scales
    )

    expected_shape = (
        NUM_COMPONENTS,
        CANONICAL_DIM,
    )

    if means.shape != expected_shape:
        raise ValueError(
            "GMM sampling means must have shape (5,4800)"
        )

    if scales.shape != expected_shape:
        raise ValueError(
            "GMM sampling scales must have shape (5,4800)"
        )

    if means.dtype != np.float64:
        raise TypeError(
            "GMM sampling means must be float64"
        )

    if scales.dtype != np.float64:
        raise TypeError(
            "GMM sampling scales must be float64"
        )

    if not np.isfinite(
        means
    ).all():
        raise ValueError(
            "GMM sampling means contain non-finite values"
        )

    if not np.isfinite(
        scales
    ).all():
        raise ValueError(
            "GMM sampling scales contain non-finite values"
        )

    if np.any(
        scales <= 0.0
    ):
        raise ValueError(
            "GMM sampling scales must be strictly positive"
        )

    return (
        means,
        scales,
    )


def _censor_latent(
    latent: np.ndarray,
) -> np.ndarray:
    """Apply the governed censored-Gaussian observation map."""

    latent = np.asarray(
        latent
    )

    if latent.dtype != np.float64:
        raise TypeError(
            "GMM latent samples must be float64"
        )

    if not np.isfinite(
        latent
    ).all():
        raise ValueError(
            "GMM latent samples contain non-finite values"
        )

    observed = np.where(
        latent <= 0.0,
        np.float64(0.0),
        np.where(
            latent >= 1.0,
            np.float64(1.0),
            latent,
        ),
    )

    if observed.dtype != np.float64:
        raise RuntimeError(
            "GMM censored observations must remain float64 internally"
        )

    if not np.isfinite(
        observed
    ).all():
        raise RuntimeError(
            "GMM censored observations contain non-finite values"
        )

    if np.any(
        observed < 0.0
    ) or np.any(
        observed > 1.0
    ):
        raise RuntimeError(
            "GMM censored observations escaped canonical range"
        )

    return observed


def sample_gmm_checkpoint(
    checkpoint_directory: Path | str,
    *,
    dataset_id: str,
    class_id: int,
    experimental_seed: int,
    sample_count: int = PRIMARY_SYNTHETIC_SAMPLES_PER_CLASS,
) -> GMMSamplingResult:
    """Generate canonical samples from one validated class checkpoint."""

    count = _exact_positive_integer(
        sample_count,
        name="sample_count",
    )

    dataset = _validate_dataset_id(
        dataset_id
    )

    governed_class = _exact_nonnegative_integer(
        class_id,
        name="class_id",
    )

    seed = _exact_nonnegative_integer(
        experimental_seed,
        name="experimental_seed",
    )

    if seed > np.iinfo(np.uint32).max:
        raise ValueError(
            "experimental_seed must lie in [0, 2**32 - 1]"
        )

    # Governed checkpoint preflight MUST succeed before any random draw.
    checkpoint = load_gmm_checkpoint(
        checkpoint_directory,
        requested_dataset_id=dataset,
        requested_class_id=governed_class,
        requested_experimental_seed=seed,
    )

    model = checkpoint.model

    weights = _validate_weights(
        model.mixture_weights().numpy()
    )

    state = model.state_arrays()

    means = state[
        "means"
    ]

    scales = model.scales().numpy()

    means, scales = _validate_sampling_state(
        means=means,
        scales=scales,
    )

    # Randomness begins only after the full checkpoint/model preflight above.
    rng, identity_digest = _sampling_rng(
        dataset_id=dataset,
        class_id=governed_class,
        experimental_seed=seed,
    )

    component_ids = rng.choice(
        NUM_COMPONENTS,
        size=count,
        replace=True,
        p=weights,
    )

    if component_ids.shape != (
        count,
    ):
        raise RuntimeError(
            "GMM component draw returned unexpected shape"
        )

    if not np.issubdtype(
        component_ids.dtype,
        np.integer,
    ):
        raise RuntimeError(
            "GMM component draw did not return integer identifiers"
        )

    if np.any(
        component_ids < 0
    ) or np.any(
        component_ids >= NUM_COMPONENTS
    ):
        raise RuntimeError(
            "GMM component draw returned out-of-range identifier"
        )

    epsilon = rng.standard_normal(
        size=(
            count,
            CANONICAL_DIM,
        ),
        dtype=np.float64,
    )

    if epsilon.dtype != np.float64:
        raise RuntimeError(
            "GMM standard-normal draw must be float64"
        )

    if not np.isfinite(
        epsilon
    ).all():
        raise RuntimeError(
            "GMM standard-normal draw contains non-finite values"
        )

    latent = (
        means[
            component_ids
        ]
        + scales[
            component_ids
        ]
        * epsilon
    )

    if latent.shape != (
        count,
        CANONICAL_DIM,
    ):
        raise RuntimeError(
            "GMM latent sample matrix has unexpected shape"
        )

    observed = _censor_latent(
        latent
    )

    images = observed.reshape(
        (
            count,
            CANONICAL_HEIGHT,
            CANONICAL_WIDTH,
            CANONICAL_CHANNELS,
        )
    ).astype(
        np.float32,
        copy=False,
    )

    if images.shape != (
        count,
        40,
        40,
        3,
    ):
        raise RuntimeError(
            "GMM canonical output has unexpected shape"
        )

    if images.dtype != np.float32:
        raise RuntimeError(
            "GMM canonical output must be float32"
        )

    if not np.isfinite(
        images
    ).all():
        raise RuntimeError(
            "GMM canonical output contains non-finite values"
        )

    if np.any(
        images < 0.0
    ) or np.any(
        images > 1.0
    ):
        raise RuntimeError(
            "GMM canonical output escaped [0,1]"
        )

    return GMMSamplingResult(
        images=images,
        dataset_id=dataset,
        class_id=governed_class,
        experimental_seed=seed,
        sampling_identity_sha256=identity_digest,
    )


__all__ = [
    "GMMSamplingResult",
    "PRIMARY_SYNTHETIC_SAMPLES_PER_CLASS",
    "SAMPLING_PURPOSE",
    "sample_gmm_checkpoint",
    "sampling_identity_sha256",
]
