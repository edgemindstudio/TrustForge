"""Governed production sampling for the Paper 1 revision-2026 RBM.

This module implements the researcher-approved fixed finite-step policy:

    Q_(theta,y,T) = mu_theta K_(theta,y)^T, with T = 10.

It composes the accepted sampling primitives only:

- one governed initialization per synthetic sample;
- exactly ten complete class-conditioned Gibbs transitions;
- zero separate burn-in;
- canonical float32 NHWC ``(N, 40, 40, 3)`` output in ``[0, 1]``.

It does not train models, publish synthetic pools, select checkpoints, or
claim convergence, mixing sufficiency, or stationary-distribution equivalence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import operator
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from trustforge.models.paper01_revision_2026.restricted_boltzmann_checkpoint import (
    METADATA_FILENAME,
    STATE_FILENAME,
    RBMPersistedCheckpoint,
    load_rbm_checkpoint,
)
from trustforge.models.paper01_revision_2026.restricted_boltzmann_model import (
    CANONICAL_CHANNELS,
    CANONICAL_HEIGHT,
    CANONICAL_WIDTH,
    VISIBLE_DIM,
    SharedConditionalRBM,
)
from trustforge.models.paper01_revision_2026.restricted_boltzmann_sampling import (
    RBMChainState,
    gibbs_transition,
    initialize_chain,
)
from trustforge.provenance import (
    canonical_json_bytes,
    sha256_file,
)


POLICY_SCHEMA = "trustforge.paper01.revision2026.rbm-production-sampling.v1"
PAPER_REVISION = "paper01_revision_2026"
MODEL_FAMILY = "restricted_boltzmann"

PRODUCTION_TRANSITION_COUNT = 10
PRODUCTION_BURN_IN = 0
PRIMARY_SAMPLES_PER_CLASS = 2000

INITIALIZATION = "h0_zero_v0_bounded_exponential_p_v_given_h0"
CLASS_CONDITIONING = "fixed_requested_class"
CHAIN_SEMANTICS = "independent_chain_per_sample"
INTERNAL_DTYPE = "float64"
CANONICAL_DTYPE = "float32"
CANONICAL_SHAPE = (
    CANONICAL_HEIGHT,
    CANONICAL_WIDTH,
    CANONICAL_CHANNELS,
)
CANONICAL_RANGE = (
    0.0,
    1.0,
)
THINNING = "none"
CONTINUATION_CHAINS = False
ADAPTIVE_STOPPING = False

_SHA256_RE = re.compile(
    r"^[0-9a-f]{64}$"
)

_POLICY_FIELDS = frozenset(
    {
        "schema",
        "paper_revision",
        "model_family",
        "dataset_id",
        "experimental_seed",
        "class_mapping",
        "checkpoint_state_sha256",
        "transition_count",
        "burn_in",
        "initialization",
        "class_conditioning",
        "chain_semantics",
        "internal_dtype",
        "canonical_shape",
        "canonical_dtype",
        "canonical_range",
        "primary_samples_per_class",
        "thinning",
        "continuation_chains",
        "adaptive_stopping",
    }
)

_POLICY_REQUIRED_FIELDS = frozenset(
    {
        "dataset_id",
        "experimental_seed",
        "class_mapping",
        "checkpoint_state_sha256",
    }
)


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


def _validate_class_mapping(
    class_mapping: Sequence[str],
) -> tuple[str, ...]:
    if isinstance(
        class_mapping,
        (
            str,
            bytes,
        ),
    ):
        raise TypeError(
            "class_mapping must be an ordered sequence of class labels"
        )

    labels = tuple(
        class_mapping
    )

    if not labels:
        raise ValueError(
            "class_mapping must not be empty"
        )

    normalized: list[str] = []

    for class_id, label in enumerate(
        labels
    ):
        if not isinstance(
            label,
            str,
        ):
            raise TypeError(
                f"class label {class_id} must be a string"
            )

        value = label.strip()

        if not value:
            raise ValueError(
                f"class label {class_id} must not be empty"
            )

        normalized.append(
            value
        )

    if len(
        set(
            normalized
        )
    ) != len(
        normalized
    ):
        raise ValueError(
            "class_mapping contains duplicate labels"
        )

    return tuple(
        normalized
    )


def _validate_sha256(
    value: str,
    *,
    name: str,
) -> str:
    if not isinstance(
        value,
        str,
    ):
        raise TypeError(
            f"{name} must be a string"
        )

    if _SHA256_RE.fullmatch(
        value
    ) is None:
        raise ValueError(
            f"{name} must be 64 lowercase hexadecimal characters"
        )

    return value


@dataclass(
    frozen=True,
)
class RBMProductionSamplingPolicy:
    """Validated RBM production-sampling execution request."""

    dataset_id: str
    experimental_seed: int
    class_mapping: tuple[str, ...]
    checkpoint_state_sha256: str
    schema: str = POLICY_SCHEMA
    paper_revision: str = PAPER_REVISION
    model_family: str = MODEL_FAMILY
    transition_count: int = PRODUCTION_TRANSITION_COUNT
    burn_in: int = PRODUCTION_BURN_IN
    initialization: str = INITIALIZATION
    class_conditioning: str = CLASS_CONDITIONING
    chain_semantics: str = CHAIN_SEMANTICS
    internal_dtype: str = INTERNAL_DTYPE
    canonical_shape: tuple[int, int, int] = CANONICAL_SHAPE
    canonical_dtype: str = CANONICAL_DTYPE
    canonical_range: tuple[float, float] = CANONICAL_RANGE
    primary_samples_per_class: int = PRIMARY_SAMPLES_PER_CLASS
    thinning: str = THINNING
    continuation_chains: bool = CONTINUATION_CHAINS
    adaptive_stopping: bool = ADAPTIVE_STOPPING

    def __post_init__(
        self,
    ) -> None:
        dataset_id = _validate_dataset_id(
            self.dataset_id
        )

        seed = _exact_nonnegative_integer(
            self.experimental_seed,
            name="experimental_seed",
        )

        if seed > np.iinfo(
            np.uint32
        ).max:
            raise ValueError(
                "experimental_seed must lie in [0, 2**32 - 1]"
            )

        class_mapping = _validate_class_mapping(
            self.class_mapping
        )

        checkpoint_hash = _validate_sha256(
            self.checkpoint_state_sha256,
            name="checkpoint_state_sha256",
        )

        if self.schema != POLICY_SCHEMA:
            raise ValueError(
                "RBM production sampling schema mismatch"
            )

        if self.paper_revision != PAPER_REVISION:
            raise ValueError(
                "RBM production sampling paper revision mismatch"
            )

        if self.model_family != MODEL_FAMILY:
            raise ValueError(
                "RBM production sampling model family mismatch"
            )

        transition_count = _exact_nonnegative_integer(
            self.transition_count,
            name="transition_count",
        )

        if transition_count != PRODUCTION_TRANSITION_COUNT:
            raise ValueError(
                "RBM production transition count must equal 10"
            )

        burn_in = _exact_nonnegative_integer(
            self.burn_in,
            name="burn_in",
        )

        if burn_in != PRODUCTION_BURN_IN:
            raise ValueError(
                "RBM production burn-in must be zero"
            )

        if self.initialization != INITIALIZATION:
            raise ValueError(
                "RBM production initialization policy mismatch"
            )

        if self.class_conditioning != CLASS_CONDITIONING:
            raise ValueError(
                "RBM production class-conditioning policy mismatch"
            )

        if self.chain_semantics != CHAIN_SEMANTICS:
            raise ValueError(
                "RBM production chain semantics must be independent "
                "chain per sample"
            )

        if self.internal_dtype != INTERNAL_DTYPE:
            raise ValueError(
                "RBM production internal dtype must be float64"
            )

        if tuple(
            self.canonical_shape
        ) != CANONICAL_SHAPE:
            raise ValueError(
                "RBM production canonical shape must be (40,40,3)"
            )

        if self.canonical_dtype != CANONICAL_DTYPE:
            raise ValueError(
                "RBM production canonical dtype must be float32"
            )

        if tuple(
            self.canonical_range
        ) != CANONICAL_RANGE:
            raise ValueError(
                "RBM production canonical range must be [0,1]"
            )

        samples_per_class = _exact_nonnegative_integer(
            self.primary_samples_per_class,
            name="primary_samples_per_class",
        )

        if samples_per_class != PRIMARY_SAMPLES_PER_CLASS:
            raise ValueError(
                "RBM production primary sample budget must be 2000"
            )

        if self.thinning != THINNING:
            raise ValueError(
                "RBM production thinning must be none"
            )

        if self.continuation_chains is not CONTINUATION_CHAINS:
            raise ValueError(
                "RBM production continuation chains are forbidden"
            )

        if self.adaptive_stopping is not ADAPTIVE_STOPPING:
            raise ValueError(
                "RBM production adaptive stopping is forbidden"
            )

        object.__setattr__(
            self,
            "dataset_id",
            dataset_id,
        )
        object.__setattr__(
            self,
            "experimental_seed",
            seed,
        )
        object.__setattr__(
            self,
            "class_mapping",
            class_mapping,
        )
        object.__setattr__(
            self,
            "checkpoint_state_sha256",
            checkpoint_hash,
        )
        object.__setattr__(
            self,
            "transition_count",
            transition_count,
        )
        object.__setattr__(
            self,
            "burn_in",
            burn_in,
        )
        object.__setattr__(
            self,
            "canonical_shape",
            tuple(
                self.canonical_shape
            ),
        )
        object.__setattr__(
            self,
            "canonical_range",
            tuple(
                self.canonical_range
            ),
        )
        object.__setattr__(
            self,
            "primary_samples_per_class",
            samples_per_class,
        )

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
    ) -> "RBMProductionSamplingPolicy":
        """Validate caller-supplied configuration against approved policy."""

        if not isinstance(
            mapping,
            Mapping,
        ):
            raise TypeError(
                "RBM production sampling configuration must be a mapping"
            )

        supplied = set(
            mapping
        )

        unknown = supplied - _POLICY_FIELDS

        if unknown:
            raise ValueError(
                "RBM production sampling configuration contains unknown "
                f"fields: {sorted(unknown)}"
            )

        missing = _POLICY_REQUIRED_FIELDS - supplied

        if missing:
            raise ValueError(
                "RBM production sampling configuration is missing required "
                f"fields: {sorted(missing)}"
            )

        normalized = dict(
            mapping
        )

        if (
            "class_mapping" in normalized
            and not isinstance(
                normalized[
                    "class_mapping"
                ],
                (
                    str,
                    bytes,
                ),
            )
        ):
            normalized[
                "class_mapping"
            ] = tuple(
                normalized[
                    "class_mapping"
                ]
            )

        if "canonical_shape" in normalized:
            normalized[
                "canonical_shape"
            ] = tuple(
                normalized[
                    "canonical_shape"
                ]
            )

        if "canonical_range" in normalized:
            normalized[
                "canonical_range"
            ] = tuple(
                normalized[
                    "canonical_range"
                ]
            )

        return cls(
            **normalized
        )

    def to_mapping(
        self,
    ) -> dict[str, Any]:
        """Return a canonical JSON-ready policy representation."""

        return {
            "schema": self.schema,
            "paper_revision": self.paper_revision,
            "model_family": self.model_family,
            "dataset_id": self.dataset_id,
            "experimental_seed": self.experimental_seed,
            "class_mapping": list(
                self.class_mapping
            ),
            "checkpoint_state_sha256": self.checkpoint_state_sha256,
            "transition_count": self.transition_count,
            "burn_in": self.burn_in,
            "initialization": self.initialization,
            "class_conditioning": self.class_conditioning,
            "chain_semantics": self.chain_semantics,
            "internal_dtype": self.internal_dtype,
            "canonical_shape": list(
                self.canonical_shape
            ),
            "canonical_dtype": self.canonical_dtype,
            "canonical_range": list(
                self.canonical_range
            ),
            "primary_samples_per_class": self.primary_samples_per_class,
            "thinning": self.thinning,
            "continuation_chains": self.continuation_chains,
            "adaptive_stopping": self.adaptive_stopping,
        }

    def digest(
        self,
    ) -> str:
        """Return the SHA-256 digest of the canonical policy."""

        return hashlib.sha256(
            canonical_json_bytes(
                self.to_mapping()
            )
        ).hexdigest()


@dataclass(
    frozen=True,
)
class RBMProductionSamplingProvenance:
    """Execution provenance for one bounded production-sampling request."""

    dataset_id: str
    experimental_seed: int
    class_id: int
    class_label: str
    sample_indices: tuple[int, ...]
    sample_count: int
    transition_count: int
    burn_in: int
    initialization: str
    class_conditioning: str
    chain_semantics: str
    internal_dtype: str
    canonical_shape: tuple[int, int, int]
    canonical_dtype: str
    checkpoint_state_sha256: str
    policy_digest: str
    schema: str = POLICY_SCHEMA
    paper_revision: str = PAPER_REVISION
    model_family: str = MODEL_FAMILY

    def to_mapping(
        self,
    ) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "paper_revision": self.paper_revision,
            "model_family": self.model_family,
            "dataset_id": self.dataset_id,
            "experimental_seed": self.experimental_seed,
            "class_id": self.class_id,
            "class_label": self.class_label,
            "sample_indices": list(
                self.sample_indices
            ),
            "sample_count": self.sample_count,
            "transition_count": self.transition_count,
            "burn_in": self.burn_in,
            "initialization": self.initialization,
            "class_conditioning": self.class_conditioning,
            "chain_semantics": self.chain_semantics,
            "internal_dtype": self.internal_dtype,
            "canonical_shape": list(
                self.canonical_shape
            ),
            "canonical_dtype": self.canonical_dtype,
            "checkpoint_state_sha256": self.checkpoint_state_sha256,
            "policy_digest": self.policy_digest,
        }


@dataclass(
    frozen=True,
)
class RBMProductionSamplingResult:
    """Validated canonical images and their bounded execution provenance."""

    images: np.ndarray
    provenance: RBMProductionSamplingProvenance


def _validate_sample_indices(
    sample_indices: Sequence[int],
    *,
    limit: int,
) -> tuple[int, ...]:
    if isinstance(
        sample_indices,
        (
            str,
            bytes,
        ),
    ):
        raise TypeError(
            "sample_indices must be an ordered sequence of integers"
        )

    try:
        raw_indices = tuple(
            sample_indices
        )
    except TypeError as exc:
        raise TypeError(
            "sample_indices must be an ordered sequence of integers"
        ) from exc

    if not raw_indices:
        raise ValueError(
            "sample_indices must not be empty"
        )

    validated: list[int] = []
    seen: set[int] = set()

    for raw_index in raw_indices:
        sample_index = _exact_nonnegative_integer(
            raw_index,
            name="sample_index",
        )

        if sample_index >= limit:
            raise ValueError(
                f"sample_index {sample_index} exceeds governed class budget "
                f"[0,{limit - 1}]"
            )

        if sample_index in seen:
            raise ValueError(
                "sample_indices must be unique and independent"
            )

        seen.add(
            sample_index
        )

        validated.append(
            sample_index
        )

    return tuple(
        validated
    )


def visible_vectors_to_canonical_images(
    visible: np.ndarray,
) -> np.ndarray:
    """Convert authoritative float64 RBM vectors to canonical float32 images.

    The established flattening convention is NHWC row-major:
    ``(N, 40, 40, 3) -> (N, 4800)``.  No clipping or silent correction is
    permitted.
    """

    array = np.asarray(
        visible
    )

    if array.dtype != np.float64:
        raise TypeError(
            "RBM visible vectors must have exact dtype float64"
        )

    if array.ndim == 1:
        array = array.reshape(
            1,
            -1,
        )

    if array.ndim != 2 or array.shape[1] != VISIBLE_DIM:
        raise ValueError(
            "RBM visible vectors must have shape (N,4800)"
        )

    if array.shape[0] < 1:
        raise ValueError(
            "RBM visible vectors must contain at least one sample"
        )

    if not np.all(
        np.isfinite(
            array
        )
    ):
        raise ValueError(
            "RBM visible vectors contain non-finite values"
        )

    if np.any(
        array < 0.0
    ) or np.any(
        array > 1.0
    ):
        raise ValueError(
            "RBM visible vectors lie outside [0,1]"
        )

    images = array.reshape(
        (
            array.shape[0],
            *CANONICAL_SHAPE,
        )
    ).astype(
        np.float32,
        copy=False,
    )

    if images.dtype != np.float32:
        raise RuntimeError(
            "RBM canonical output is not float32"
        )

    if images.shape != (
        array.shape[0],
        *CANONICAL_SHAPE,
    ):
        raise RuntimeError(
            "RBM canonical output has invalid shape"
        )

    if not np.all(
        np.isfinite(
            images
        )
    ):
        raise RuntimeError(
            "RBM canonical output contains non-finite values"
        )

    if np.any(
        images < 0.0
    ) or np.any(
        images > 1.0
    ):
        raise RuntimeError(
            "RBM canonical output lies outside [0,1]"
        )

    return images


def _checkpoint_class_mapping(
    checkpoint: RBMPersistedCheckpoint,
) -> tuple[str, ...]:
    metadata_mapping = checkpoint.metadata[
        "class_mapping"
    ]

    if not isinstance(
        metadata_mapping,
        list,
    ):
        raise ValueError(
            "RBM checkpoint class_mapping must be a list"
        )

    labels: list[str] = []

    for expected_id, entry in enumerate(
        metadata_mapping
    ):
        if (
            not isinstance(
                entry,
                dict,
            )
            or set(
                entry
            )
            != {
                "class_id",
                "class_label",
            }
        ):
            raise ValueError(
                "RBM checkpoint class_mapping entry has invalid schema"
            )

        if entry[
            "class_id"
        ] != expected_id:
            raise ValueError(
                "RBM checkpoint class IDs must be contiguous and ordered"
            )

        labels.append(
            entry[
                "class_label"
            ]
        )

    return _validate_class_mapping(
        labels
    )


def _governed_checkpoint_directory(
    checkpoint: RBMPersistedCheckpoint,
) -> Path:
    """Require state and metadata paths to be the governed directory pair."""

    directory = Path(
        checkpoint.directory
    ).expanduser()

    if checkpoint.state_path != directory / STATE_FILENAME:
        raise ValueError(
            "RBM checkpoint state path is not the governed checkpoint path"
        )

    if checkpoint.metadata_path != directory / METADATA_FILENAME:
        raise ValueError(
            "RBM checkpoint metadata path is not the governed checkpoint path"
        )

    return directory


def validate_production_sampling_policy(
    *,
    checkpoint: RBMPersistedCheckpoint,
    policy: RBMProductionSamplingPolicy,
) -> None:
    """Fail closed unless policy, checkpoint and class mapping agree."""

    if not isinstance(
        checkpoint,
        RBMPersistedCheckpoint,
    ):
        raise TypeError(
            "checkpoint must be RBMPersistedCheckpoint"
        )

    if not isinstance(
        checkpoint.model,
        SharedConditionalRBM,
    ):
        raise TypeError(
            "checkpoint model must be SharedConditionalRBM"
        )

    if not isinstance(
        policy,
        RBMProductionSamplingPolicy,
    ):
        raise TypeError(
            "policy must be RBMProductionSamplingPolicy"
        )

    _governed_checkpoint_directory(
        checkpoint
    )

    metadata = checkpoint.metadata

    if metadata.get(
        "paper_revision"
    ) != PAPER_REVISION:
        raise ValueError(
            "RBM checkpoint paper revision does not match production policy"
        )

    if metadata.get(
        "model_family"
    ) != MODEL_FAMILY:
        raise ValueError(
            "RBM checkpoint model family does not match production policy"
        )

    if metadata.get(
        "dataset_id"
    ) != policy.dataset_id:
        raise ValueError(
            "RBM production dataset identity does not match checkpoint"
        )

    if metadata.get(
        "experimental_seed"
    ) != policy.experimental_seed:
        raise ValueError(
            "RBM production experimental seed does not match checkpoint"
        )

    labels = _checkpoint_class_mapping(
        checkpoint
    )

    if policy.class_mapping != labels:
        raise ValueError(
            "RBM production class mapping does not match checkpoint"
        )

    if metadata.get(
        "class_count"
    ) != len(
        labels
    ):
        raise ValueError(
            "RBM checkpoint class count does not match class mapping"
        )

    if checkpoint.model.class_count != len(
        labels
    ):
        raise ValueError(
            "RBM checkpoint model class count does not match class mapping"
        )

    declared_hash = metadata.get(
        "state_sha256"
    )

    if declared_hash != policy.checkpoint_state_sha256:
        raise ValueError(
            "RBM production checkpoint state hash does not match policy"
        )

    observed_hash = sha256_file(
        checkpoint.state_path
    )

    if observed_hash != policy.checkpoint_state_sha256:
        raise ValueError(
            "RBM production checkpoint state file does not match policy"
        )


def _restore_verified_checkpoint(
    *,
    checkpoint: RBMPersistedCheckpoint,
    policy: RBMProductionSamplingPolicy,
) -> RBMPersistedCheckpoint:
    """Restore the persisted state and bind it to the in-memory model.

    Sampling must consume parameters restored from the verified checkpoint
    file, not an arbitrary in-memory model carried by the wrapper object.
    """

    directory = _governed_checkpoint_directory(
        checkpoint
    )

    restored = load_rbm_checkpoint(
        directory,
        requested_dataset_id=policy.dataset_id,
        requested_experimental_seed=policy.experimental_seed,
        requested_class_mapping=policy.class_mapping,
    )

    if restored.metadata.get(
        "state_sha256"
    ) != policy.checkpoint_state_sha256:
        raise ValueError(
            "restored RBM checkpoint state hash does not match policy"
        )

    in_memory = checkpoint.model.state_dict()
    persisted = restored.model.state_dict()

    expected_keys = {
        "W",
        "a",
        "b",
        "U",
    }

    if set(
        in_memory
    ) != expected_keys:
        raise ValueError(
            "RBM in-memory model state keys do not exactly match W,a,b,U"
        )

    if set(
        persisted
    ) != expected_keys:
        raise ValueError(
            "restored RBM model state keys do not exactly match W,a,b,U"
        )

    for key in (
        "W",
        "a",
        "b",
        "U",
    ):
        in_memory_array = in_memory[
            key
        ]
        persisted_array = persisted[
            key
        ]

        if (
            in_memory_array.dtype != np.float64
            or persisted_array.dtype != np.float64
        ):
            raise TypeError(
                f"RBM model parameter {key} must be float64"
            )

        if in_memory_array.shape != persisted_array.shape:
            raise ValueError(
                f"RBM model parameter {key} has a shape mismatch"
            )

        if not np.array_equal(
            in_memory_array,
            persisted_array,
        ):
            raise ValueError(
                "RBM in-memory model parameters do not match the verified "
                "checkpoint state"
            )

    return restored


def sample_production_images(
    *,
    checkpoint: RBMPersistedCheckpoint,
    policy: RBMProductionSamplingPolicy,
    class_id: int,
    sample_indices: Sequence[int],
) -> RBMProductionSamplingResult:
    """Generate canonical RBM images under the approved T=10 policy.

    Every requested sample receives one fresh governed initialization and
    exactly ten complete Gibbs transitions.  No chain state is reused across
    samples.
    """

    validate_production_sampling_policy(
        checkpoint=checkpoint,
        policy=policy,
    )

    verified_checkpoint = _restore_verified_checkpoint(
        checkpoint=checkpoint,
        policy=policy,
    )

    governed_class_id = _exact_nonnegative_integer(
        class_id,
        name="class_id",
    )

    if governed_class_id >= len(
        policy.class_mapping
    ):
        raise ValueError(
            "requested class identity exceeds governed class universe"
        )

    indices = _validate_sample_indices(
        sample_indices,
        limit=policy.primary_samples_per_class,
    )

    visible_states: list[np.ndarray] = []

    for sample_index in indices:
        state: RBMChainState = initialize_chain(
            checkpoint=verified_checkpoint,
            class_id=governed_class_id,
            sample_index=sample_index,
        )

        if state.class_id != governed_class_id:
            raise RuntimeError(
                "RBM production initialization changed requested class"
            )

        if state.transition_count != 0:
            raise RuntimeError(
                "RBM production chain did not start at transition zero"
            )

        for _ in range(
            policy.transition_count
        ):
            state = gibbs_transition(
                checkpoint=verified_checkpoint,
                state=state,
                sample_index=sample_index,
            )

        if state.class_id != governed_class_id:
            raise RuntimeError(
                "RBM production chain changed requested class"
            )

        if state.transition_count != policy.transition_count:
            raise RuntimeError(
                "RBM production chain did not execute exactly ten transitions"
            )

        if state.visible.dtype != np.float64:
            raise RuntimeError(
                "RBM production visible state is not float64"
            )

        visible_states.append(
            state.visible
        )

    visible = np.concatenate(
        visible_states,
        axis=0,
    )

    images = visible_vectors_to_canonical_images(
        visible
    )

    provenance = RBMProductionSamplingProvenance(
        dataset_id=policy.dataset_id,
        experimental_seed=policy.experimental_seed,
        class_id=governed_class_id,
        class_label=policy.class_mapping[
            governed_class_id
        ],
        sample_indices=indices,
        sample_count=len(
            indices
        ),
        transition_count=policy.transition_count,
        burn_in=policy.burn_in,
        initialization=policy.initialization,
        class_conditioning=policy.class_conditioning,
        chain_semantics=policy.chain_semantics,
        internal_dtype=policy.internal_dtype,
        canonical_shape=policy.canonical_shape,
        canonical_dtype=policy.canonical_dtype,
        checkpoint_state_sha256=policy.checkpoint_state_sha256,
        policy_digest=policy.digest(),
    )

    return RBMProductionSamplingResult(
        images=images,
        provenance=provenance,
    )


__all__ = [
    "ADAPTIVE_STOPPING",
    "CANONICAL_DTYPE",
    "CANONICAL_RANGE",
    "CANONICAL_SHAPE",
    "CHAIN_SEMANTICS",
    "CLASS_CONDITIONING",
    "CONTINUATION_CHAINS",
    "INITIALIZATION",
    "INTERNAL_DTYPE",
    "MODEL_FAMILY",
    "PAPER_REVISION",
    "POLICY_SCHEMA",
    "PRIMARY_SAMPLES_PER_CLASS",
    "PRODUCTION_BURN_IN",
    "PRODUCTION_TRANSITION_COUNT",
    "RBMProductionSamplingPolicy",
    "RBMProductionSamplingProvenance",
    "RBMProductionSamplingResult",
    "THINNING",
    "sample_production_images",
    "validate_production_sampling_policy",
    "visible_vectors_to_canonical_images",
]
