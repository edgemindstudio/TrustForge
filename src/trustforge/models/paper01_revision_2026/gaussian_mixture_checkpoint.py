"""Governed checkpoint persistence for the Paper 1 revision-2026 GMM.

The authoritative class-model checkpoint consists of exactly:

    model_state.npz
    model_metadata.json

Publication is state-first and metadata-last. Metadata is therefore the final
pair-completion marker.

This module deliberately does not generate synthetic samples.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import operator
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

import numpy as np

from trustforge.models.paper01_revision_2026.gaussian_mixture_model import (
    CensoredDiagonalGaussianMixture,
)
from trustforge.models.paper01_revision_2026.gaussian_mixture_observation import (
    CANONICAL_DIM,
    NUM_COMPONENTS,
)
from trustforge.models.paper01_revision_2026.gaussian_mixture_training import (
    GMMClassTrainingResult,
)
from trustforge.provenance import (
    canonical_json_bytes,
    sha256_file,
)


STATE_FILENAME = "model_state.npz"
METADATA_FILENAME = "model_metadata.json"

SCHEMA_VERSION = "1.0"
SCHEMA_TYPE = "paper01_revision2026_gmm_class_model"

PARAMETER_DTYPE = "float64"
OBSERVATION_MODEL = "censored_diagonal_gaussian_mixture"

STATE_KEYS = (
    "logits",
    "means",
    "raw_scales",
)

SHA256_RE = re.compile(
    r"^[0-9a-f]{64}$"
)
GIT_COMMIT_RE = re.compile(
    r"^[0-9a-f]{40}$"
)
EXECUTION_EVIDENCE_KEYS = {
    "schema",
    "source_freeze_sha256",
    "expected_source_commit",
    "historical_git_baseline",
    "implementation_parent_revision",
    "source_files_sha256",
    "governed_experimental_seeds",
    "dataset_adapter",
    "dataset_source_paths",
}
GOVERNED_SEEDS = list(range(42, 49))


@dataclass(frozen=True)
class GMMPersistedCheckpoint:
    """Validated authoritative checkpoint pair."""

    directory: Path
    state_path: Path
    metadata_path: Path
    metadata: dict[str, Any]
    model: CensoredDiagonalGaussianMixture


def _exact_integer(
    value: int,
    *,
    name: str,
    minimum: int = 0,
) -> int:
    """Return one governed exact integer."""

    if isinstance(
        value,
        bool,
    ):
        raise TypeError(
            f"{name} must be an integer, not bool"
        )

    try:
        converted = operator.index(
            value
        )
    except TypeError as exc:
        raise TypeError(
            f"{name} must be an exact integer"
        ) from exc

    if converted < minimum:
        raise ValueError(
            f"{name} must be >= {minimum}"
        )

    return converted


def _validate_dataset_id(
    dataset_id: str,
) -> str:
    """Validate one explicit scientific dataset identity."""

    if not isinstance(
        dataset_id,
        str,
    ):
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


def _validate_state_arrays(
    state: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Validate authoritative numerical state exactly."""

    if set(
        state
    ) != set(
        STATE_KEYS
    ):
        raise ValueError(
            "GMM checkpoint state must contain exactly "
            "logits, means, and raw_scales"
        )

    logits = np.asarray(
        state[
            "logits"
        ]
    )

    means = np.asarray(
        state[
            "means"
        ]
    )

    raw_scales = np.asarray(
        state[
            "raw_scales"
        ]
    )

    expected = {
        "logits": (
            logits,
            (
                NUM_COMPONENTS,
            ),
        ),
        "means": (
            means,
            (
                NUM_COMPONENTS,
                CANONICAL_DIM,
            ),
        ),
        "raw_scales": (
            raw_scales,
            (
                NUM_COMPONENTS,
                CANONICAL_DIM,
            ),
        ),
    }

    validated: dict[
        str,
        np.ndarray
    ] = {}

    for key in STATE_KEYS:
        array, shape = expected[
            key
        ]

        if array.shape != shape:
            raise ValueError(
                f"{key} has invalid shape: "
                f"expected {shape}, got {array.shape}"
            )

        if array.dtype != np.float64:
            raise TypeError(
                f"{key} must be float64"
            )

        if not np.isfinite(
            array
        ).all():
            raise ValueError(
                f"{key} contains non-finite values"
            )

        validated[
            key
        ] = array.copy()

    # Constructing the governed model performs semantic validation of:
    # transformed scales, mixture weights, and state structure.
    model = CensoredDiagonalGaussianMixture(
        logits=validated[
            "logits"
        ],
        means=validated[
            "means"
        ],
        raw_scales=validated[
            "raw_scales"
        ],
    )

    model.state_arrays()
    model.scales()
    model.mixture_weights()

    return validated


def _validation_history_json(
    result: GMMClassTrainingResult,
) -> list[dict[str, Any]]:
    """Return validated ordered checkpoint-selection history."""

    history: list[
        dict[str, Any]
    ] = []

    previous_step = -1

    for checkpoint in result.validation_history:
        step = _exact_integer(
            checkpoint.step,
            name="validation history step",
            minimum=0,
        )

        nll = float(
            checkpoint.validation_nll
        )

        if not np.isfinite(
            nll
        ):
            raise ValueError(
                "validation history contains non-finite NLL"
            )

        if step <= previous_step:
            raise ValueError(
                "validation history steps must be strictly increasing"
            )

        history.append(
            {
                "step": step,
                "validation_nll": nll,
            }
        )

        previous_step = step

    if not history:
        raise ValueError(
            "validation history must not be empty"
        )

    if history[
        0
    ][
        "step"
    ] != 0:
        raise ValueError(
            "validation history must begin at step 0"
        )

    if tuple(
        item[
            "step"
        ]
        for item in history
    ) != result.budget.validation_steps:
        raise ValueError(
            "validation history does not match governed budget schedule"
        )

    selected_step = _exact_integer(
        result.selected_step,
        name="selected_step",
        minimum=0,
    )

    selected_nll = float(
        result.selected_validation_nll
    )

    if not np.isfinite(
        selected_nll
    ):
        raise ValueError(
            "selected validation NLL must be finite"
        )

    matching = [
        item
        for item in history
        if (
            item[
                "step"
            ]
            == selected_step
            and item[
                "validation_nll"
            ]
            == selected_nll
        )
    ]

    if len(
        matching
    ) != 1:
        raise ValueError(
            "selected checkpoint does not exactly correspond "
            "to validation history"
        )

    minimum_nll = min(
        item[
            "validation_nll"
        ]
        for item in history
    )

    earliest_minimum_step = min(
        item[
            "step"
        ]
        for item in history
        if item[
            "validation_nll"
        ]
        == minimum_nll
    )

    if (
        selected_nll
        != minimum_nll
        or selected_step
        != earliest_minimum_step
    ):
        raise ValueError(
            "selected checkpoint violates lowest-NLL/earliest-step policy"
        )

    return history


def _validate_execution_evidence(
    evidence: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Validate optional source/config evidence for future GMM artifacts."""
    if evidence is None:
        return None
    if (
        not isinstance(evidence, Mapping)
        or set(evidence) != EXECUTION_EVIDENCE_KEYS
    ):
        raise ValueError("invalid GMM execution evidence schema")
    if (
        not isinstance(evidence["schema"], str)
        or not evidence["schema"].startswith(
            "trustforge.paper01.gmm.execution-binding."
        )
    ):
        raise ValueError("invalid GMM execution evidence binding schema")
    freeze_digest = evidence["source_freeze_sha256"]
    if (
        not isinstance(freeze_digest, str)
        or SHA256_RE.fullmatch(freeze_digest) is None
    ):
        raise ValueError("invalid GMM execution evidence digest")
    for key in (
        "expected_source_commit",
        "historical_git_baseline",
        "implementation_parent_revision",
    ):
        value = evidence[key]
        if not isinstance(value, str) or GIT_COMMIT_RE.fullmatch(value) is None:
            raise ValueError("invalid GMM execution evidence commit")
    files = evidence["source_files_sha256"]
    if not isinstance(files, Mapping) or not files:
        raise ValueError("invalid GMM execution source-file evidence")
    for name, value in files.items():
        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(value, str)
            or SHA256_RE.fullmatch(value) is None
        ):
            raise ValueError("invalid GMM execution source-file evidence")
    if evidence["governed_experimental_seeds"] != GOVERNED_SEEDS:
        raise ValueError("GMM execution evidence seed policy mismatch")
    if (
        not isinstance(evidence["dataset_adapter"], str)
        or not evidence["dataset_adapter"].strip()
    ):
        raise ValueError("invalid GMM execution dataset adapter")
    paths = evidence["dataset_source_paths"]
    if not isinstance(paths, Mapping) or not paths:
        raise ValueError("invalid GMM execution dataset source evidence")
    for name, value in paths.items():
        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(value, str)
            or not value.strip()
        ):
            raise ValueError("invalid GMM execution dataset source evidence")
    return dict(evidence)


def _build_metadata(
    *,
    dataset_id: str,
    class_id: int,
    experimental_seed: int,
    state_sha256: str,
    training_result: GMMClassTrainingResult,
    validation_class_count: int,
    execution_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct canonical class-model metadata."""

    dataset = _validate_dataset_id(
        dataset_id
    )

    governed_class = _exact_integer(
        class_id,
        name="class_id",
        minimum=0,
    )

    seed = _exact_integer(
        experimental_seed,
        name="experimental_seed",
        minimum=0,
    )

    if seed > np.iinfo(
        np.uint32
    ).max:
        raise ValueError(
            "experimental_seed must lie in [0, 2**32 - 1]"
        )

    validation_count = _exact_integer(
        validation_class_count,
        name="validation_class_count",
        minimum=1,
    )

    if not isinstance(
        state_sha256,
        str,
    ) or SHA256_RE.fullmatch(
        state_sha256
    ) is None:
        raise ValueError(
            "state_sha256 must be 64 lowercase hexadecimal characters"
        )

    history = _validation_history_json(
        training_result
    )

    metadata: dict[
        str,
        Any
    ] = {
        "schema_version": SCHEMA_VERSION,
        "schema_type": SCHEMA_TYPE,
        "dataset_id": dataset,
        "class_id": governed_class,
        "experimental_seed": seed,
        "state_file": STATE_FILENAME,
        "state_sha256": state_sha256,
        "components": NUM_COMPONENTS,
        "dimension": CANONICAL_DIM,
        "parameter_dtype": PARAMETER_DTYPE,
        "observation_model": OBSERVATION_MODEL,
        "selected_step": int(
            training_result.selected_step
        ),
        "selected_validation_nll": float(
            training_result.selected_validation_nll
        ),
        "train_class_count": int(
            training_result.budget.class_count
        ),
        "validation_class_count": validation_count,
        "maximum_updates": int(
            training_result.budget.max_updates
        ),
        "validation_interval_updates": int(
            training_result.budget.validation_interval
        ),
        "validation_history": history,
    }
    validated_evidence = _validate_execution_evidence(execution_evidence)
    if validated_evidence is not None:
        metadata["execution_evidence"] = validated_evidence

    # Canonical serialization itself is also a validation gate:
    # NaN, Infinity, and non-JSON values fail here.
    canonical_json_bytes(
        metadata
    )

    return metadata


def _fsync_file(
    path: Path,
) -> None:
    """fsync one concrete file."""

    with path.open(
        "rb"
    ) as handle:
        os.fsync(
            handle.fileno()
        )


def _fsync_directory(
    path: Path,
) -> None:
    """fsync one directory after publication changes."""

    flags = (
        os.O_RDONLY
        | getattr(
            os,
            "O_DIRECTORY",
            0,
        )
    )

    descriptor = os.open(
        path,
        flags,
    )

    try:
        os.fsync(
            descriptor
        )
    finally:
        os.close(
            descriptor
        )


def _link_no_overwrite(
    temporary: Path,
    destination: Path,
) -> None:
    """Atomically publish one file without overwriting an existing path."""

    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite authoritative checkpoint file: "
            f"{destination}"
        )

    try:
        os.link(
            temporary,
            destination,
        )
    except FileExistsError:
        raise FileExistsError(
            f"refusing to overwrite authoritative checkpoint file: "
            f"{destination}"
        ) from None

    _fsync_directory(
        destination.parent
    )


def _write_state_temp(
    directory: Path,
    state: Mapping[str, np.ndarray],
) -> Path:
    """Write and fsync one uncompressed NPZ temporary state."""

    descriptor, name = tempfile.mkstemp(
        prefix=".model_state.",
        suffix=".npz.tmp",
        dir=directory,
    )

    os.close(
        descriptor
    )

    path = Path(
        name
    )

    try:
        # Passing an open binary handle prevents NumPy from appending ".npz".
        with path.open(
            "wb"
        ) as handle:
            np.savez(
                handle,
                logits=state[
                    "logits"
                ],
                means=state[
                    "means"
                ],
                raw_scales=state[
                    "raw_scales"
                ],
            )

            handle.flush()
            os.fsync(
                handle.fileno()
            )
    except Exception:
        path.unlink(
            missing_ok=True
        )
        raise

    return path


def _write_metadata_temp(
    directory: Path,
    metadata: Mapping[str, Any],
) -> Path:
    """Write canonical compact JSON with exactly one trailing newline."""

    descriptor, name = tempfile.mkstemp(
        prefix=".model_metadata.",
        suffix=".json.tmp",
        dir=directory,
    )

    os.close(
        descriptor
    )

    path = Path(
        name
    )

    payload = (
        canonical_json_bytes(
            dict(
                metadata
            )
        )
        + b"\n"
    )

    try:
        with path.open(
            "wb"
        ) as handle:
            handle.write(
                payload
            )

            handle.flush()
            os.fsync(
                handle.fileno()
            )
    except Exception:
        path.unlink(
            missing_ok=True
        )
        raise

    return path


def _read_metadata(
    path: Path,
) -> dict[str, Any]:
    """Load metadata and require canonical byte representation."""

    raw = path.read_bytes()

    if not raw.endswith(
        b"\n"
    ):
        raise ValueError(
            "GMM metadata must contain exactly one trailing newline"
        )

    if raw.endswith(
        b"\n\n"
    ):
        raise ValueError(
            "GMM metadata contains more than one trailing newline"
        )

    try:
        value = json.loads(
            raw.decode(
                "utf-8"
            )
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise ValueError(
            "GMM metadata is not valid UTF-8 JSON"
        ) from exc

    if not isinstance(
        value,
        dict,
    ):
        raise ValueError(
            "GMM metadata root must be an object"
        )

    canonical = (
        canonical_json_bytes(
            value
        )
        + b"\n"
    )

    if raw != canonical:
        raise ValueError(
            "GMM metadata is not in governed canonical JSON representation"
        )

    return value


def _validate_loaded_metadata(
    metadata: Mapping[str, Any],
    *,
    requested_dataset_id: str | None = None,
    requested_class_id: int | None = None,
    requested_experimental_seed: int | None = None,
) -> None:
    """Validate schema, structure, identity, and checkpoint selection."""

    required_keys = {
        "schema_version",
        "schema_type",
        "dataset_id",
        "class_id",
        "experimental_seed",
        "state_file",
        "state_sha256",
        "components",
        "dimension",
        "parameter_dtype",
        "observation_model",
        "selected_step",
        "selected_validation_nll",
        "train_class_count",
        "validation_class_count",
        "maximum_updates",
        "validation_interval_updates",
        "validation_history",
    }

    if set(metadata) not in (
        required_keys,
        required_keys | {"execution_evidence"},
    ):
        raise ValueError(
            "GMM metadata keys do not exactly match governed schema"
        )
    if "execution_evidence" in metadata:
        _validate_execution_evidence(metadata["execution_evidence"])

    if metadata[
        "schema_version"
    ] != SCHEMA_VERSION:
        raise ValueError(
            "unsupported GMM metadata schema version"
        )

    if metadata[
        "schema_type"
    ] != SCHEMA_TYPE:
        raise ValueError(
            "unsupported GMM metadata schema type"
        )

    _validate_dataset_id(
        metadata[
            "dataset_id"
        ]
    )

    class_id = _exact_integer(
        metadata[
            "class_id"
        ],
        name="metadata class_id",
        minimum=0,
    )

    seed = _exact_integer(
        metadata[
            "experimental_seed"
        ],
        name="metadata experimental_seed",
        minimum=0,
    )

    if seed > np.iinfo(
        np.uint32
    ).max:
        raise ValueError(
            "metadata experimental_seed is outside governed range"
        )

    if metadata[
        "state_file"
    ] != STATE_FILENAME:
        raise ValueError(
            "GMM metadata state filename mismatch"
        )

    if (
        not isinstance(
            metadata[
                "state_sha256"
            ],
            str,
        )
        or SHA256_RE.fullmatch(
            metadata[
                "state_sha256"
            ]
        )
        is None
    ):
        raise ValueError(
            "GMM metadata state_sha256 is invalid"
        )

    if metadata[
        "components"
    ] != NUM_COMPONENTS:
        raise ValueError(
            "GMM metadata component count mismatch"
        )

    if metadata[
        "dimension"
    ] != CANONICAL_DIM:
        raise ValueError(
            "GMM metadata dimension mismatch"
        )

    if metadata[
        "parameter_dtype"
    ] != PARAMETER_DTYPE:
        raise ValueError(
            "GMM metadata parameter dtype mismatch"
        )

    if metadata[
        "observation_model"
    ] != OBSERVATION_MODEL:
        raise ValueError(
            "GMM metadata observation model mismatch"
        )

    for key in (
        "train_class_count",
        "validation_class_count",
        "maximum_updates",
        "validation_interval_updates",
    ):
        _exact_integer(
            metadata[
                key
            ],
            name=f"metadata {key}",
            minimum=1,
        )

    history = metadata[
        "validation_history"
    ]

    if not isinstance(
        history,
        list,
    ) or not history:
        raise ValueError(
            "GMM validation_history must be a non-empty list"
        )

    previous_step = -1

    parsed_history: list[
        tuple[
            int,
            float,
        ]
    ] = []

    for entry in history:
        if not isinstance(
            entry,
            dict,
        ) or set(
            entry
        ) != {
            "step",
            "validation_nll",
        }:
            raise ValueError(
                "GMM validation history entry has invalid schema"
            )

        step = _exact_integer(
            entry[
                "step"
            ],
            name="metadata validation step",
            minimum=0,
        )

        nll = float(
            entry[
                "validation_nll"
            ]
        )

        if not np.isfinite(
            nll
        ):
            raise ValueError(
                "GMM metadata contains non-finite validation NLL"
            )

        if step <= previous_step:
            raise ValueError(
                "GMM metadata validation steps are not strictly increasing"
            )

        parsed_history.append(
            (
                step,
                nll,
            )
        )

        previous_step = step

    if parsed_history[
        0
    ][
        0
    ] != 0:
        raise ValueError(
            "GMM metadata validation history must begin at step 0"
        )

    selected_step = _exact_integer(
        metadata[
            "selected_step"
        ],
        name="metadata selected_step",
        minimum=0,
    )

    selected_nll = float(
        metadata[
            "selected_validation_nll"
        ]
    )

    if not np.isfinite(
        selected_nll
    ):
        raise ValueError(
            "GMM metadata selected validation NLL is non-finite"
        )

    minimum_nll = min(
        nll
        for _, nll in parsed_history
    )

    earliest_step = min(
        step
        for step, nll in parsed_history
        if nll == minimum_nll
    )

    if (
        selected_nll
        != minimum_nll
        or selected_step
        != earliest_step
    ):
        raise ValueError(
            "GMM metadata selected checkpoint violates governed policy"
        )

    if (
        selected_step,
        selected_nll,
    ) not in parsed_history:
        raise ValueError(
            "GMM metadata selected checkpoint is absent from history"
        )

    maximum_updates = int(
        metadata[
            "maximum_updates"
        ]
    )

    interval = int(
        metadata[
            "validation_interval_updates"
        ]
    )

    expected_steps = (
        0,
        *range(
            interval,
            maximum_updates
            + 1,
            interval,
        ),
    )

    observed_steps = tuple(
        step
        for step, _ in parsed_history
    )

    if observed_steps != expected_steps:
        raise ValueError(
            "GMM metadata validation history does not match budget schedule"
        )

    if requested_dataset_id is not None:
        if metadata[
            "dataset_id"
        ] != _validate_dataset_id(
            requested_dataset_id
        ):
            raise ValueError(
                "requested dataset identity does not match GMM checkpoint"
            )

    if requested_class_id is not None:
        requested = _exact_integer(
            requested_class_id,
            name="requested_class_id",
            minimum=0,
        )

        if class_id != requested:
            raise ValueError(
                "requested class identity does not match GMM checkpoint"
            )

    if requested_experimental_seed is not None:
        requested = _exact_integer(
            requested_experimental_seed,
            name="requested_experimental_seed",
            minimum=0,
        )

        if seed != requested:
            raise ValueError(
                "requested experimental seed does not match GMM checkpoint"
            )


def load_gmm_checkpoint(
    directory: Path | str,
    *,
    requested_dataset_id: str | None = None,
    requested_class_id: int | None = None,
    requested_experimental_seed: int | None = None,
) -> GMMPersistedCheckpoint:
    """Load and fully validate one authoritative checkpoint pair."""

    root = Path(
        directory
    ).expanduser()

    metadata_path = root / METADATA_FILENAME
    state_path = root / STATE_FILENAME

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"missing GMM metadata file: {metadata_path}"
        )

    if not state_path.is_file():
        raise FileNotFoundError(
            f"missing GMM state file: {state_path}"
        )

    metadata = _read_metadata(
        metadata_path
    )

    _validate_loaded_metadata(
        metadata,
        requested_dataset_id=requested_dataset_id,
        requested_class_id=requested_class_id,
        requested_experimental_seed=requested_experimental_seed,
    )

    observed_hash = sha256_file(
        state_path
    )

    if observed_hash != metadata[
        "state_sha256"
    ]:
        raise ValueError(
            "GMM checkpoint state SHA-256 mismatch"
        )

    try:
        with np.load(
            state_path,
            allow_pickle=False,
        ) as archive:
            if set(
                archive.files
            ) != set(
                STATE_KEYS
            ):
                raise ValueError(
                    "GMM state NPZ contains unexpected or missing keys"
                )

            state = {
                key: archive[
                    key
                ].copy()
                for key in STATE_KEYS
            }
    except (
        OSError,
        ValueError,
    ) as exc:
        if isinstance(
            exc,
            ValueError,
        ):
            raise

        raise ValueError(
            "unable to load governed GMM state NPZ"
        ) from exc

    validated = _validate_state_arrays(
        state
    )

    model = CensoredDiagonalGaussianMixture(
        logits=validated[
            "logits"
        ],
        means=validated[
            "means"
        ],
        raw_scales=validated[
            "raw_scales"
        ],
    )

    return GMMPersistedCheckpoint(
        directory=root,
        state_path=state_path,
        metadata_path=metadata_path,
        metadata=dict(
            metadata
        ),
        model=model,
    )


def persist_gmm_checkpoint(
    directory: Path | str,
    *,
    dataset_id: str,
    class_id: int,
    experimental_seed: int,
    training_result: GMMClassTrainingResult,
    validation_class_count: int,
    execution_evidence: Mapping[str, Any] | None = None,
) -> GMMPersistedCheckpoint:
    """Persist one selected authoritative class-model checkpoint pair.

    The destination directory must already exist.

    Existing authoritative state or metadata files are never overwritten.
    """

    if not isinstance(
        training_result,
        GMMClassTrainingResult,
    ):
        raise TypeError(
            "training_result must be GMMClassTrainingResult"
        )

    root = Path(
        directory
    ).expanduser()

    if not root.is_dir():
        raise ValueError(
            f"GMM checkpoint directory does not exist: {root}"
        )

    state_path = root / STATE_FILENAME
    metadata_path = root / METADATA_FILENAME

    if state_path.exists():
        raise FileExistsError(
            f"refusing to overwrite authoritative GMM state: {state_path}"
        )

    if metadata_path.exists():
        raise FileExistsError(
            f"refusing to overwrite authoritative GMM metadata: "
            f"{metadata_path}"
        )

    state = _validate_state_arrays(
        training_result.model.state_arrays()
    )

    state_temp: Path | None = None
    metadata_temp: Path | None = None

    state_published = False

    try:
        state_temp = _write_state_temp(
            root,
            state,
        )

        state_hash = sha256_file(
            state_temp
        )

        metadata = _build_metadata(
            dataset_id=dataset_id,
            class_id=class_id,
            experimental_seed=experimental_seed,
            state_sha256=state_hash,
            training_result=training_result,
            validation_class_count=validation_class_count,
            execution_evidence=execution_evidence,
        )

        metadata_temp = _write_metadata_temp(
            root,
            metadata,
        )

        # Validate the temporary state artifact exactly before publication.
        with np.load(
            state_temp,
            allow_pickle=False,
        ) as archive:
            if set(
                archive.files
            ) != set(
                STATE_KEYS
            ):
                raise ValueError(
                    "temporary GMM state has invalid keys"
                )

            roundtrip = {
                key: archive[
                    key
                ].copy()
                for key in STATE_KEYS
            }

        roundtrip = _validate_state_arrays(
            roundtrip
        )

        for key in STATE_KEYS:
            np.testing.assert_array_equal(
                roundtrip[
                    key
                ],
                state[
                    key
                ],
            )

        # Validate temporary metadata bytes before publication.
        temporary_metadata = _read_metadata(
            metadata_temp
        )

        _validate_loaded_metadata(
            temporary_metadata,
            requested_dataset_id=dataset_id,
            requested_class_id=class_id,
            requested_experimental_seed=experimental_seed,
        )

        if temporary_metadata[
            "state_sha256"
        ] != state_hash:
            raise ValueError(
                "temporary GMM metadata/state digest binding failed"
            )

        # Publish state first.
        _link_no_overwrite(
            state_temp,
            state_path,
        )

        state_published = True

        if sha256_file(
            state_path
        ) != state_hash:
            raise ValueError(
                "published GMM state digest changed"
            )

        # Metadata is the final pair-completion marker.
        _link_no_overwrite(
            metadata_temp,
            metadata_path,
        )

        _fsync_directory(
            root
        )

    except Exception:
        # If metadata publication fails, remove the newly published state so
        # this call does not knowingly leave an incomplete authoritative pair.
        if state_published and not metadata_path.exists():
            try:
                state_path.unlink(
                    missing_ok=True
                )
                _fsync_directory(
                    root
                )
            except OSError:
                pass

        raise

    finally:
        if state_temp is not None:
            state_temp.unlink(
                missing_ok=True
            )

        if metadata_temp is not None:
            metadata_temp.unlink(
                missing_ok=True
            )

    return load_gmm_checkpoint(
        root,
        requested_dataset_id=dataset_id,
        requested_class_id=class_id,
        requested_experimental_seed=experimental_seed,
    )


__all__ = [
    "GMMPersistedCheckpoint",
    "METADATA_FILENAME",
    "OBSERVATION_MODEL",
    "PARAMETER_DTYPE",
    "SCHEMA_TYPE",
    "SCHEMA_VERSION",
    "STATE_FILENAME",
    "STATE_KEYS",
    "load_gmm_checkpoint",
    "persist_gmm_checkpoint",
]
