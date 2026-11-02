"""Governed checkpoint persistence for the Paper 1 revision-2026 RBM.

One authoritative checkpoint pair represents one shared class-conditional
model identified scientifically by:

    (dataset_id, experimental_seed)

The pair is:

    model_state.npz
    model_metadata.json

Publication is state-first and metadata-last. Existing authoritative files are
never overwritten.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import operator
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from trustforge.models.paper01_revision_2026.restricted_boltzmann_cd import (
    CD_K,
)
from trustforge.models.paper01_revision_2026.restricted_boltzmann_model import (
    HIDDEN_UNITS,
    VISIBLE_DIM,
    SharedConditionalRBM,
)
from trustforge.models.paper01_revision_2026.restricted_boltzmann_optimizer import (
    LEARNING_RATE,
)
from trustforge.models.paper01_revision_2026.restricted_boltzmann_training import (
    RBMTrainingResult,
)
from trustforge.models.paper01_revision_2026.restricted_boltzmann_training_budget import (
    BATCH_SIZE,
)
from trustforge.models.paper01_revision_2026.restricted_boltzmann_traversal import (
    CLASS_BALANCE_EXPONENT,
)
from trustforge.provenance import (
    canonical_json_bytes,
    sha256_file,
)


STATE_FILENAME = "model_state.npz"
METADATA_FILENAME = "model_metadata.json"

SCHEMA_VERSION = "1.0"
SCHEMA_TYPE = "paper01_revision2026_restricted_boltzmann_shared_model"

MODEL_FAMILY = "restricted_boltzmann"
PAPER_REVISION = "paper01_revision_2026"
PARAMETER_DTYPE = "float64"
OPTIMIZER_IDENTITY = "manual_sgd"
CHECKPOINT_SELECTION_RULE = "final_governed_update"

STATE_KEYS = (
    "W",
    "a",
    "b",
    "U",
)

SHA256_RE = re.compile(
    r"^[0-9a-f]{64}$"
)


@dataclass(frozen=True)
class RBMPersistedCheckpoint:
    """Validated authoritative revision-2026 RBM checkpoint pair."""

    directory: Path
    state_path: Path
    metadata_path: Path
    metadata: dict[str, Any]
    model: SharedConditionalRBM


def _exact_integer(
    value: int,
    *,
    name: str,
    minimum: int = 0,
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

    if result < minimum:
        raise ValueError(
            f"{name} must be >= {minimum}"
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

    value = dataset_id.strip()

    if not value:
        raise ValueError(
            "dataset_id must not be empty"
        )

    return value


def _validate_seed(
    experimental_seed: int,
) -> int:
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

    return seed


def _validate_class_mapping(
    class_mapping: Sequence[
        str
    ],
    *,
    expected_class_count: int | None = None,
) -> tuple[str, ...]:
    if isinstance(
        class_mapping,
        (
            str,
            bytes,
        ),
    ):
        raise TypeError(
            "class_mapping must be an ordered sequence of labels"
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

    if (
        expected_class_count is not None
        and len(
            normalized
        )
        != expected_class_count
    ):
        raise ValueError(
            "class_mapping length does not match governed class count"
        )

    return tuple(
        normalized
    )


def _state_arrays_from_model(
    model: SharedConditionalRBM,
) -> dict[str, np.ndarray]:
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
    ) != set(
        STATE_KEYS
    ):
        raise ValueError(
            "RBM model state keys do not exactly match W,a,b,U"
        )

    return {
        key: np.asarray(
            state[
                key
            ]
        ).copy()
        for key in STATE_KEYS
    }


def _validate_state_arrays(
    state: Mapping[
        str,
        np.ndarray,
    ],
    *,
    expected_class_count: int | None = None,
) -> dict[str, np.ndarray]:
    if set(
        state
    ) != set(
        STATE_KEYS
    ):
        raise ValueError(
            "RBM state keys do not exactly match W,a,b,U"
        )

    arrays = {
        key: np.asarray(
            state[
                key
            ]
        )
        for key in STATE_KEYS
    }

    class_count = int(
        arrays[
            "U"
        ].shape[
            0
        ]
    ) if arrays[
        "U"
    ].ndim == 2 else -1

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
            class_count,
            HIDDEN_UNITS,
        ),
    }

    if class_count < 1:
        raise ValueError(
            "RBM U must contain at least one class row"
        )

    if (
        expected_class_count is not None
        and class_count
        != expected_class_count
    ):
        raise ValueError(
            "RBM U class count does not match governed class mapping"
        )

    validated: dict[
        str,
        np.ndarray,
    ] = {}

    for key in STATE_KEYS:
        array = arrays[
            key
        ]

        if array.shape != expected_shapes[
            key
        ]:
            raise ValueError(
                f"RBM state {key} has invalid shape {array.shape}"
            )

        if array.dtype != np.float64:
            raise TypeError(
                f"RBM state {key} must have exact dtype float64"
            )

        if not np.isfinite(
            array
        ).all():
            raise ValueError(
                f"RBM state {key} contains non-finite values"
            )

        validated[
            key
        ] = array.copy()

    return validated


def _validation_history_json(
    training_result: RBMTrainingResult,
) -> list[dict[str, Any]]:
    history = []

    for observation in training_result.validation_history:
        diagnostics = observation.diagnostics

        mse = float(
            diagnostics.reconstruction_mse
        )

        if not np.isfinite(
            mse
        ):
            raise ValueError(
                "RBM validation history contains non-finite MSE"
            )

        history.append(
            {
                "step": int(
                    observation.step
                ),
                "reconstruction_mse": mse,
                "finite_state": True,
            }
        )

    expected_steps = list(
        training_result.budget.validation_steps
    )

    if [
        item[
            "step"
        ]
        for item in history
    ] != expected_steps:
        raise ValueError(
            "RBM validation history does not match governed schedule"
        )

    if len(
        history
    ) != 3:
        raise ValueError(
            "RBM validation history must contain exactly three observations"
        )

    return history


def _class_mapping_json(
    labels: Sequence[
        str
    ],
) -> list[dict[str, Any]]:
    return [
        {
            "class_id": int(
                class_id
            ),
            "class_label": label,
        }
        for class_id, label in enumerate(
            labels
        )
    ]


def _build_metadata(
    *,
    dataset_id: str,
    experimental_seed: int,
    class_mapping: Sequence[
        str
    ],
    state_sha256: str,
    training_result: RBMTrainingResult,
) -> dict[str, Any]:
    dataset = _validate_dataset_id(
        dataset_id
    )

    seed = _validate_seed(
        experimental_seed
    )

    if (
        not isinstance(
            state_sha256,
            str,
        )
        or SHA256_RE.fullmatch(
            state_sha256
        )
        is None
    ):
        raise ValueError(
            "state_sha256 must be 64 lowercase hexadecimal characters"
        )

    class_count = int(
        training_result.model.class_count
    )

    labels = _validate_class_mapping(
        class_mapping,
        expected_class_count=class_count,
    )

    if (
        training_result.optimizer_steps_completed
        != training_result.budget.max_updates
    ):
        raise ValueError(
            "RBM training result did not complete governed update budget"
        )

    if (
        training_result.budget.authoritative_step
        != training_result.budget.max_updates
    ):
        raise ValueError(
            "RBM authoritative step must equal max_updates"
        )

    validation_history = _validation_history_json(
        training_result
    )

    metadata: dict[
        str,
        Any,
    ] = {
        "schema_version": SCHEMA_VERSION,
        "schema_type": SCHEMA_TYPE,
        "model_family": MODEL_FAMILY,
        "paper_revision": PAPER_REVISION,
        "dataset_id": dataset,
        "experimental_seed": seed,
        "class_count": class_count,
        "class_mapping": _class_mapping_json(
            labels
        ),
        "visible_dimension": VISIBLE_DIM,
        "hidden_units": HIDDEN_UNITS,
        "parameter_dtype": PARAMETER_DTYPE,
        "state_file": STATE_FILENAME,
        "state_sha256": state_sha256,
        "batch_size": BATCH_SIZE,
        "class_balance_exponent": CLASS_BALANCE_EXPONENT,
        "cd_depth": CD_K,
        "optimizer": OPTIMIZER_IDENTITY,
        "learning_rate": LEARNING_RATE,
        "maximum_updates": int(
            training_result.budget.max_updates
        ),
        "selected_step": int(
            training_result.budget.authoritative_step
        ),
        "validation_schedule": [
            int(
                step
            )
            for step in training_result.budget.validation_steps
        ],
        "validation_history": validation_history,
        "checkpoint_selection_rule": CHECKPOINT_SELECTION_RULE,
    }

    canonical_json_bytes(
        metadata
    )

    return metadata


def _fsync_directory(
    path: Path,
) -> None:
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
    if destination.exists():
        raise FileExistsError(
            "refusing to overwrite authoritative RBM checkpoint file: "
            f"{destination}"
        )

    try:
        os.link(
            temporary,
            destination,
        )
    except FileExistsError:
        raise FileExistsError(
            "refusing to overwrite authoritative RBM checkpoint file: "
            f"{destination}"
        ) from None

    _fsync_directory(
        destination.parent
    )


def _write_state_temp(
    directory: Path,
    state: Mapping[
        str,
        np.ndarray,
    ],
) -> Path:
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
        with path.open(
            "wb"
        ) as handle:
            np.savez(
                handle,
                W=state[
                    "W"
                ],
                a=state[
                    "a"
                ],
                b=state[
                    "b"
                ],
                U=state[
                    "U"
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
    metadata: Mapping[
        str,
        Any,
    ],
) -> Path:
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
    raw = path.read_bytes()

    if not raw.endswith(
        b"\n"
    ):
        raise ValueError(
            "RBM metadata must contain exactly one trailing newline"
        )

    if raw.endswith(
        b"\n\n"
    ):
        raise ValueError(
            "RBM metadata contains more than one trailing newline"
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
            "unable to parse governed RBM metadata JSON"
        ) from exc

    if not isinstance(
        value,
        dict,
    ):
        raise ValueError(
            "RBM metadata root must be a JSON object"
        )

    expected = (
        canonical_json_bytes(
            value
        )
        + b"\n"
    )

    if raw != expected:
        raise ValueError(
            "RBM metadata is not canonical deterministic JSON"
        )

    return value


def _mapping_from_metadata(
    metadata_mapping,
) -> tuple[str, ...]:
    if not isinstance(
        metadata_mapping,
        list,
    ):
        raise ValueError(
            "RBM metadata class_mapping must be a list"
        )

    labels = []

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
                "RBM metadata class_mapping entry has invalid schema"
            )

        if entry[
            "class_id"
        ] != expected_id:
            raise ValueError(
                "RBM metadata class IDs must be contiguous and ordered"
            )

        labels.append(
            entry[
                "class_label"
            ]
        )

    return _validate_class_mapping(
        labels
    )


def _validate_loaded_metadata(
    metadata: Mapping[
        str,
        Any,
    ],
    *,
    requested_dataset_id: str | None = None,
    requested_experimental_seed: int | None = None,
    requested_class_mapping: Sequence[
        str
    ] | None = None,
) -> tuple[str, ...]:
    required_keys = {
        "schema_version",
        "schema_type",
        "model_family",
        "paper_revision",
        "dataset_id",
        "experimental_seed",
        "class_count",
        "class_mapping",
        "visible_dimension",
        "hidden_units",
        "parameter_dtype",
        "state_file",
        "state_sha256",
        "batch_size",
        "class_balance_exponent",
        "cd_depth",
        "optimizer",
        "learning_rate",
        "maximum_updates",
        "selected_step",
        "validation_schedule",
        "validation_history",
        "checkpoint_selection_rule",
    }

    if set(
        metadata
    ) != required_keys:
        raise ValueError(
            "RBM metadata keys do not exactly match governed schema"
        )

    if metadata[
        "schema_version"
    ] != SCHEMA_VERSION:
        raise ValueError(
            "unsupported RBM metadata schema version"
        )

    if metadata[
        "schema_type"
    ] != SCHEMA_TYPE:
        raise ValueError(
            "unsupported RBM metadata schema type"
        )

    if metadata[
        "model_family"
    ] != MODEL_FAMILY:
        raise ValueError(
            "RBM metadata model family mismatch"
        )

    if metadata[
        "paper_revision"
    ] != PAPER_REVISION:
        raise ValueError(
            "RBM metadata Paper 1 revision mismatch"
        )

    dataset = _validate_dataset_id(
        metadata[
            "dataset_id"
        ]
    )

    seed = _validate_seed(
        metadata[
            "experimental_seed"
        ]
    )

    class_count = _exact_integer(
        metadata[
            "class_count"
        ],
        name="metadata class_count",
        minimum=1,
    )

    labels = _mapping_from_metadata(
        metadata[
            "class_mapping"
        ]
    )

    if len(
        labels
    ) != class_count:
        raise ValueError(
            "RBM metadata class mapping length mismatch"
        )

    if metadata[
        "visible_dimension"
    ] != VISIBLE_DIM:
        raise ValueError(
            "RBM metadata visible dimension mismatch"
        )

    if metadata[
        "hidden_units"
    ] != HIDDEN_UNITS:
        raise ValueError(
            "RBM metadata hidden width mismatch"
        )

    if metadata[
        "parameter_dtype"
    ] != PARAMETER_DTYPE:
        raise ValueError(
            "RBM metadata parameter dtype mismatch"
        )

    if metadata[
        "state_file"
    ] != STATE_FILENAME:
        raise ValueError(
            "RBM metadata state filename mismatch"
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
            "RBM metadata state_sha256 is invalid"
        )

    if metadata[
        "batch_size"
    ] != BATCH_SIZE:
        raise ValueError(
            "RBM metadata batch size mismatch"
        )

    if metadata[
        "class_balance_exponent"
    ] != CLASS_BALANCE_EXPONENT:
        raise ValueError(
            "RBM metadata class-balance exponent mismatch"
        )

    if metadata[
        "cd_depth"
    ] != CD_K:
        raise ValueError(
            "RBM metadata CD depth mismatch"
        )

    if metadata[
        "optimizer"
    ] != OPTIMIZER_IDENTITY:
        raise ValueError(
            "RBM metadata optimizer identity mismatch"
        )

    if metadata[
        "learning_rate"
    ] != LEARNING_RATE:
        raise ValueError(
            "RBM metadata learning rate mismatch"
        )

    maximum_updates = _exact_integer(
        metadata[
            "maximum_updates"
        ],
        name="metadata maximum_updates",
        minimum=1,
    )

    selected_step = _exact_integer(
        metadata[
            "selected_step"
        ],
        name="metadata selected_step",
        minimum=0,
    )

    if selected_step != maximum_updates:
        raise ValueError(
            "RBM selected step must equal final governed update"
        )

    if metadata[
        "checkpoint_selection_rule"
    ] != CHECKPOINT_SELECTION_RULE:
        raise ValueError(
            "RBM checkpoint-selection rule mismatch"
        )

    schedule = metadata[
        "validation_schedule"
    ]

    if (
        not isinstance(
            schedule,
            list,
        )
        or len(
            schedule
        )
        != 3
    ):
        raise ValueError(
            "RBM validation schedule must contain exactly three steps"
        )

    schedule = [
        _exact_integer(
            value,
            name="metadata validation step",
            minimum=0,
        )
        for value in schedule
    ]

    if (
        schedule[
            0
        ]
        != 0
        or schedule[
            -1
        ]
        != maximum_updates
        or schedule[
            1
        ]
        * 2
        != maximum_updates
    ):
        raise ValueError(
            "RBM validation schedule is not governed 0,B,2B"
        )

    history = metadata[
        "validation_history"
    ]

    if (
        not isinstance(
            history,
            list,
        )
        or len(
            history
        )
        != 3
    ):
        raise ValueError(
            "RBM validation history must contain exactly three records"
        )

    observed_steps = []

    for record in history:
        if (
            not isinstance(
                record,
                dict,
            )
            or set(
                record
            )
            != {
                "step",
                "reconstruction_mse",
                "finite_state",
            }
        ):
            raise ValueError(
                "RBM validation-history record has invalid schema"
            )

        step = _exact_integer(
            record[
                "step"
            ],
            name="validation-history step",
            minimum=0,
        )

        mse = record[
            "reconstruction_mse"
        ]

        if (
            isinstance(
                mse,
                bool,
            )
            or not isinstance(
                mse,
                (
                    int,
                    float,
                ),
            )
            or not np.isfinite(
                float(
                    mse
                )
            )
        ):
            raise ValueError(
                "RBM validation-history MSE is invalid"
            )

        if record[
            "finite_state"
        ] is not True:
            raise ValueError(
                "RBM validation-history finite-state status is not true"
            )

        observed_steps.append(
            step
        )

    if observed_steps != schedule:
        raise ValueError(
            "RBM validation history does not match governed schedule"
        )

    if requested_dataset_id is not None:
        if dataset != _validate_dataset_id(
            requested_dataset_id
        ):
            raise ValueError(
                "requested dataset does not match RBM checkpoint"
            )

    if requested_experimental_seed is not None:
        if seed != _validate_seed(
            requested_experimental_seed
        ):
            raise ValueError(
                "requested experimental seed does not match RBM checkpoint"
            )

    if requested_class_mapping is not None:
        requested_labels = _validate_class_mapping(
            requested_class_mapping,
            expected_class_count=class_count,
        )

        if tuple(
            labels
        ) != tuple(
            requested_labels
        ):
            raise ValueError(
                "requested class mapping does not match RBM checkpoint"
            )

    return labels


def load_rbm_checkpoint(
    directory: Path | str,
    *,
    requested_dataset_id: str | None = None,
    requested_experimental_seed: int | None = None,
    requested_class_mapping: Sequence[
        str
    ] | None = None,
) -> RBMPersistedCheckpoint:
    """Load and fully validate one authoritative RBM checkpoint pair."""

    root = Path(
        directory
    ).expanduser()

    metadata_path = root / METADATA_FILENAME
    state_path = root / STATE_FILENAME

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"missing RBM metadata file: {metadata_path}"
        )

    if not state_path.is_file():
        raise FileNotFoundError(
            f"missing RBM state file: {state_path}"
        )

    metadata = _read_metadata(
        metadata_path
    )

    labels = _validate_loaded_metadata(
        metadata,
        requested_dataset_id=requested_dataset_id,
        requested_experimental_seed=requested_experimental_seed,
        requested_class_mapping=requested_class_mapping,
    )

    observed_hash = sha256_file(
        state_path
    )

    if observed_hash != metadata[
        "state_sha256"
    ]:
        raise ValueError(
            "RBM checkpoint state SHA-256 mismatch"
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
                    "RBM state NPZ contains unexpected or missing keys"
                )

            state = {
                key: archive[
                    key
                ].copy()
                for key in STATE_KEYS
            }
    except OSError as exc:
        raise ValueError(
            "unable to load governed RBM state NPZ"
        ) from exc

    validated = _validate_state_arrays(
        state,
        expected_class_count=len(
            labels
        ),
    )

    model = SharedConditionalRBM(
        W=validated[
            "W"
        ],
        a=validated[
            "a"
        ],
        b=validated[
            "b"
        ],
        U=validated[
            "U"
        ],
    )

    return RBMPersistedCheckpoint(
        directory=root,
        state_path=state_path,
        metadata_path=metadata_path,
        metadata=dict(
            metadata
        ),
        model=model,
    )


def persist_rbm_checkpoint(
    directory: Path | str,
    *,
    dataset_id: str,
    experimental_seed: int,
    class_mapping: Sequence[
        str
    ],
    training_result: RBMTrainingResult,
) -> RBMPersistedCheckpoint:
    """Persist one final-update authoritative RBM checkpoint pair."""

    if not isinstance(
        training_result,
        RBMTrainingResult,
    ):
        raise TypeError(
            "training_result must be RBMTrainingResult"
        )

    root = Path(
        directory
    ).expanduser()

    if not root.is_dir():
        raise ValueError(
            f"RBM checkpoint directory does not exist: {root}"
        )

    state_path = root / STATE_FILENAME
    metadata_path = root / METADATA_FILENAME

    if state_path.exists():
        raise FileExistsError(
            f"refusing to overwrite authoritative RBM state: {state_path}"
        )

    if metadata_path.exists():
        raise FileExistsError(
            "refusing to overwrite authoritative RBM metadata: "
            f"{metadata_path}"
        )

    labels = _validate_class_mapping(
        class_mapping,
        expected_class_count=training_result.model.class_count,
    )

    state = _validate_state_arrays(
        _state_arrays_from_model(
            training_result.model
        ),
        expected_class_count=len(
            labels
        ),
    )

    state_temp: Path | None = None
    metadata_temp: Path | None = None
    state_published = False

    try:
        state_temp = _write_state_temp(
            root,
            state,
        )

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
                    "temporary RBM state has invalid keys"
                )

            roundtrip = {
                key: archive[
                    key
                ].copy()
                for key in STATE_KEYS
            }

        roundtrip = _validate_state_arrays(
            roundtrip,
            expected_class_count=len(
                labels
            ),
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

        state_hash = sha256_file(
            state_temp
        )

        metadata = _build_metadata(
            dataset_id=dataset_id,
            experimental_seed=experimental_seed,
            class_mapping=labels,
            state_sha256=state_hash,
            training_result=training_result,
        )

        metadata_temp = _write_metadata_temp(
            root,
            metadata,
        )

        temporary_metadata = _read_metadata(
            metadata_temp
        )

        _validate_loaded_metadata(
            temporary_metadata,
            requested_dataset_id=dataset_id,
            requested_experimental_seed=experimental_seed,
            requested_class_mapping=labels,
        )

        if temporary_metadata[
            "state_sha256"
        ] != state_hash:
            raise ValueError(
                "temporary RBM metadata/state digest binding failed"
            )

        _link_no_overwrite(
            state_temp,
            state_path,
        )

        state_published = True

        if sha256_file(
            state_path
        ) != state_hash:
            raise ValueError(
                "published RBM state digest changed"
            )

        _link_no_overwrite(
            metadata_temp,
            metadata_path,
        )

        _fsync_directory(
            root
        )

    except Exception:
        if (
            state_published
            and not metadata_path.exists()
        ):
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

    return load_rbm_checkpoint(
        root,
        requested_dataset_id=dataset_id,
        requested_experimental_seed=experimental_seed,
        requested_class_mapping=labels,
    )


__all__ = [
    "CHECKPOINT_SELECTION_RULE",
    "METADATA_FILENAME",
    "MODEL_FAMILY",
    "PARAMETER_DTYPE",
    "PAPER_REVISION",
    "RBMPersistedCheckpoint",
    "SCHEMA_TYPE",
    "SCHEMA_VERSION",
    "STATE_FILENAME",
    "STATE_KEYS",
    "load_rbm_checkpoint",
    "persist_rbm_checkpoint",
]
