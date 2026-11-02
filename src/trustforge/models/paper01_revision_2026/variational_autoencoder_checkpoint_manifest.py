"""Strict VAE checkpoint-v1 manifest validation.

Paper 1 revision 2026.

This module validates checkpoint metadata only. It does not
inspect files, validate actual execution provenance, select
checkpoints, restore weights, or authorize training.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
import re

from trustforge.provenance import canonical_json_bytes

from trustforge.models.paper01_revision_2026.gan_checkpoint_manifest import (
    DATASET_CLASS_COUNTS,
)
from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_provenance import (
    VAE_PROVENANCE_FILENAME,
)


SCHEMA_VERSION = "1.0"
SCHEMA_TYPE = "paper01_revision2026_conditional_vae_checkpoint"
PAPER_REVISION = "paper01_revision_2026"
MODEL_FAMILY = "conditional_vae"

MANIFEST_FILENAME = "vae_checkpoint_manifest.json"

STATE_FILENAMES = frozenset({
    "vae_encoder_state.npz",
    "vae_decoder_state.npz",
})

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

REQUIRED_FIELDS = frozenset({
    "schema_version",
    "schema_type",
    "paper_revision",
    "model_family",
    "dataset_id",
    "experimental_seed",
    "class_mapping",
    "class_count",
    "image_shape",
    "latent_dim",
    "encoder_dense_units",
    "decoder_feature_channels",
    "architecture_version",
    "observation_version",
    "observation_variance",
    "kl_weight",
    "kl_annealing",
    "completed_epochs",
    "completed_updates",
    "checkpoint_role",
    "selection_policy_id",
    "validation_criterion",
    "validation_seed_policy_id",
    "effective_configuration",
    "state_files",
    "dataset_provenance_file",
    "dataset_provenance_sha256",
})

# Structural recognition does not authorize a selection policy.
CHECKPOINT_ROLES = frozenset({"best", "last"})


def _exact_int(value, name, minimum, maximum=None):
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if value < minimum:
        raise ValueError(f"{name} below minimum")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} above maximum")
    return value


def _identifier(value, name):
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or len(value) > 128
        or re.fullmatch(r"[A-Za-z0-9_.:-]+", value) is None
    ):
        raise ValueError(f"invalid {name}")
    return value


def _validate_class_mapping(mapping, count):
    if type(mapping) is not list or len(mapping) != count:
        raise ValueError("invalid class mapping length or type")

    if any(
        type(name) is not str
        or not name
        or name != name.strip()
        for name in mapping
    ):
        raise ValueError("invalid class mapping label")

    if len(set(mapping)) != count:
        raise ValueError("duplicate class mapping labels")


def _validate_configuration(configuration):
    if type(configuration) is not dict or not configuration:
        raise ValueError("effective configuration must be a nonempty dict")

    # Reject non-JSON values, NaN, infinities, and unsupported types.
    canonical_json_bytes(configuration)

    for name in configuration:
        _identifier(name, "configuration key")


def validate_vae_checkpoint_manifest(metadata: Mapping) -> dict:
    """Validate the exact proposed VAE checkpoint-v1 metadata schema.

    Success establishes schema conformity, not checkpoint validity,
    execution authenticity, file integrity, or policy approval.
    """
    if not isinstance(metadata, Mapping):
        raise TypeError("checkpoint metadata must be a mapping")

    if set(metadata) != REQUIRED_FIELDS:
        missing = sorted(REQUIRED_FIELDS - set(metadata))
        extra = sorted(set(metadata) - REQUIRED_FIELDS)
        raise ValueError(
            "checkpoint metadata fields mismatch: "
            f"missing={missing}, extra={extra}"
        )

    result = deepcopy(dict(metadata))

    constants = {
        "schema_version": SCHEMA_VERSION,
        "schema_type": SCHEMA_TYPE,
        "paper_revision": PAPER_REVISION,
        "model_family": MODEL_FAMILY,
        "image_shape": [40, 40, 3],
        "latent_dim": 100,
        "encoder_dense_units": 256,
        "decoder_feature_channels": 64,
        "observation_variance": 1.0,
        "kl_weight": 1.0,
        "kl_annealing": False,
    }

    for name, expected in constants.items():
        actual = result[name]
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"unsupported VAE constant: {name}")

    dataset_id = result["dataset_id"]

    if type(dataset_id) is not str or dataset_id not in DATASET_CLASS_COUNTS:
        raise ValueError("unsupported governed dataset")

    _exact_int(
        result["experimental_seed"],
        "experimental_seed",
        0,
        2**32 - 1,
    )

    class_count = _exact_int(
        result["class_count"],
        "class_count",
        1,
    )

    if class_count != DATASET_CLASS_COUNTS[dataset_id]:
        raise ValueError("dataset class count mismatch")

    _validate_class_mapping(result["class_mapping"], class_count)

    _identifier(result["architecture_version"], "architecture_version")
    _identifier(result["observation_version"], "observation_version")

    _exact_int(result["completed_epochs"], "completed_epochs", 1)
    _exact_int(result["completed_updates"], "completed_updates", 1)

    if result["checkpoint_role"] not in CHECKPOINT_ROLES:
        raise ValueError("invalid checkpoint role")

    _identifier(result["selection_policy_id"], "selection_policy_id")
    _identifier(result["validation_seed_policy_id"], "validation_seed_policy_id")

    criterion = result["validation_criterion"]
    if type(criterion) is not dict or set(criterion) != {
        "metric",
        "direction",
        "value",
    }:
        raise ValueError("invalid validation criterion structure")

    _identifier(criterion["metric"], "validation metric")

    if criterion["direction"] not in ("minimize", "maximize"):
        raise ValueError("invalid validation direction")

    value = criterion["value"]
    if (
        type(value) is not float
        or not math.isfinite(value)
    ):
        raise ValueError("invalid validation metric value")

    _validate_configuration(result["effective_configuration"])

    state_files = result["state_files"]
    if type(state_files) is not dict or set(state_files) != STATE_FILENAMES:
        raise ValueError("checkpoint state-file inventory mismatch")

    for filename, digest in state_files.items():
        if (
            type(filename) is not str
            or type(digest) is not str
            or SHA256_PATTERN.fullmatch(digest) is None
        ):
            raise ValueError("invalid checkpoint state-file digest")

    if result["dataset_provenance_file"] != VAE_PROVENANCE_FILENAME:
        raise ValueError("invalid dataset provenance filename")

    digest = result["dataset_provenance_sha256"]
    if (
        type(digest) is not str
        or SHA256_PATTERN.fullmatch(digest) is None
    ):
        raise ValueError("invalid dataset provenance digest")

    canonical_json_bytes(result)
    return result


__all__ = [
    "SCHEMA_VERSION",
    "SCHEMA_TYPE",
    "MANIFEST_FILENAME",
    "STATE_FILENAMES",
    "REQUIRED_FIELDS",
    "validate_vae_checkpoint_manifest",
]
