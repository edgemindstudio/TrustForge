"""Manifest contract for revision-2026 conditional GAN checkpoints.

This module validates metadata only. TensorFlow checkpoint persistence
and restoration are implemented separately.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from trustforge.provenance import canonical_json_bytes


SCHEMA_VERSION = "2.0"
SCHEMA_TYPE = "paper01_revision2026_conditional_gan"
MODEL_FAMILY = "conditional_gan"
PAPER_REVISION = "paper01_revision_2026"

CHECKPOINT_SELECTION = "terminal_governed_iteration"

DATASET_CLASS_COUNTS = {
    "ustc_tfc2016": 9,
    "cicmaldroid2020": 5,
    "malnet_image": 47,
    "malex": 2,
}

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
STATE_FILE_PATTERN = re.compile(
    r"^model_state\.ckpt\.(?:index|data-[0-9]+-of-[0-9]+)$"
)

REQUIRED_FIELDS = frozenset({
    "schema_version",
    "schema_type",
    "paper_revision",
    "model_family",
    "dataset_id",
    "experimental_seed",
    "class_mapping",
    "class_count",
    "generator_iteration",
    "discriminator_updates",
    "planned_generator_iterations",
    "checkpoint_selection",
    "image_shape",
    "latent_dim",
    "learning_rate",
    "beta_1",
    "class_sampling_alpha",
    "batch_size",
    "state_files",
    "exposure_evidence_file",
    "exposure_evidence_sha256",
})


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an exact integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def validate_gan_checkpoint_manifest(metadata: Mapping) -> dict:
    """Validate an exact, authoritative feasibility-checkpoint manifest."""
    if not isinstance(metadata, Mapping):
        raise TypeError("checkpoint metadata must be a mapping")

    if set(metadata) != REQUIRED_FIELDS:
        missing = sorted(REQUIRED_FIELDS - set(metadata))
        extra = sorted(set(metadata) - REQUIRED_FIELDS)
        raise ValueError(
            f"checkpoint metadata fields mismatch: "
            f"missing={missing}, extra={extra}"
        )

    result = dict(metadata)

    constants = {
        "schema_version": SCHEMA_VERSION,
        "schema_type": SCHEMA_TYPE,
        "paper_revision": PAPER_REVISION,
        "model_family": MODEL_FAMILY,
        "checkpoint_selection": CHECKPOINT_SELECTION,
        "latent_dim": 100,
        "learning_rate": 0.0002,
        "beta_1": 0.5,
        "class_sampling_alpha": 0.85,
        "batch_size": 256,
    }

    for name, expected in constants.items():
        value = result[name]
        if isinstance(value, bool) or type(value) is not type(expected):
            raise TypeError(f"{name}: invalid type")
        if value != expected:
            raise ValueError(f"{name}: unsupported value")

    dataset_id = result["dataset_id"]
    if not isinstance(dataset_id, str):
        raise TypeError("dataset_id must be a string")
    if dataset_id not in DATASET_CLASS_COUNTS:
        raise ValueError("unsupported governed dataset")

    seed = _integer(result["experimental_seed"], "experimental_seed", 0)
    if seed > 2**32 - 1:
        raise ValueError("experimental_seed outside uint32 range")

    class_count = _integer(result["class_count"], "class_count", 1)
    if class_count != DATASET_CLASS_COUNTS[dataset_id]:
        raise ValueError("dataset class count mismatch")

    mapping = result["class_mapping"]
    if (
        not isinstance(mapping, (list, tuple))
        or len(mapping) != class_count
        or any(
            not isinstance(label, str)
            or not label.strip()
            or label != label.strip()
            for label in mapping
        )
        or len(set(mapping)) != class_count
    ):
        raise ValueError("invalid ordered class mapping")

    if result["image_shape"] != [40, 40, 3]:
        raise ValueError("invalid canonical image shape")

    completed = _integer(
        result["generator_iteration"], "generator_iteration", 1
    )
    planned = _integer(
        result["planned_generator_iterations"],
        "planned_generator_iterations",
        1,
    )
    updates = _integer(
        result["discriminator_updates"],
        "discriminator_updates",
        2,
    )

    if completed != planned:
        raise ValueError("terminal iteration does not match declared limit")
    if updates != 2 * completed:
        raise ValueError("discriminator update count mismatch")

    if result["exposure_evidence_file"] != "gan_exposure_evidence.json":
        raise ValueError("unsupported GAN exposure evidence filename")

    digest = result["exposure_evidence_sha256"]
    if (
        not isinstance(digest, str)
        or SHA256_PATTERN.fullmatch(digest) is None
    ):
        raise ValueError("invalid GAN exposure evidence SHA-256")

    state_files = result["state_files"]
    if not isinstance(state_files, dict) or len(state_files) < 2:
        raise ValueError("missing checkpoint file inventory")

    if "model_state.ckpt.index" not in state_files:
        raise ValueError("missing checkpoint index")

    if not any(
        name.startswith("model_state.ckpt.data-")
        for name in state_files
    ):
        raise ValueError("missing checkpoint data shard")

    for name, digest in state_files.items():
        if (
            not isinstance(name, str)
            or STATE_FILE_PATTERN.fullmatch(name) is None
            or not isinstance(digest, str)
            or SHA256_PATTERN.fullmatch(digest) is None
        ):
            raise ValueError("invalid checkpoint filename or digest")

    # Filenames have already passed STATE_FILE_PATTERN validation.
    # Require a complete, contiguous, internally consistent shard set.
    shard_indices = []
    shard_totals = set()
    prefix = "model_state.ckpt.data-"

    for name in state_files:
        if not name.startswith(prefix):
            continue

        suffix = name[len(prefix):]
        index_text, total_text = suffix.split("-of-", 1)
        shard_indices.append(int(index_text))
        shard_totals.add(int(total_text))

    if len(shard_totals) != 1:
        raise ValueError("inconsistent checkpoint shard totals")

    total = next(iter(shard_totals))

    if total < 1 or total != len(shard_indices):
        raise ValueError("incomplete checkpoint shard inventory")

    if set(shard_indices) != set(range(total)):
        raise ValueError("missing or duplicate checkpoint shard index")

    # Prove metadata has a supported canonical JSON representation.
    canonical_json_bytes(result)
    return result
