"""Dataset-provenance validation for revised Paper 1 VAE checkpoints.

This module reuses established revision-2026 dataset-specific
evidence contracts without importing TensorFlow, reading datasets,
writing checkpoint artifacts, or authorizing training.

The underlying historical provenance schema retains its GAN-specific
identifier. This adapter does not relabel or falsify that evidence.
"""

from __future__ import annotations

from collections.abc import Mapping

from trustforge.models.paper01_revision_2026.gan_checkpoint_manifest import (
    DATASET_CLASS_COUNTS,
)
from trustforge.models.paper01_revision_2026.gan_checkpoint_v3_provenance import (
    REQUIRED_FIELDS as SOURCE_REQUIRED_FIELDS,
    SCHEMA_TYPE as SOURCE_SCHEMA_TYPE,
    SCHEMA_VERSION as SOURCE_SCHEMA_VERSION,
    validate_gan_v3_provenance_evidence,
)


VAE_PROVENANCE_ADAPTER_VERSION = "1.0"

VAE_PROVENANCE_FILENAME = "vae_dataset_provenance.json"

SUPPORTED_DATASET_IDS = frozenset(DATASET_CLASS_COUNTS)


def validate_vae_checkpoint_provenance(
    evidence: Mapping,
    *,
    expected_dataset_id: str,
) -> dict:
    """Validate existing governed dataset evidence for VAE use.

    Acceptance means that the evidence conforms to the established
    dataset-specific record contract.

    Acceptance does not independently rehash source files, prove
    historical training inputs, authenticate checkpoint origin,
    or grant permission to train or publish checkpoints.
    """
    if type(expected_dataset_id) is not str:
        raise TypeError("dataset ID must be a string")

    if expected_dataset_id not in SUPPORTED_DATASET_IDS:
        raise ValueError("unsupported governed VAE dataset")

    if not isinstance(evidence, Mapping):
        raise TypeError("provenance evidence must be a mapping")

    if set(evidence) != SOURCE_REQUIRED_FIELDS:
        raise ValueError("provenance field inventory mismatch")

    if evidence["schema_type"] != SOURCE_SCHEMA_TYPE:
        raise ValueError("unsupported source provenance schema type")

    if evidence["schema_version"] != SOURCE_SCHEMA_VERSION:
        raise ValueError("unsupported source provenance schema version")

    validated = validate_gan_v3_provenance_evidence(
        evidence,
        expected_dataset_id=expected_dataset_id,
    )

    if validated["dataset_id"] != expected_dataset_id:
        raise ValueError("provenance dataset mismatch")

    return validated


__all__ = [
    "VAE_PROVENANCE_ADAPTER_VERSION",
    "VAE_PROVENANCE_FILENAME",
    "SUPPORTED_DATASET_IDS",
    "validate_vae_checkpoint_provenance",
]
