"""Strict checkpoint v3 manifest contract.

Preserves v2 scientific and state-file constraints while requiring
an independently bound dataset-provenance artifact.

Metadata validation does not read dataset bytes or authorize training.
"""

from collections.abc import Mapping
from copy import deepcopy

from trustforge.provenance import canonical_json_bytes
from trustforge.models.paper01_revision_2026.gan_checkpoint_manifest import (
    REQUIRED_FIELDS as V2_REQUIRED_FIELDS,
    SHA256_PATTERN,
    validate_gan_checkpoint_manifest,
)

SCHEMA_VERSION = "3.0"
PROVENANCE_FILENAME = "gan_dataset_provenance.json"

REQUIRED_FIELDS = frozenset(V2_REQUIRED_FIELDS) | {
    "dataset_provenance_file",
    "dataset_provenance_sha256",
}


def validate_gan_checkpoint_v3_manifest(metadata: Mapping) -> dict:
    """Fail closed on unsupported or incomplete checkpoint metadata."""
    if not isinstance(metadata, Mapping):
        raise TypeError("checkpoint v3 metadata must be a mapping")

    if set(metadata) != REQUIRED_FIELDS:
        raise ValueError("checkpoint v3 metadata fields mismatch")

    result = deepcopy(dict(metadata))

    if type(result["schema_version"]) is not str:
        raise TypeError("checkpoint v3 schema version must be a string")

    if result["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint v3 schema version")

    if (
        type(result["dataset_provenance_file"]) is not str
        or result["dataset_provenance_file"] != PROVENANCE_FILENAME
    ):
        raise ValueError("invalid checkpoint provenance filename")

    digest = result["dataset_provenance_sha256"]
    if (
        type(digest) is not str
        or SHA256_PATTERN.fullmatch(digest) is None
    ):
        raise ValueError("invalid checkpoint provenance SHA-256")

    # Apply the existing v2 checks to all inherited fields.
    # This is validation only: no v2 checkpoint is changed or published.
    inherited = {
        name: result[name] for name in V2_REQUIRED_FIELDS
    }
    inherited["schema_version"] = "2.0"
    validate_gan_checkpoint_manifest(inherited)

    canonical_json_bytes(result)
    return result
