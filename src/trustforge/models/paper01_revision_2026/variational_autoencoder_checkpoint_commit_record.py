"""Candidate VAE checkpoint generation commit-record contract.

Defines canonical publication evidence without performing filesystem I/O,
authenticating a producer, or authorizing scientific use.
"""

from __future__ import annotations

import hashlib
import re

from trustforge.provenance import canonical_json_bytes

from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_manifest import (
    MANIFEST_FILENAME,
)


COMMIT_SCHEMA_VERSION = "1.0"
COMMIT_SCHEMA_TYPE = "paper01_revision2026_vae_generation_commit"

COMMIT_FIELDS = frozenset({
    "schema_version",
    "schema_type",
    "generation_id",
    "manifest_filename",
    "manifest_sha256",
})

_GENERATION_PATTERN = re.compile(r"[0-9a-f]{32}\Z")
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def validate_vae_generation_id(generation_id):
    """Require one canonical, path-free 128-bit generation identifier."""
    if (
        type(generation_id) is not str
        or _GENERATION_PATTERN.fullmatch(generation_id) is None
    ):
        raise ValueError("invalid VAE checkpoint generation ID")

    return generation_id


def validate_vae_commit_record(record):
    """Validate the exact structural commit-record contract."""
    if type(record) is not dict:
        raise TypeError("VAE commit record must be a dictionary")

    if set(record) != COMMIT_FIELDS:
        raise ValueError("VAE commit record field mismatch")

    if record["schema_version"] != COMMIT_SCHEMA_VERSION:
        raise ValueError("unsupported VAE commit schema version")

    if record["schema_type"] != COMMIT_SCHEMA_TYPE:
        raise ValueError("unsupported VAE commit schema type")

    validate_vae_generation_id(record["generation_id"])

    if record["manifest_filename"] != MANIFEST_FILENAME:
        raise ValueError("invalid committed VAE manifest filename")

    digest = record["manifest_sha256"]

    if (
        type(digest) is not str
        or _DIGEST_PATTERN.fullmatch(digest) is None
    ):
        raise ValueError("invalid VAE commit manifest digest")

    return dict(record)


def build_vae_commit_record(*, generation_id, manifest_bytes):
    """Bind one generation ID to the exact canonical manifest bytes."""
    validate_vae_generation_id(generation_id)

    if type(manifest_bytes) is not bytes or not manifest_bytes:
        raise ValueError("VAE manifest bytes must be nonempty bytes")

    record = {
        "schema_version": COMMIT_SCHEMA_VERSION,
        "schema_type": COMMIT_SCHEMA_TYPE,
        "generation_id": generation_id,
        "manifest_filename": MANIFEST_FILENAME,
        "manifest_sha256": hashlib.sha256(
            manifest_bytes
        ).hexdigest(),
    }

    return validate_vae_commit_record(record)


def encode_vae_commit_record(record):
    """Serialize a validated record using canonical JSON plus newline."""
    validated = validate_vae_commit_record(record)
    return canonical_json_bytes(validated) + b"\n"


__all__ = [
    "COMMIT_SCHEMA_VERSION",
    "COMMIT_SCHEMA_TYPE",
    "COMMIT_FIELDS",
    "validate_vae_generation_id",
    "validate_vae_commit_record",
    "build_vae_commit_record",
    "encode_vae_commit_record",
]
