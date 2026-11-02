"""Pure VAE generation commit-record contract regressions.

No checkpoint publication, dataset access, or model training.
"""

import hashlib
import json

import pytest

from trustforge.provenance import canonical_json_bytes

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_checkpoint_commit_record as commit,
)
from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_manifest import (
    MANIFEST_FILENAME,
)


GENERATION_ID = "0123456789abcdef0123456789abcdef"
MANIFEST_BYTES = b'{"synthetic":"manifest"}\n'


def valid_record():
    return commit.build_vae_commit_record(
        generation_id=GENERATION_ID,
        manifest_bytes=MANIFEST_BYTES,
    )


def test_commit_record_binds_exact_manifest_bytes():
    record = valid_record()

    assert record["generation_id"] == GENERATION_ID
    assert record["manifest_filename"] == MANIFEST_FILENAME
    assert record["manifest_sha256"] == hashlib.sha256(
        MANIFEST_BYTES
    ).hexdigest()

    assert commit.validate_vae_commit_record(record) == record


def test_commit_record_encoding_is_canonical():
    record = valid_record()
    raw = commit.encode_vae_commit_record(record)

    assert raw == canonical_json_bytes(record) + b"\n"
    assert json.loads(raw) == record


def test_manifest_mutation_changes_digest_binding():
    record = valid_record()
    changed = MANIFEST_BYTES + b"modified"

    assert hashlib.sha256(changed).hexdigest() != (
        record["manifest_sha256"]
    )


@pytest.mark.parametrize(
    "generation_id",
    [
        "",
        "a" * 31,
        "a" * 33,
        "A" * 32,
        "../" + "a" * 29,
        "g" * 32,
        42,
    ],
)
def test_malformed_generation_id_rejected(generation_id):
    with pytest.raises(ValueError, match="generation ID"):
        commit.validate_vae_generation_id(generation_id)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "2.0"),
        ("schema_type", "unsupported_schema"),
        ("manifest_filename", "../manifest.json"),
        ("manifest_sha256", "x" * 64),
        ("manifest_sha256", "A" * 64),
        ("manifest_sha256", "a" * 63),
    ],
)
def test_invalid_commit_record_fields_rejected(field, value):
    record = valid_record()
    record[field] = value

    with pytest.raises(ValueError):
        commit.validate_vae_commit_record(record)


def test_unexpected_commit_field_rejected():
    record = valid_record()
    record["scientifically_approved"] = True

    with pytest.raises(ValueError, match="field mismatch"):
        commit.validate_vae_commit_record(record)


def test_missing_commit_field_rejected():
    record = valid_record()
    record.pop("manifest_sha256")

    with pytest.raises(ValueError, match="field mismatch"):
        commit.validate_vae_commit_record(record)


def test_non_dictionary_record_rejected():
    with pytest.raises(TypeError):
        commit.validate_vae_commit_record([])


@pytest.mark.parametrize(
    "manifest_bytes",
    [b"", "not-bytes", None],
)
def test_invalid_manifest_bytes_rejected(manifest_bytes):
    with pytest.raises(ValueError, match="manifest bytes"):
        commit.build_vae_commit_record(
            generation_id=GENERATION_ID,
            manifest_bytes=manifest_bytes,
        )
