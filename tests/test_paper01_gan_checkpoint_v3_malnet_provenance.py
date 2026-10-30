"""Strict MalNet checkpoint v3 provenance-record tests."""

from copy import deepcopy

import pytest

from trustforge.models.paper01_revision_2026.gan_checkpoint_v3_provenance import (
    SCHEMA_TYPE,
    SCHEMA_VERSION,
    MALNET_SOURCE_FINGERPRINTS,
    MALNET_PARTITION_POLICY,
    MALNET_LIMITATIONS,
    validate_gan_v3_provenance_evidence,
)


def valid_record():
    return {
        "schema_type": SCHEMA_TYPE,
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "malnet_image",
        "source_contract": "malnet_preparation_integrity_baseline",
        "source_fingerprints": dict(MALNET_SOURCE_FINGERPRINTS),
        "partition_policy": dict(MALNET_PARTITION_POLICY),
        "verification_method":
            "sha256_full_prepared_artifact_stream_on_compute_node",
        "verification_scope":
            "baseline_manifest_and_six_prepared_artifacts",
        "verification_limitations": list(MALNET_LIMITATIONS),
    }


def validate(record):
    return validate_gan_v3_provenance_evidence(
        record,
        expected_dataset_id="malnet_image",
    )


def test_valid_record():
    record = valid_record()
    assert validate(record) == record


def test_input_is_not_modified():
    record = valid_record()
    before = deepcopy(record)
    validate(record)
    assert record == before


@pytest.mark.parametrize("name", sorted(MALNET_SOURCE_FINGERPRINTS))
def test_fingerprint_tampering_rejected(name):
    record = valid_record()
    record["source_fingerprints"][name] = "0" * 64
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate(record)


@pytest.mark.parametrize("name", sorted(MALNET_SOURCE_FINGERPRINTS))
def test_missing_fingerprint_rejected(name):
    record = valid_record()
    del record["source_fingerprints"][name]
    with pytest.raises(ValueError, match="inventory mismatch"):
        validate(record)


@pytest.mark.parametrize(
    "name,value",
    [
        ("train_size", 883415),
        ("validation_size", 126203),
        ("test_size", 252405),
        ("train_size", True),
        ("preparation", "unverified"),
        ("policy_id", "unsupported"),
    ],
)
def test_partition_deviation_rejected(name, value):
    record = valid_record()
    record["partition_policy"][name] = value
    with pytest.raises(ValueError, match="partition mismatch"):
        validate(record)


@pytest.mark.parametrize(
    "name,value",
    [
        ("source_contract", "unverified"),
        ("verification_method", "declaration_only"),
        ("verification_scope", "original_image_bytes"),
    ],
)
def test_claim_inflation_rejected(name, value):
    record = valid_record()
    record[name] = value
    with pytest.raises(ValueError):
        validate(record)


def test_limitations_cannot_be_removed():
    record = valid_record()
    record["verification_limitations"] = []
    with pytest.raises(ValueError, match="limitations mismatch"):
        validate(record)


def test_extra_fingerprint_rejected():
    record = valid_record()
    record["source_fingerprints"]["provider_archive.zip"] = "a" * 64
    with pytest.raises(ValueError, match="inventory mismatch"):
        validate(record)


def test_extra_field_rejected():
    record = valid_record()
    record["original_image_bytes_verified"] = True
    with pytest.raises(ValueError, match="fields mismatch"):
        validate(record)


def test_wrong_expected_dataset_rejected():
    with pytest.raises(ValueError, match="dataset identity mismatch"):
        validate_gan_v3_provenance_evidence(
            valid_record(),
            expected_dataset_id="cicmaldroid2020",
        )
