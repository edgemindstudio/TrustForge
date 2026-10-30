"""Strict MaleX checkpoint-v3 evidence record tests."""

from copy import deepcopy
import pytest

from trustforge.models.paper01_revision_2026.gan_checkpoint_v3_provenance import (
    SCHEMA_TYPE,
    SCHEMA_VERSION,
    MALEX_SOURCE_FINGERPRINTS,
    MALEX_PARTITION_POLICY,
    MALEX_LIMITATIONS,
    validate_gan_v3_provenance_evidence,
)


def valid_record():
    return {
        "schema_type": SCHEMA_TYPE,
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "malex",
        "source_contract": "governed_adapter_constructor",
        "source_fingerprints": dict(MALEX_SOURCE_FINGERPRINTS),
        "partition_policy": dict(MALEX_PARTITION_POLICY),
        "verification_method":
            "production_adapter_integrity_checks_on_compute_node",
        "verification_scope":
            "label_archive_identity_sequences_and_partition_fingerprints",
        "verification_limitations": list(MALEX_LIMITATIONS),
    }


def validate(record):
    return validate_gan_v3_provenance_evidence(
        record,
        expected_dataset_id="malex",
    )


def test_valid_record():
    record = valid_record()
    assert validate(record) == record


def test_input_not_modified():
    record = valid_record()
    original = deepcopy(record)
    validate(record)
    assert record == original


@pytest.mark.parametrize(
    "name", sorted(MALEX_SOURCE_FINGERPRINTS)
)
def test_fingerprint_tampering_rejected(name):
    record = valid_record()
    record["source_fingerprints"][name] = "0" * 64
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate(record)


@pytest.mark.parametrize(
    "name", sorted(MALEX_SOURCE_FINGERPRINTS)
)
def test_missing_fingerprint_rejected(name):
    record = valid_record()
    del record["source_fingerprints"][name]
    with pytest.raises(ValueError, match="inventory mismatch"):
        validate(record)


@pytest.mark.parametrize(
    "name,value",
    [
        ("train_size", 731050),
        ("validation_size", 104434),
        ("test_size", 208873),
        ("cross_class_conflicting_identities", 0),
        ("post_exclusion_benign_count", 179725),
        ("post_exclusion_malware_count", 864669),
        ("train_size", True),
        ("policy_id", "unverified"),
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
        ("source_contract", "independent_image_byte_audit"),
        ("verification_method", "declaration_only"),
        ("verification_scope", "all_original_image_bytes"),
    ],
)
def test_overstated_scope_rejected(name, value):
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
    record["source_fingerprints"]["image_bytes"] = "a" * 64
    with pytest.raises(ValueError, match="inventory mismatch"):
        validate(record)


def test_missing_field_rejected():
    record = valid_record()
    del record["verification_scope"]
    with pytest.raises(ValueError, match="fields mismatch"):
        validate(record)


def test_wrong_dataset_rejected():
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_gan_v3_provenance_evidence(
            valid_record(),
            expected_dataset_id="malnet_image",
        )
