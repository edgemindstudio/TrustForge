"""CICMalDroid v3 provenance record-validation tests."""

from copy import deepcopy

import pytest

from trustforge.models.paper01_revision_2026.gan_checkpoint_v3_provenance import (
    SCHEMA_TYPE,
    SCHEMA_VERSION,
    CIC_SOURCE_FINGERPRINTS,
    CIC_PARTITION_POLICY,
    CIC_LIMITATIONS,
    validate_gan_v3_provenance_evidence,
)


def valid_record():
    return {
        "schema_type": SCHEMA_TYPE,
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "cicmaldroid2020",
        "source_contract": "cic_authoritative_source_revision_2026",
        "source_fingerprints": dict(CIC_SOURCE_FINGERPRINTS),
        "partition_policy": dict(CIC_PARTITION_POLICY),
        "verification_method":
            "sha256_full_file_stream_on_compute_node",
        "verification_scope":
            "authoritative_csv_and_feature_columns_json",
        "verification_limitations": list(CIC_LIMITATIONS),
    }


def validate(record):
    return validate_gan_v3_provenance_evidence(
        record,
        expected_dataset_id="cicmaldroid2020",
    )


def test_cic_valid_record():
    record = valid_record()
    assert validate(record) == record


def test_cic_input_not_modified():
    record = valid_record()
    original = deepcopy(record)
    validate(record)
    assert record == original


@pytest.mark.parametrize("field", sorted(valid_record()))
def test_cic_missing_field_rejected(field):
    record = valid_record()
    del record[field]
    with pytest.raises(ValueError):
        validate(record)


@pytest.mark.parametrize(
    "field",
    ["source_csv", "feature_columns_path"],
)
def test_cic_source_fingerprint_tampering_rejected(field):
    record = valid_record()
    record["source_fingerprints"][field] = "0" * 64
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate(record)


@pytest.mark.parametrize(
    "field,value",
    [
        ("partition_seed", 43),
        ("partition_seed", True),
        ("train_fraction", 0.8),
        ("validation_fraction", 0.15),
        ("test_fraction", 0.15),
        ("train_size", 8118),
        ("validation_size", 1740),
        ("test_size", 1740),
    ],
)
def test_cic_partition_deviation_rejected(field, value):
    record = valid_record()
    record["partition_policy"][field] = value
    with pytest.raises(ValueError, match="partition policy mismatch"):
        validate(record)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_contract", "incorrect"),
        ("verification_method", "unverified"),
        ("verification_scope", "all_original_image_bytes"),
    ],
)
def test_cic_scope_or_identity_tampering_rejected(field, value):
    record = valid_record()
    record[field] = value
    with pytest.raises(ValueError):
        validate(record)


def test_cic_limitations_cannot_be_removed():
    record = valid_record()
    record["verification_limitations"] = []
    with pytest.raises(ValueError, match="limitations mismatch"):
        validate(record)


def test_cic_wrong_expected_dataset_rejected():
    with pytest.raises(ValueError, match="dataset identity mismatch"):
        validate_gan_v3_provenance_evidence(
            valid_record(),
            expected_dataset_id="malex",
        )


def test_cic_unapproved_extra_field_rejected():
    record = valid_record()
    record["unsupported"] = True
    with pytest.raises(ValueError, match="fields mismatch"):
        validate(record)
