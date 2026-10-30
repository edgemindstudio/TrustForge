"""Negative and positive tests for GAN v3 provenance evidence."""

from copy import deepcopy

import pytest

from trustforge.models.paper01_revision_2026.gan_checkpoint_v3_provenance import (
    SCHEMA_TYPE,
    SCHEMA_VERSION,
    USTC_LIMITATIONS,
    USTC_PARTITION_POLICY,
    validate_gan_v3_provenance_evidence,
)
from trustforge.models.paper01_revision_2026.gan_dataset_provenance import (
    USTC_NCHW_SOURCE_SHA256,
)


def valid_record():
    return {
        "schema_type": SCHEMA_TYPE,
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "ustc_tfc2016",
        "source_contract": "ustc_nchw_revision_2026",
        "source_fingerprints": dict(
            USTC_NCHW_SOURCE_SHA256
        ),
        "partition_policy": dict(USTC_PARTITION_POLICY),
        "verification_method":
            "sha256_full_file_stream_on_compute_node",
        "verification_scope":
            "four_original_nchw_numpy_files",
        "verification_limitations": list(USTC_LIMITATIONS),
    }


def validate(record):
    return validate_gan_v3_provenance_evidence(
        record,
        expected_dataset_id="ustc_tfc2016",
    )


def test_valid_record():
    record = valid_record()
    assert validate(record) == record


def test_validation_does_not_modify_input():
    record = valid_record()
    original = deepcopy(record)
    validate(record)
    assert record == original


def test_nonmapping_rejected():
    with pytest.raises(TypeError):
        validate(None)


@pytest.mark.parametrize(
    "field",
    sorted(valid_record()),
)
def test_missing_field_rejected(field):
    record = valid_record()
    del record[field]
    with pytest.raises(ValueError):
        validate(record)


def test_extra_field_rejected():
    record = valid_record()
    record["unapproved"] = True
    with pytest.raises(ValueError):
        validate(record)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_type", "incorrect"),
        ("schema_version", "2.0"),
        ("dataset_id", "malex"),
        ("source_contract", "ustc_nhwc_revision_2026"),
        ("verification_method", "unverified"),
        ("verification_scope", "all_training_images"),
    ],
)
def test_invalid_identity_or_scope_rejected(field, value):
    record = valid_record()
    record[field] = value
    with pytest.raises(ValueError):
        validate(record)


def test_wrong_expected_dataset_rejected():
    with pytest.raises(ValueError):
        validate_gan_v3_provenance_evidence(
            valid_record(),
            expected_dataset_id="malex",
        )


def test_fingerprint_tampering_rejected():
    record = valid_record()
    record["source_fingerprints"]["train_data.npy"] = "0" * 64
    with pytest.raises(ValueError):
        validate(record)


def test_missing_fingerprint_rejected():
    record = valid_record()
    del record["source_fingerprints"]["test_labels.npy"]
    with pytest.raises(ValueError):
        validate(record)


def test_extra_fingerprint_rejected():
    record = valid_record()
    record["source_fingerprints"]["extra.npy"] = "a" * 64
    with pytest.raises(ValueError):
        validate(record)


@pytest.mark.parametrize(
    "field,value",
    [
        ("partition_seed", 43),
        ("partition_seed", True),
        ("validation_fraction", 0.2),
        ("validation_fraction", 0),
        ("train_size", 81000),
        ("test_size", 8100),
    ],
)
def test_partition_deviation_rejected(field, value):
    record = valid_record()
    record["partition_policy"][field] = value
    with pytest.raises(ValueError):
        validate(record)


def test_missing_partition_field_rejected():
    record = valid_record()
    del record["partition_policy"]["validation_size"]
    with pytest.raises(ValueError):
        validate(record)


def test_limitations_cannot_be_removed():
    record = valid_record()
    record["verification_limitations"] = []
    with pytest.raises(ValueError):
        validate(record)


def test_unimplemented_dataset_rejected():
    record = valid_record()
    record["dataset_id"] = "unsupported_dataset"
    with pytest.raises(ValueError, match="not yet implemented"):
        validate_gan_v3_provenance_evidence(
            record,
            expected_dataset_id="unsupported_dataset",
        )
