"""VAE checkpoint provenance regression tests.

All evidence is constructed from existing contract constants.
No dataset bytes, checkpoint files, or training are involved.
"""

from copy import deepcopy

import pytest

from trustforge.models.paper01_revision_2026 import (
    gan_checkpoint_v3_provenance as source,
    gan_dataset_provenance as ustc,
    variational_autoencoder_checkpoint_provenance as vae,
)


CONTRACTS = {
    "ustc_tfc2016": (
        "ustc_nchw_revision_2026",
        ustc.USTC_NCHW_SOURCE_SHA256,
        source.USTC_PARTITION_POLICY,
        "sha256_full_file_stream_on_compute_node",
        "four_original_nchw_numpy_files",
        source.USTC_LIMITATIONS,
    ),
    "cicmaldroid2020": (
        "cic_authoritative_source_revision_2026",
        source.CIC_SOURCE_FINGERPRINTS,
        source.CIC_PARTITION_POLICY,
        "sha256_full_file_stream_on_compute_node",
        "authoritative_csv_and_feature_columns_json",
        source.CIC_LIMITATIONS,
    ),
    "malnet_image": (
        "malnet_preparation_integrity_baseline",
        source.MALNET_SOURCE_FINGERPRINTS,
        source.MALNET_PARTITION_POLICY,
        "sha256_full_prepared_artifact_stream_on_compute_node",
        "baseline_manifest_and_six_prepared_artifacts",
        source.MALNET_LIMITATIONS,
    ),
    "malex": (
        "governed_adapter_constructor",
        source.MALEX_SOURCE_FINGERPRINTS,
        source.MALEX_PARTITION_POLICY,
        "production_adapter_integrity_checks_on_compute_node",
        "label_archive_identity_sequences_and_partition_fingerprints",
        source.MALEX_LIMITATIONS,
    ),
}


def evidence_for(dataset_id):
    (
        contract,
        fingerprints,
        partition,
        method,
        scope,
        limitations,
    ) = CONTRACTS[dataset_id]

    return {
        "schema_type": source.SCHEMA_TYPE,
        "schema_version": source.SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "source_contract": contract,
        "source_fingerprints": deepcopy(fingerprints),
        "partition_policy": deepcopy(partition),
        "verification_method": method,
        "verification_scope": scope,
        "verification_limitations": list(limitations),
    }


def validate(dataset_id, evidence):
    return vae.validate_vae_checkpoint_provenance(
        evidence,
        expected_dataset_id=dataset_id,
    )


@pytest.mark.parametrize("dataset_id", sorted(CONTRACTS))
def test_governed_dataset_evidence_accepted(dataset_id):
    evidence = evidence_for(dataset_id)
    assert validate(dataset_id, evidence) == evidence


@pytest.mark.parametrize("dataset_id", sorted(CONTRACTS))
def test_dataset_identity_mismatch_rejected(dataset_id):
    evidence = evidence_for(dataset_id)
    evidence["dataset_id"] = "invalid_dataset"

    with pytest.raises((ValueError, TypeError)):
        validate(dataset_id, evidence)


@pytest.mark.parametrize("dataset_id", sorted(CONTRACTS))
def test_source_fingerprint_tampering_rejected(dataset_id):
    evidence = evidence_for(dataset_id)
    key = next(iter(evidence["source_fingerprints"]))
    evidence["source_fingerprints"][key] = "0" * 64

    with pytest.raises((ValueError, TypeError)):
        validate(dataset_id, evidence)


@pytest.mark.parametrize("dataset_id", sorted(CONTRACTS))
def test_partition_policy_tampering_rejected(dataset_id):
    evidence = evidence_for(dataset_id)
    evidence["partition_policy"]["policy_id"] = "invalid_policy"

    with pytest.raises((ValueError, TypeError)):
        validate(dataset_id, evidence)


@pytest.mark.parametrize("dataset_id", sorted(CONTRACTS))
def test_missing_field_rejected(dataset_id):
    evidence = evidence_for(dataset_id)
    del evidence["verification_scope"]

    with pytest.raises(ValueError):
        validate(dataset_id, evidence)


@pytest.mark.parametrize("dataset_id", sorted(CONTRACTS))
def test_unexpected_field_rejected(dataset_id):
    evidence = evidence_for(dataset_id)
    evidence["unauthorized_field"] = True

    with pytest.raises(ValueError):
        validate(dataset_id, evidence)


@pytest.mark.parametrize("dataset_id", sorted(CONTRACTS))
def test_schema_version_mismatch_rejected(dataset_id):
    evidence = evidence_for(dataset_id)
    evidence["schema_version"] = "999.0"

    with pytest.raises(ValueError):
        validate(dataset_id, evidence)


@pytest.mark.parametrize("dataset_id", sorted(CONTRACTS))
def test_schema_type_mismatch_rejected(dataset_id):
    evidence = evidence_for(dataset_id)
    evidence["schema_type"] = "unrelated_provenance"

    with pytest.raises(ValueError):
        validate(dataset_id, evidence)


@pytest.mark.parametrize("dataset_id", sorted(CONTRACTS))
def test_verification_method_tampering_rejected(dataset_id):
    evidence = evidence_for(dataset_id)
    evidence["verification_method"] = "not_verified"

    with pytest.raises(ValueError):
        validate(dataset_id, evidence)


@pytest.mark.parametrize("dataset_id", sorted(CONTRACTS))
def test_verification_limitations_tampering_rejected(dataset_id):
    evidence = evidence_for(dataset_id)
    evidence["verification_limitations"] = []

    with pytest.raises(ValueError):
        validate(dataset_id, evidence)


@pytest.mark.parametrize("dataset_id", sorted(CONTRACTS))
def test_validation_returns_detached_record(dataset_id):
    evidence = evidence_for(dataset_id)
    validated = validate(dataset_id, evidence)

    validated["source_fingerprints"].clear()

    assert evidence["source_fingerprints"]


@pytest.mark.parametrize(
    "invalid_dataset",
    ["unknown", "", None, 42, True],
)
def test_unsupported_requested_dataset_rejected(invalid_dataset):
    evidence = evidence_for("ustc_tfc2016")

    with pytest.raises((ValueError, TypeError)):
        vae.validate_vae_checkpoint_provenance(
            evidence,
            expected_dataset_id=invalid_dataset,
        )


@pytest.mark.parametrize(
    "invalid_evidence",
    [None, [], "invalid", 42],
)
def test_nonmapping_evidence_rejected(invalid_evidence):
    with pytest.raises(TypeError):
        validate("ustc_tfc2016", invalid_evidence)
