"""Bridge verified dataset preflight to the governed GAN v3 schema.

Pure record transformation. Does not perform training, source I/O,
checkpoint publication, or authorization.
"""

from collections.abc import Mapping

from trustforge.models.paper01_revision_2026.gan_checkpoint_v3_provenance import (
    SCHEMA_TYPE,
    SCHEMA_VERSION,
    USTC_PARTITION_POLICY,
    USTC_LIMITATIONS,
    CIC_PARTITION_POLICY,
    CIC_LIMITATIONS,
    MALNET_PARTITION_POLICY,
    MALNET_LIMITATIONS,
    MALEX_PARTITION_POLICY,
    MALEX_LIMITATIONS,
    MALNET_SOURCE_FINGERPRINTS,
    MALEX_SOURCE_FINGERPRINTS,
    validate_gan_v3_provenance_evidence,
)


def build_gan_v3_provenance(*, dataset_id, verified_preflight):
    """Construct v3 evidence only from a successful governed preflight."""
    if not isinstance(verified_preflight, Mapping):
        raise TypeError("verified preflight must be a mapping")

    if verified_preflight.get("dataset_id") != dataset_id:
        raise ValueError("preflight dataset identity mismatch")

    contracts = {
        "ustc_tfc2016": (
            "ustc_nchw_revision_2026",
            USTC_PARTITION_POLICY,
            USTC_LIMITATIONS,
            "sha256_full_file_stream_on_compute_node",
            "four_original_nchw_numpy_files",
        ),
        "cicmaldroid2020": (
            "cic_authoritative_source_revision_2026",
            CIC_PARTITION_POLICY,
            CIC_LIMITATIONS,
            "sha256_full_file_stream_on_compute_node",
            "authoritative_csv_and_feature_columns_json",
        ),
        "malnet_image": (
            "malnet_preparation_integrity_baseline",
            MALNET_PARTITION_POLICY,
            MALNET_LIMITATIONS,
            "sha256_full_prepared_artifact_stream_on_compute_node",
            "baseline_manifest_and_six_prepared_artifacts",
        ),
        "malex": (
            "governed_adapter_constructor",
            MALEX_PARTITION_POLICY,
            MALEX_LIMITATIONS,
            "production_adapter_integrity_checks_on_compute_node",
            "label_archive_identity_sequences_and_partition_fingerprints",
        ),
    }

    if dataset_id not in contracts:
        raise ValueError("unsupported governed dataset")

    contract, partition, limitations, method, scope = contracts[dataset_id]

    if verified_preflight.get("source_contract") != contract:
        raise ValueError("preflight source contract mismatch")

    if dataset_id in ("ustc_tfc2016", "cicmaldroid2020"):
        fingerprints = verified_preflight.get("verified_sha256")
        if type(fingerprints) is not dict:
            raise ValueError("verified source hashes are missing")
    elif dataset_id == "malnet_image":
        integrity = verified_preflight.get("preparation_integrity")
        if type(integrity) is not dict:
            raise ValueError("MalNet preparation integrity is missing")

        if verified_preflight.get("integrity_settings_verified") is not True:
            raise ValueError("MalNet integrity settings not verified")

        if integrity.get("dataset_id") != "malnet_image":
            raise ValueError("MalNet preparation dataset identity mismatch")

        if integrity.get("baseline_sha256") != MALNET_SOURCE_FINGERPRINTS["baseline.json"]:
            raise ValueError("MalNet baseline fingerprint mismatch")

        if integrity.get("manifest_sha256") != MALNET_SOURCE_FINGERPRINTS["manifest.json"]:
            raise ValueError("MalNet manifest fingerprint mismatch")

        verified_artifacts = integrity.get("verified_artifacts")
        expected_artifacts = {
            name: digest
            for name, digest in MALNET_SOURCE_FINGERPRINTS.items()
            if name not in ("baseline.json", "manifest.json")
        }

        if type(verified_artifacts) is not dict:
            raise ValueError("MalNet verified artifacts missing")

        if verified_artifacts != expected_artifacts:
            raise ValueError("MalNet prepared-artifact fingerprints mismatch")

        if integrity.get("original_image_bytes_verified") is not False:
            raise ValueError("MalNet original-image verification scope mismatch")

        if integrity.get("provider_archive_rehashed") is not False:
            raise ValueError("MalNet provider-archive verification scope mismatch")

        fingerprints = dict(MALNET_SOURCE_FINGERPRINTS)
    else:
        if verified_preflight.get("integrity_settings_verified") is not True:
            raise ValueError("MaleX integrity settings not verified")
        fingerprints = MALEX_SOURCE_FINGERPRINTS

    record = {
        "schema_type": SCHEMA_TYPE,
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "source_contract": contract,
        "source_fingerprints": dict(fingerprints),
        "partition_policy": dict(partition),
        "verification_method": method,
        "verification_scope": scope,
        "verification_limitations": list(limitations),
    }

    return validate_gan_v3_provenance_evidence(
        record,
        expected_dataset_id=dataset_id,
    )
