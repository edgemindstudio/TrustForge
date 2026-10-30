"""Strict, independent GAN v3 provenance evidence validation.

This module validates evidence records, not dataset bytes.
It does not import TensorFlow, construct models, or authorize training.
"""

from collections.abc import Mapping
from copy import deepcopy
import re

from trustforge.models.paper01_revision_2026.gan_dataset_provenance import (
    USTC_NCHW_SOURCE_SHA256,
)


SCHEMA_TYPE = "paper01_revision2026_gan_dataset_provenance"
SCHEMA_VERSION = "1.0"

REQUIRED_FIELDS = frozenset({
    "schema_type",
    "schema_version",
    "dataset_id",
    "source_contract",
    "source_fingerprints",
    "partition_policy",
    "verification_method",
    "verification_scope",
    "verification_limitations",
})

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

CIC_SOURCE_FINGERPRINTS = {
    "source_csv":
        "3415babdd76a2b77c400a9ec3a8a6057705d5a57a9e91753191ecc2946df7c00",
    "feature_columns_path":
        "7a6045fe10bfe27c1faee98d729de470401b7cf733ccccc107499f11dea8fc8f",
}

CIC_PARTITION_POLICY = {
    "policy_id": "cic_case3_revision_2026",
    "partition_seed": 42,
    "train_fraction": 0.7,
    "validation_fraction": 0.1,
    "test_fraction": 0.2,
    "train_size": 8117,
    "validation_size": 1158,
    "test_size": 2323,
}

CIC_LIMITATIONS = (
    "does_not_authenticate_checkpoint_origin",
    "does_not_prove_continuous_source_immutability",
    "does_not_prove_training_used_only_verified_bytes",
    "does_not_independently_verify_partition_execution",
)

MALNET_SOURCE_FINGERPRINTS = {
    "baseline.json":
        "a66be1309058de641f120fa96909cf151a565db2912f465a84797db683655e7f",
    "manifest.json":
        "182eabd0c5beb36cda1f27c02a704a12996cb74d7122456d3ebe2a884e084cc8",
    "train.txt":
        "4368c31d2e7eda5f5a12558e42f607751d8fd8c7ba9ab6d599e1989030cdf51e",
    "train_offsets.npy":
        "f864111888b734d7c798c10778e4736bd85511211f19a335be3d670378ff5549",
    "validation.txt":
        "7f992f91a6499c8577ada80510b1373d03f566c0097cf98b47a9cc620bce3146",
    "validation_offsets.npy":
        "773d977a2c9d9b80b3a7157165ade72e9f06af9939ecdbaac46c8ac21abaee48",
    "test.txt":
        "9f44e0958880f2529e2e7ee2d8ec986e3c84e8b242e3624438e934a9568272ee",
    "test_offsets.npy":
        "c4f61cd7d03975bfed4f53539cd19564e3516e0a2c9ef617f8bda5f979694709",
}

MALNET_PARTITION_POLICY = {
    "policy_id": "malnet_provider_split_revision_2026",
    "preparation": "type-1.0-prepared",
    "train_size": 883416,
    "validation_size": 126202,
    "test_size": 252406,
}

MALNET_LIMITATIONS = (
    "does_not_authenticate_checkpoint_origin",
    "does_not_prove_continuous_source_immutability",
    "does_not_prove_training_used_only_verified_bytes",
    "original_image_bytes_not_rehashed",
    "provider_archive_not_rehashed",
    "manifest_origin_not_independently_authenticated",
)

MALEX_SOURCE_FINGERPRINTS = {'label_archive_sha256': '0ef4285cc14d1edbf6f937f783bc9a3b896bff70e762907cf556d41413f9f0ff', 'benign_identity_sequence_sha256': '30b7e6d4cead5263bb983356f6be77348b2ac48a2878b934ebfe68623b0f8424', 'malware_identity_sequence_sha256': 'e20eff6ceaaaa3d87bcf2b185fa6ddd80615570e3677d887199dcf08754059e9', 'train_index_sha256': '8c31521b7c2151d2c14bd3a9a3025ca5881bf241e27aaadc998279a9def3b32e', 'train_membership_sha256': '4c051c7e5ac388ab12e017c1f34eb1b93e650cded71de08abcad923bda7ebcf2', 'validation_index_sha256': 'e0b6d6d63060f2b6bb54f244aa11ba3b5fd4e8455e2536a49e355bff1841d9ac', 'validation_membership_sha256': '5fcc103c2657b811b1a85b30d40370db3a6459da9cb4f36bb9e9dec24867b91b', 'test_index_sha256': 'c60e118991f4db59a30dd312c9c906a3e9fc8234b0a3fd743009b65097c4d8a7', 'test_membership_sha256': 'f974c0c533ec23ef3e71b8037d22329bc5fcbc63e69d925662e8436b2429663c'}

MALEX_PARTITION_POLICY = {'policy_id': 'malex_governed_revision_2026', 'source_benign_count': 179725, 'source_malware_count': 864669, 'cross_class_conflicting_identities': 19, 'post_exclusion_benign_count': 179706, 'post_exclusion_malware_count': 864650, 'train_size': 731049, 'validation_size': 104435, 'test_size': 208872}

MALEX_LIMITATIONS = ('does_not_authenticate_checkpoint_origin', 'does_not_prove_continuous_source_immutability', 'does_not_prove_training_used_only_verified_bytes', 'does_not_independently_rehash_image_contents', 'record_validation_does_not_repeat_adapter_checks')

USTC_PARTITION_POLICY = {
    "policy_id": "ustc_case2_revision_2026",
    "partition_seed": 42,
    "validation_fraction": 0.1,
    "train_size": 72900,
    "validation_size": 8100,
    "test_size": 9000,
}

USTC_LIMITATIONS = (
    "does_not_authenticate_checkpoint_origin",
    "does_not_prove_continuous_source_immutability",
    "does_not_prove_training_used_only_verified_bytes",
)


def validate_gan_v3_provenance_evidence(
    evidence, *, expected_dataset_id
):
    """Validate the exact scientific provenance evidence contract.

    Successful validation proves only that the record conforms to
    its declared dataset contract. It does not rehash source files.
    """
    if not isinstance(evidence, Mapping):
        raise TypeError("provenance evidence must be a mapping")

    if set(evidence) != REQUIRED_FIELDS:
        raise ValueError("provenance evidence fields mismatch")

    result = deepcopy(dict(evidence))

    if result["schema_type"] != SCHEMA_TYPE:
        raise ValueError("unsupported provenance schema type")

    if result["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported provenance schema version")

    if (
        type(expected_dataset_id) is not str
        or result["dataset_id"] != expected_dataset_id
    ):
        raise ValueError("provenance dataset identity mismatch")

    if result["dataset_id"] == "cicmaldroid2020":
        if result["source_contract"] != (
            "cic_authoritative_source_revision_2026"
        ):
            raise ValueError("CIC source contract mismatch")

        fingerprints = result["source_fingerprints"]

        if not isinstance(fingerprints, dict):
            raise TypeError("CIC fingerprints must be a dictionary")

        if set(fingerprints) != set(CIC_SOURCE_FINGERPRINTS):
            raise ValueError("CIC fingerprint inventory mismatch")

        for name, expected in CIC_SOURCE_FINGERPRINTS.items():
            observed = fingerprints[name]
            if (
                type(observed) is not str
                or SHA256_PATTERN.fullmatch(observed) is None
                or observed != expected
            ):
                raise ValueError(
                    f"CIC source fingerprint mismatch: {name}"
                )

        partition = result["partition_policy"]

        if not isinstance(partition, dict):
            raise TypeError("CIC partition policy must be a dictionary")

        if set(partition) != set(CIC_PARTITION_POLICY):
            raise ValueError("CIC partition policy fields mismatch")

        for name, expected in CIC_PARTITION_POLICY.items():
            actual = partition[name]
            if (
                type(actual) is not type(expected)
                or actual != expected
            ):
                raise ValueError(
                    f"CIC partition policy mismatch: {name}"
                )

        if result["verification_method"] != (
            "sha256_full_file_stream_on_compute_node"
        ):
            raise ValueError("CIC verification method mismatch")

        if result["verification_scope"] != (
            "authoritative_csv_and_feature_columns_json"
        ):
            raise ValueError("CIC verification scope mismatch")

        if (
            type(result["verification_limitations"]) is not list
            or result["verification_limitations"]
            != list(CIC_LIMITATIONS)
        ):
            raise ValueError("CIC verification limitations mismatch")

        return result

    if result["dataset_id"] == "malnet_image":
        if result["source_contract"] != (
            "malnet_preparation_integrity_baseline"
        ):
            raise ValueError("MalNet source contract mismatch")

        fingerprints = result["source_fingerprints"]

        if type(fingerprints) is not dict:
            raise TypeError("MalNet fingerprints must be a dictionary")

        if set(fingerprints) != set(MALNET_SOURCE_FINGERPRINTS):
            raise ValueError("MalNet fingerprint inventory mismatch")

        for name, expected in MALNET_SOURCE_FINGERPRINTS.items():
            actual = fingerprints[name]
            if (
                type(actual) is not str
                or SHA256_PATTERN.fullmatch(actual) is None
                or actual != expected
            ):
                raise ValueError(
                    f"MalNet fingerprint mismatch: {name}"
                )

        partition = result["partition_policy"]

        if type(partition) is not dict:
            raise TypeError("MalNet partition must be a dictionary")

        if set(partition) != set(MALNET_PARTITION_POLICY):
            raise ValueError("MalNet partition fields mismatch")

        for name, expected in MALNET_PARTITION_POLICY.items():
            actual = partition[name]
            if type(actual) is not type(expected) or actual != expected:
                raise ValueError(
                    f"MalNet partition mismatch: {name}"
                )

        if result["verification_method"] != (
            "sha256_full_prepared_artifact_stream_on_compute_node"
        ):
            raise ValueError("MalNet verification method mismatch")

        if result["verification_scope"] != (
            "baseline_manifest_and_six_prepared_artifacts"
        ):
            raise ValueError("MalNet verification scope mismatch")

        if (
            type(result["verification_limitations"]) is not list
            or result["verification_limitations"]
            != list(MALNET_LIMITATIONS)
        ):
            raise ValueError("MalNet limitations mismatch")

        return result

    if result["dataset_id"] == "malex":
        if result["source_contract"] != (
            "governed_adapter_constructor"
        ):
            raise ValueError("MaleX source contract mismatch")

        fingerprints = result["source_fingerprints"]

        if type(fingerprints) is not dict:
            raise TypeError("MaleX fingerprints must be a dictionary")

        if set(fingerprints) != set(MALEX_SOURCE_FINGERPRINTS):
            raise ValueError("MaleX fingerprint inventory mismatch")

        for name, expected in MALEX_SOURCE_FINGERPRINTS.items():
            actual = fingerprints[name]
            if (
                type(actual) is not str
                or SHA256_PATTERN.fullmatch(actual) is None
                or actual != expected
            ):
                raise ValueError(
                    f"MaleX fingerprint mismatch: {name}"
                )

        partition = result["partition_policy"]

        if type(partition) is not dict:
            raise TypeError("MaleX partition must be a dictionary")

        if set(partition) != set(MALEX_PARTITION_POLICY):
            raise ValueError("MaleX partition fields mismatch")

        for name, expected in MALEX_PARTITION_POLICY.items():
            actual = partition[name]
            if type(actual) is not type(expected) or actual != expected:
                raise ValueError(
                    f"MaleX partition mismatch: {name}"
                )

        if result["verification_method"] != (
            "production_adapter_integrity_checks_on_compute_node"
        ):
            raise ValueError("MaleX verification method mismatch")

        if result["verification_scope"] != (
            "label_archive_identity_sequences_and_partition_fingerprints"
        ):
            raise ValueError("MaleX verification scope mismatch")

        if (
            type(result["verification_limitations"]) is not list
            or result["verification_limitations"]
            != list(MALEX_LIMITATIONS)
        ):
            raise ValueError("MaleX limitations mismatch")

        return result

    if result["dataset_id"] != "ustc_tfc2016":
        raise ValueError(
            "v3 provenance contract not yet implemented "
            "for this dataset"
        )

    if result["source_contract"] != "ustc_nchw_revision_2026":
        raise ValueError("USTC source contract mismatch")

    fingerprints = result["source_fingerprints"]

    if not isinstance(fingerprints, dict):
        raise TypeError("source fingerprints must be a dictionary")

    if set(fingerprints) != set(USTC_NCHW_SOURCE_SHA256):
        raise ValueError("USTC source fingerprint inventory mismatch")

    for name, expected_digest in USTC_NCHW_SOURCE_SHA256.items():
        digest = fingerprints[name]

        if (
            type(digest) is not str
            or SHA256_PATTERN.fullmatch(digest) is None
            or digest != expected_digest
        ):
            raise ValueError(
                f"USTC source fingerprint mismatch: {name}"
            )

    partition = result["partition_policy"]

    if not isinstance(partition, dict):
        raise TypeError("partition policy must be a dictionary")

    if set(partition) != set(USTC_PARTITION_POLICY):
        raise ValueError("USTC partition policy fields mismatch")

    for name, expected in USTC_PARTITION_POLICY.items():
        actual = partition[name]

        if (
            type(actual) is not type(expected)
            or actual != expected
        ):
            raise ValueError(
                f"USTC partition policy mismatch: {name}"
            )

    if (
        result["verification_method"]
        != "sha256_full_file_stream_on_compute_node"
    ):
        raise ValueError("unsupported verification method")

    if (
        result["verification_scope"]
        != "four_original_nchw_numpy_files"
    ):
        raise ValueError("unsupported verification scope")

    limitations = result["verification_limitations"]

    if (
        type(limitations) is not list
        or limitations != list(USTC_LIMITATIONS)
    ):
        raise ValueError("verification limitations mismatch")

    return result
