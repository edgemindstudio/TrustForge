"""Fail-closed dataset provenance preflight for Paper 1 GAN.

Standard-library only. Does not import TensorFlow or dataset adapters.
Source verification must run on an allocated compute node.
"""

from hashlib import sha256
from pathlib import Path


USTC_NCHW_SOURCE_SHA256 = {
    "train_data.npy":
        "4dc2fd447fc9bc7751fb480a1372b38293b09463d6751b1af1af521d05e880b0",
    "train_labels.npy":
        "1781f9681217dba24967f56a4c17b84e2b95d06e89a683976277a80de46c5416",
    "test_data.npy":
        "dcc1b09ef5e8c82c7d93b5da712e69abb3f74f1b092a9afccba243b141c0e95f",
    "test_labels.npy":
        "9842101d83e8209ddf8d1c0b4e7bfcc6a1be3ff4b5b394fae974b6552a9d7dee",
}


def _sha256_file(path):
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_true(adapter, attribute):
    if getattr(adapter, attribute, None) is not True:
        raise ValueError(
            f"dataset provenance requires {attribute}=True"
        )


def _require_int(adapter, attribute, expected):
    observed = getattr(adapter, attribute, None)
    if type(observed) is not int or observed != expected:
        raise ValueError(
            f"dataset provenance requires {attribute}={expected}"
        )


def _require_float(adapter, attribute, expected):
    observed = getattr(adapter, attribute, None)
    if type(observed) is not float or observed != expected:
        raise ValueError(
            f"dataset provenance requires {attribute}={expected}"
        )


def require_gan_dataset_provenance(*, dataset_id, adapter):
    """Check governed dataset settings and supported source evidence.

    USTC and CIC source files and MalNet prepared artifacts are hashed here.
    MaleX retains its existing adapter-settings policy.
    This is not end-to-end verification of all source-image bytes.
    Run full-file verification only on an allocated compute node.
    """
    if dataset_id == "ustc_tfc2016":
        _require_int(adapter, "partition_seed", 42)
        _require_float(adapter, "validation_fraction", 0.1)

        source = getattr(adapter, "source_dir", None)
        if not isinstance(source, Path) or not source.is_dir():
            raise ValueError(
                "USTC provenance requires an existing source directory"
            )

        for filename, expected in USTC_NCHW_SOURCE_SHA256.items():
            path = source / filename
            if not path.is_file():
                raise FileNotFoundError(
                    f"missing governed USTC source: {filename}"
                )
            if _sha256_file(path) != expected:
                raise ValueError(
                    f"USTC source SHA-256 mismatch: {filename}"
                )

        return {
            "dataset_id": dataset_id,
            "source_contract": "ustc_nchw_revision_2026",
            "verified_sha256": dict(USTC_NCHW_SOURCE_SHA256),
        }

    if dataset_id == "cicmaldroid2020":
        _require_true(adapter, "validate_hashes")
        _require_int(adapter, "partition_seed", 42)
        _require_float(adapter, "train_fraction", 0.7)
        _require_float(adapter, "validation_fraction", 0.1)
        _require_float(adapter, "test_fraction", 0.2)

        expected_sources = (
            (
                "source_csv",
                "3415babdd76a2b77c400a9ec3a8a6057705d5a57a9e91753191ecc2946df7c00",
            ),
            (
                "feature_columns_path",
                "7a6045fe10bfe27c1faee98d729de470401b7cf733ccccc107499f11dea8fc8f",
            ),
        )

        # Validate every source path before reading any source bytes.
        for attribute, _ in expected_sources:
            path = getattr(adapter, attribute, None)

            if not isinstance(path, Path):
                raise ValueError(
                    f"CIC provenance requires Path {attribute}"
                )

            if not path.is_file():
                raise FileNotFoundError(
                    f"CIC provenance source missing: {attribute}"
                )

        verified = {}

        for attribute, expected in expected_sources:
            path = getattr(adapter, attribute, None)

            if not isinstance(path, Path):
                raise ValueError(
                    f"CIC provenance requires Path {attribute}"
                )

            if not path.is_file():
                raise FileNotFoundError(
                    f"CIC provenance source missing: {attribute}"
                )

            observed = _sha256_file(path)

            if observed != expected:
                raise ValueError(
                    f"CIC provenance SHA-256 mismatch: {attribute}"
                )

            verified[attribute] = observed

        return {
            "dataset_id": dataset_id,
            "source_contract": "cic_authoritative_source_revision_2026",
            "integrity_settings_verified": True,
            "verified_sha256": verified,
        }

    elif dataset_id == "malnet_image":
        _require_true(adapter, "validate_hashes")
        _require_true(adapter, "validate_production_contract")

        preparation_dir = getattr(adapter, "preparation_dir", None)
        if not isinstance(preparation_dir, Path):
            raise ValueError(
                "MalNet provenance requires a Path preparation_dir"
            )

        if preparation_dir.name != "type-1.0-prepared":
            raise ValueError(
                "MalNet provenance requires type-1.0-prepared"
            )

        from trustforge.models.paper01_revision_2026 import (
            gan_malnet_preparation_integrity,
        )

        baseline_path = (
            Path(__file__).resolve().parents[4]
            / "studies/paper01_benchmark/audits/revision_2026"
            / "malnet_preparation_integrity_baseline.json"
        )

        evidence = (
            gan_malnet_preparation_integrity
            .verify_malnet_preparation_integrity(
                preparation_dir=preparation_dir,
                baseline_path=baseline_path,
            )
        )

        if (
            not isinstance(evidence, dict)
            or evidence.get("dataset_id") != "malnet_image"
            or evidence.get("baseline_sha256")
            != gan_malnet_preparation_integrity.BASELINE_SHA256
            or evidence.get("manifest_sha256")
            != gan_malnet_preparation_integrity.MANIFEST_SHA256
            or not isinstance(evidence.get("verified_artifacts"), dict)
            or len(evidence["verified_artifacts"]) != 6
            or evidence.get("original_image_bytes_verified") is not False
            or evidence.get("provider_archive_rehashed") is not False
        ):
            raise ValueError("MalNet provenance evidence invalid")

        return {
            "dataset_id": dataset_id,
            "source_contract": "malnet_preparation_integrity_baseline",
            "integrity_settings_verified": True,
            "preparation_integrity": evidence,
        }

    elif dataset_id == "malex":
        _require_true(adapter, "validate_hashes")
        _require_true(adapter, "validate_production_contract")

    else:
        raise ValueError(
            f"unsupported governed dataset: {dataset_id!r}"
        )

    return {
        "dataset_id": dataset_id,
        "source_contract": "governed_adapter_constructor",
        "integrity_settings_verified": True,
    }
