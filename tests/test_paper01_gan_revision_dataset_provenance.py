"""Lightweight GAN provenance tests: no TensorFlow imports."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import importlib.util

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/trustforge/models/paper01_revision_2026"
    / "gan_dataset_provenance.py"
)

spec = importlib.util.spec_from_file_location(
    "isolated_gan_dataset_provenance",
    MODULE_PATH,
)
provenance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(provenance)


def validate(dataset_id, adapter):
    return provenance.require_gan_dataset_provenance(
        dataset_id=dataset_id,
        adapter=adapter,
    )


def cic(**changes):
    settings = dict(
        validate_hashes=True,
        partition_seed=42,
        train_fraction=0.7,
        validation_fraction=0.1,
        test_fraction=0.2,
    )
    settings.update(changes)
    return SimpleNamespace(**settings)


def large(**changes):
    settings = dict(
        validate_hashes=True,
        validate_production_contract=True,
    )
    settings.update(changes)
    return SimpleNamespace(**settings)


def test_cic_valid_policy(tmp_path):
    csv = tmp_path / "source.csv"
    features = tmp_path / "features.json"
    csv.write_bytes(b"synthetic-source")
    features.write_bytes(b"synthetic-columns")

    expected = {
        csv: "3415babdd76a2b77c400a9ec3a8a6057705d5a57a9e91753191ecc2946df7c00",
        features: "7a6045fe10bfe27c1faee98d729de470401b7cf733ccccc107499f11dea8fc8f",
    }

    # Mock only the digest values, not the provenance decision.
    # Actual byte hashing is tested separately with synthetic files;
    # authoritative production bytes will be checked on Slurm.
    with patch.object(
        provenance,
        "_sha256_file",
        side_effect=lambda path: expected[path],
    ) as hasher:
        result = validate(
            "cicmaldroid2020",
            cic(
                source_csv=csv,
                feature_columns_path=features,
            ),
        )

    assert result["dataset_id"] == "cicmaldroid2020"
    assert result["integrity_settings_verified"] is True
    assert result["source_contract"] == (
        "cic_authoritative_source_revision_2026"
    )
    assert result["verified_sha256"] == {
        "source_csv": expected[csv],
        "feature_columns_path": expected[features],
    }
    assert hasher.call_count == 2



@pytest.mark.parametrize(
    "changes",
    [
        {"validate_hashes": False},
        {"validate_hashes": 1},
        {"partition_seed": 43},
        {"train_fraction": 0.8},
        {"validation_fraction": 0.15},
        {"test_fraction": 0.15},
    ],
)
def test_cic_rejects_policy_deviation(changes):
    with pytest.raises(ValueError):
        validate("cicmaldroid2020", cic(**changes))


@pytest.mark.parametrize("dataset_id", ["malnet_image", "malex"])
def test_large_dataset_valid_policy(dataset_id, tmp_path, monkeypatch):
    adapter = large()
    if dataset_id == "malnet_image":
        adapter.preparation_dir = tmp_path / "type-1.0-prepared"
        from trustforge.models.paper01_revision_2026 import (
            gan_malnet_preparation_integrity as integrity,
        )
        monkeypatch.setattr(
            integrity,
            "verify_malnet_preparation_integrity",
            lambda **kwargs: {
                "dataset_id": "malnet_image",
                "baseline_sha256": integrity.BASELINE_SHA256,
                "manifest_sha256": integrity.MANIFEST_SHA256,
                "verified_artifacts": {
                    str(i): "verified" for i in range(6)
                },
                "original_image_bytes_verified": False,
                "provider_archive_rehashed": False,
            },
        )
    assert validate(
        dataset_id, adapter
    )["integrity_settings_verified"]


@pytest.mark.parametrize("dataset_id", ["malnet_image", "malex"])
@pytest.mark.parametrize(
    "changes",
    [
        {"validate_hashes": False},
        {"validate_production_contract": False},
        {"validate_hashes": None},
    ],
)
def test_large_dataset_rejects_disabled_checks(dataset_id, changes):
    with pytest.raises(ValueError):
        validate(dataset_id, large(**changes))


def test_ustc_accepts_matching_source_hashes(tmp_path):
    for filename in provenance.USTC_NCHW_SOURCE_SHA256:
        (tmp_path / filename).write_bytes(b"test")

    adapter = SimpleNamespace(
        source_dir=tmp_path,
        partition_seed=42,
        validation_fraction=0.1,
    )

    expected = provenance.USTC_NCHW_SOURCE_SHA256

    with patch.object(
        provenance,
        "_sha256_file",
        side_effect=lambda path: expected[path.name],
    ) as hasher:
        result = validate("ustc_tfc2016", adapter)

    assert hasher.call_count == 4
    assert result["verified_sha256"] == expected


def test_ustc_rejects_modified_source(tmp_path):
    for filename in provenance.USTC_NCHW_SOURCE_SHA256:
        (tmp_path / filename).write_bytes(b"incorrect source")

    adapter = SimpleNamespace(
        source_dir=tmp_path,
        partition_seed=42,
        validation_fraction=0.1,
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate("ustc_tfc2016", adapter)


@pytest.mark.parametrize(
    "changes",
    [
        {"partition_seed": 43},
        {"validation_fraction": 0.2},
    ],
)
def test_ustc_rejects_partition_changes(tmp_path, changes):
    settings = dict(
        source_dir=tmp_path,
        partition_seed=42,
        validation_fraction=0.1,
    )
    settings.update(changes)

    with pytest.raises(ValueError):
        validate("ustc_tfc2016", SimpleNamespace(**settings))


def test_ustc_rejects_missing_source_file(tmp_path):
    adapter = SimpleNamespace(
        source_dir=tmp_path,
        partition_seed=42,
        validation_fraction=0.1,
    )
    with pytest.raises(FileNotFoundError):
        validate("ustc_tfc2016", adapter)


def test_unknown_dataset_rejected():
    with pytest.raises(ValueError):
        validate("unknown_dataset", object())
