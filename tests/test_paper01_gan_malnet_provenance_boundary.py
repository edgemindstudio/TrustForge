"""MalNet provenance integration: no real data or training."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from trustforge.models.paper01_revision_2026 import (
    gan_dataset_provenance as provenance,
    gan_malnet_preparation_integrity as integrity,
)


def adapter(tmp_path, **changes):
    settings = {
        "validate_hashes": True,
        "validate_production_contract": True,
        "preparation_dir": tmp_path / "type-1.0-prepared",
    }
    settings.update(changes)
    return SimpleNamespace(**settings)


def good_evidence():
    return {
        "dataset_id": "malnet_image",
        "baseline_sha256": integrity.BASELINE_SHA256,
        "manifest_sha256": integrity.MANIFEST_SHA256,
        "verified_artifacts": {
            f"file-{index}": "verified" for index in range(6)
        },
        "original_image_bytes_verified": False,
        "provider_archive_rehashed": False,
    }


def test_malnet_invokes_verifier_and_returns_evidence(
    tmp_path, monkeypatch
):
    calls = []

    def fake_verify(**kwargs):
        calls.append(kwargs)
        return good_evidence()

    monkeypatch.setattr(
        integrity, "verify_malnet_preparation_integrity", fake_verify
    )

    dataset = adapter(tmp_path)
    result = provenance.require_gan_dataset_provenance(
        dataset_id="malnet_image", adapter=dataset
    )

    assert result["integrity_settings_verified"] is True
    assert result["preparation_integrity"] == good_evidence()
    assert len(calls) == 1
    assert calls[0]["preparation_dir"] == dataset.preparation_dir
    assert calls[0]["baseline_path"].name == (
        "malnet_preparation_integrity_baseline.json"
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"validate_hashes": False},
        {"validate_production_contract": False},
        {"preparation_dir": None},
        {"preparation_dir": "type-1.0-prepared"},
        {"preparation_dir": Path("/tmp/other-preparation")},
    ],
)
def test_invalid_policy_blocks_verifier(tmp_path, monkeypatch, changes):
    calls = []

    def unexpected(**kwargs):
        calls.append(kwargs)
        raise AssertionError("verification should not start")

    monkeypatch.setattr(
        integrity, "verify_malnet_preparation_integrity", unexpected
    )

    with pytest.raises(ValueError):
        provenance.require_gan_dataset_provenance(
            dataset_id="malnet_image",
            adapter=adapter(tmp_path, **changes),
        )

    assert calls == []


def test_verifier_failure_propagates(tmp_path, monkeypatch):
    def fail(**kwargs):
        raise ValueError("prepared artifact hash mismatch")

    monkeypatch.setattr(
        integrity, "verify_malnet_preparation_integrity", fail
    )

    with pytest.raises(ValueError, match="hash mismatch"):
        provenance.require_gan_dataset_provenance(
            dataset_id="malnet_image", adapter=adapter(tmp_path)
        )


@pytest.mark.parametrize(
    "change",
    [
        {"dataset_id": "malex"},
        {"baseline_sha256": "0" * 64},
        {"manifest_sha256": "0" * 64},
        {"verified_artifacts": {}},
        {"original_image_bytes_verified": True},
        {"provider_archive_rehashed": True},
    ],
)
def test_invalid_verifier_evidence_rejected(
    tmp_path, monkeypatch, change
):
    evidence = good_evidence()
    evidence.update(change)

    monkeypatch.setattr(
        integrity,
        "verify_malnet_preparation_integrity",
        lambda **kwargs: evidence,
    )

    with pytest.raises(ValueError, match="evidence invalid"):
        provenance.require_gan_dataset_provenance(
            dataset_id="malnet_image", adapter=adapter(tmp_path)
        )
