"""Tests for checkpoint-bound GAN scientific evidence verification."""

import json
from pathlib import Path

import pytest

from trustforge.provenance import canonical_json_bytes, sha256_file
from trustforge.models.paper01_revision_2026.gan_run_governance import (
    plan_gan_run,
)
from trustforge.models.paper01_revision_2026.gan_exposure_evidence import (
    build_gan_exposure_evidence,
)
from trustforge.models.paper01_revision_2026.gan_checkpoint_evidence import (
    verify_gan_checkpoint_evidence,
)


def fixture(tmp_path):
    plan = plan_gan_run(
        dataset_id="malex",
        experimental_seed=42,
        class_mapping=("benign", "malware"),
        train_size=5,
    )
    draws = plan.nominal_real_exposures // 2

    report = {
        "batches": plan.planned_generator_iterations,
        "total_real_draws": plan.nominal_real_exposures,
        "total_unique_positions": 5,
        "per_class": [
            {
                "class_id": i,
                "train_size": size,
                "draws": draws,
                "unique_positions": size,
                "repeated_draws": draws - size,
                "nominal_passes": draws / size,
                "unique_coverage": 1.0,
            }
            for i, size in enumerate((2, 3))
        ],
    }

    evidence, digest = build_gan_exposure_evidence(
        plan=plan,
        exposure_report=report,
    )

    path = tmp_path / "gan_exposure_evidence.json"
    path.write_bytes(canonical_json_bytes(evidence))

    metadata = {
        "schema_version": "2.0",
        "exposure_evidence_file": path.name,
        "exposure_evidence_sha256": sha256_file(path),
        "dataset_id": "malex",
        "experimental_seed": 42,
        "class_mapping": ["benign", "malware"],
        "batch_size": 256,
        "class_sampling_alpha": 0.85,
        "planned_generator_iterations": plan.planned_generator_iterations,
        "generator_iteration": plan.planned_generator_iterations,
        "discriminator_updates": plan.planned_discriminator_updates,
    }

    assert metadata["exposure_evidence_sha256"] == digest
    return path, metadata


def rewrite(path, metadata, evidence):
    path.write_bytes(canonical_json_bytes(evidence))
    metadata["exposure_evidence_sha256"] = sha256_file(path)


def test_valid_evidence_verified(tmp_path):
    _, metadata = fixture(tmp_path)
    result = verify_gan_checkpoint_evidence(tmp_path, metadata)
    assert result["per_class"][0]["class_label"] == "benign"


def test_missing_evidence_rejected(tmp_path):
    path, metadata = fixture(tmp_path)
    path.unlink()
    with pytest.raises(ValueError):
        verify_gan_checkpoint_evidence(tmp_path, metadata)


def test_tampered_evidence_rejected(tmp_path):
    path, metadata = fixture(tmp_path)
    with path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError):
        verify_gan_checkpoint_evidence(tmp_path, metadata)


def test_wrong_checkpoint_identity_rejected(tmp_path):
    _, metadata = fixture(tmp_path)
    metadata["experimental_seed"] = 43
    with pytest.raises(ValueError):
        verify_gan_checkpoint_evidence(tmp_path, metadata)


def test_noncanonical_json_rejected(tmp_path):
    path, metadata = fixture(tmp_path)
    evidence = json.loads(path.read_text())
    path.write_text(json.dumps(evidence, indent=2) + "\n")
    metadata["exposure_evidence_sha256"] = sha256_file(path)
    with pytest.raises(ValueError):
        verify_gan_checkpoint_evidence(tmp_path, metadata)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda x: x.update({"total_real_draws": 1}),
        lambda x: x["per_class"][0].update({"class_label": "malware"}),
        lambda x: x["per_class"][0].update({"repeated_draws": 0}),
        lambda x: x.update({"completed_generator_iterations": 499}),
        lambda x: x.update({"train_size": 6}),
        lambda x: x.update({"schema_version": "3.0"}),
    ],
)
def test_inconsistent_evidence_rejected_with_matching_hash(
    tmp_path, mutation
):
    path, metadata = fixture(tmp_path)
    evidence = json.loads(path.read_text())
    mutation(evidence)
    rewrite(path, metadata, evidence)

    with pytest.raises((TypeError, ValueError)):
        verify_gan_checkpoint_evidence(tmp_path, metadata)


def test_evidence_symlink_rejected(tmp_path):
    path, metadata = fixture(tmp_path)
    target = tmp_path / "indirect.json"
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(ValueError):
        verify_gan_checkpoint_evidence(tmp_path, metadata)


def test_trailing_newline_rejected_even_with_matching_file_hash(tmp_path):
    path, metadata = fixture(tmp_path)

    with path.open("ab") as handle:
        handle.write(bytes([10]))

    metadata["exposure_evidence_sha256"] = sha256_file(path)

    with pytest.raises(ValueError, match="canonical JSON"):
        verify_gan_checkpoint_evidence(tmp_path, metadata)
