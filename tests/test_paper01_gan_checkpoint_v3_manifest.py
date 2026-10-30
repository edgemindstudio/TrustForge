"""Focused tests for the independent GAN checkpoint v3 manifest."""

from copy import deepcopy
import pytest

from trustforge.models.paper01_revision_2026.gan_checkpoint_v3_manifest import (
    REQUIRED_FIELDS,
    validate_gan_checkpoint_v3_manifest,
)


def valid_manifest():
    return {
        "schema_version": "3.0",
        "schema_type": "paper01_revision2026_conditional_gan",
        "paper_revision": "paper01_revision_2026",
        "model_family": "conditional_gan",
        "dataset_id": "malex",
        "experimental_seed": 42,
        "class_mapping": ["benign", "malware"],
        "class_count": 2,
        "generator_iteration": 10,
        "discriminator_updates": 20,
        "planned_generator_iterations": 10,
        "checkpoint_selection": "terminal_governed_iteration",
        "image_shape": [40, 40, 3],
        "latent_dim": 100,
        "learning_rate": 0.0002,
        "beta_1": 0.5,
        "class_sampling_alpha": 0.85,
        "batch_size": 256,
        "state_files": {
            "model_state.ckpt.index": "a" * 64,
            "model_state.ckpt.data-00000-of-00001": "b" * 64,
        },
        "exposure_evidence_file": "gan_exposure_evidence.json",
        "exposure_evidence_sha256": "c" * 64,
        "dataset_provenance_file": "gan_dataset_provenance.json",
        "dataset_provenance_sha256": "d" * 64,
    }


def test_valid_manifest():
    record = valid_manifest()
    assert set(record) == REQUIRED_FIELDS
    assert validate_gan_checkpoint_v3_manifest(record) == record


def test_validation_does_not_mutate_input():
    record = valid_manifest()
    original = deepcopy(record)
    validate_gan_checkpoint_v3_manifest(record)
    assert record == original


@pytest.mark.parametrize("field", sorted(valid_manifest()))
def test_missing_field_rejected(field):
    record = valid_manifest()
    del record[field]
    with pytest.raises(ValueError):
        validate_gan_checkpoint_v3_manifest(record)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "2.0"),
        ("dataset_provenance_file", "../provenance.json"),
        ("dataset_provenance_file", "other.json"),
        ("dataset_provenance_file", True),
        ("dataset_provenance_sha256", "0" * 63),
        ("dataset_provenance_sha256", "G" * 64),
        ("dataset_provenance_sha256", True),
        ("dataset_id", "unsupported"),
        ("class_count", 47),
        ("generator_iteration", 9),
        ("discriminator_updates", 19),
        ("image_shape", [256, 256, 1]),
        ("exposure_evidence_sha256", "invalid"),
    ],
)
def test_invalid_contract_rejected(field, value):
    record = valid_manifest()
    record[field] = value
    with pytest.raises((ValueError, TypeError)):
        validate_gan_checkpoint_v3_manifest(record)


def test_extra_field_rejected():
    record = valid_manifest()
    record["unapproved"] = True
    with pytest.raises(ValueError):
        validate_gan_checkpoint_v3_manifest(record)


def test_v2_manifest_cannot_pass_as_v3():
    record = valid_manifest()
    del record["dataset_provenance_file"]
    del record["dataset_provenance_sha256"]
    record["schema_version"] = "2.0"
    with pytest.raises(ValueError):
        validate_gan_checkpoint_v3_manifest(record)


def test_empty_state_inventory_rejected():
    record = valid_manifest()
    record["state_files"] = {}
    with pytest.raises(ValueError):
        validate_gan_checkpoint_v3_manifest(record)
