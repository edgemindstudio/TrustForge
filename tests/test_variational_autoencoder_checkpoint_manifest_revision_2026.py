"""Regression tests for the proposed VAE checkpoint-v1 manifest.

Fixtures are synthetic metadata, not approved training configurations.
No dataset files, checkpoint publication, or training are involved.
"""

from copy import deepcopy

import pytest

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_checkpoint_manifest as manifest,
)


DATASETS = {
    "ustc_tfc2016": 9,
    "cicmaldroid2020": 5,
    "malnet_image": 47,
    "malex": 2,
}


def make_manifest(dataset_id="ustc_tfc2016"):
    count = DATASETS[dataset_id]
    return {
        "schema_version": manifest.SCHEMA_VERSION,
        "schema_type": manifest.SCHEMA_TYPE,
        "paper_revision": manifest.PAPER_REVISION,
        "model_family": manifest.MODEL_FAMILY,
        "dataset_id": dataset_id,
        "experimental_seed": 42,
        "class_mapping": [f"class_{i}" for i in range(count)],
        "class_count": count,
        "image_shape": [40, 40, 3],
        "latent_dim": 100,
        "encoder_dense_units": 256,
        "decoder_feature_channels": 64,
        "architecture_version": "candidate_v1",
        "observation_version": "candidate_v1",
        "observation_variance": 1.0,
        "kl_weight": 1.0,
        "kl_annealing": False,
        "completed_epochs": 1,
        "completed_updates": 10,
        "checkpoint_role": "last",
        "selection_policy_id": "candidate_unapproved_v1",
        "validation_criterion": {
            "metric": "validation_objective",
            "direction": "minimize",
            "value": 1.0,
        },
        "validation_seed_policy_id": "candidate_unapproved_v1",
        "effective_configuration": {
            "batch_size": 8,
            "learning_rate": 0.001,
        },
        "state_files": {
            "vae_encoder_state.npz": "a" * 64,
            "vae_decoder_state.npz": "b" * 64,
        },
        "dataset_provenance_file": "vae_dataset_provenance.json",
        "dataset_provenance_sha256": "c" * 64,
    }


@pytest.mark.parametrize("dataset_id", sorted(DATASETS))
def test_valid_governed_dataset_manifest(dataset_id):
    candidate = make_manifest(dataset_id)
    result = manifest.validate_vae_checkpoint_manifest(candidate)
    assert result == candidate
    assert result is not candidate


@pytest.mark.parametrize("role", ["best", "last"])
def test_recognized_checkpoint_roles(role):
    candidate = make_manifest()
    candidate["checkpoint_role"] = role
    assert manifest.validate_vae_checkpoint_manifest(
        candidate
    )["checkpoint_role"] == role


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "999.0"),
        ("schema_type", "unknown"),
        ("paper_revision", "wrong_revision"),
        ("model_family", "gan"),
        ("dataset_id", "unknown"),
        ("experimental_seed", -1),
        ("experimental_seed", True),
        ("experimental_seed", 2**32),
        ("class_count", 5),
        ("image_shape", [12, 12, 1]),
        ("latent_dim", 64),
        ("encoder_dense_units", 128),
        ("decoder_feature_channels", 32),
        ("observation_variance", 2.0),
        ("kl_weight", 0.5),
        ("kl_annealing", True),
        ("completed_epochs", 0),
        ("completed_updates", 0),
        ("checkpoint_role", "latest"),
        ("architecture_version", ""),
        ("observation_version", "invalid value"),
        ("selection_policy_id", ""),
        ("validation_seed_policy_id", ""),
        ("dataset_provenance_file", "wrong.json"),
        ("dataset_provenance_sha256", "invalid"),
    ],
)
def test_invalid_scalar_or_identity_rejected(field, value):
    candidate = make_manifest()
    candidate[field] = value
    with pytest.raises((ValueError, TypeError)):
        manifest.validate_vae_checkpoint_manifest(candidate)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_field",
        "extra_field",
        "duplicate_class",
        "blank_class",
        "missing_encoder",
        "missing_decoder",
        "bad_encoder_digest",
        "bad_decoder_digest",
        "bad_criterion_metric",
        "bad_criterion_direction",
        "nan_criterion",
        "infinite_criterion",
        "extra_criterion_field",
        "empty_configuration",
        "invalid_configuration_key",
    ],
)
def test_invalid_structures_rejected(mutation):
    candidate = make_manifest()

    if mutation == "missing_field":
        del candidate["architecture_version"]
    elif mutation == "extra_field":
        candidate["unexpected"] = True
    elif mutation == "duplicate_class":
        candidate["class_mapping"][1] = candidate["class_mapping"][0]
    elif mutation == "blank_class":
        candidate["class_mapping"][0] = " "
    elif mutation == "missing_encoder":
        del candidate["state_files"]["vae_encoder_state.npz"]
    elif mutation == "missing_decoder":
        del candidate["state_files"]["vae_decoder_state.npz"]
    elif mutation == "bad_encoder_digest":
        candidate["state_files"]["vae_encoder_state.npz"] = "wrong"
    elif mutation == "bad_decoder_digest":
        candidate["state_files"]["vae_decoder_state.npz"] = "wrong"
    elif mutation == "bad_criterion_metric":
        candidate["validation_criterion"]["metric"] = ""
    elif mutation == "bad_criterion_direction":
        candidate["validation_criterion"]["direction"] = "unknown"
    elif mutation == "nan_criterion":
        candidate["validation_criterion"]["value"] = float("nan")
    elif mutation == "infinite_criterion":
        candidate["validation_criterion"]["value"] = float("inf")
    elif mutation == "extra_criterion_field":
        candidate["validation_criterion"]["unapproved"] = True
    elif mutation == "empty_configuration":
        candidate["effective_configuration"] = {}
    elif mutation == "invalid_configuration_key":
        candidate["effective_configuration"]["invalid key"] = 1

    with pytest.raises((ValueError, TypeError)):
        manifest.validate_vae_checkpoint_manifest(candidate)


def test_returned_manifest_is_deeply_detached():
    candidate = make_manifest()
    result = manifest.validate_vae_checkpoint_manifest(candidate)

    result["class_mapping"][0] = "changed"
    result["state_files"].clear()
    result["effective_configuration"].clear()

    assert candidate["class_mapping"][0] == "class_0"
    assert len(candidate["state_files"]) == 2
    assert candidate["effective_configuration"]


@pytest.mark.parametrize(
    "invalid",
    [None, [], 42, "manifest"],
)
def test_nonmapping_manifest_rejected(invalid):
    with pytest.raises(TypeError):
        manifest.validate_vae_checkpoint_manifest(invalid)


def test_nested_configuration_is_retained():
    candidate = make_manifest()
    candidate["effective_configuration"]["nested"] = {
        "candidate_parameter": [1, 2, 3]
    }

    validated = manifest.validate_vae_checkpoint_manifest(candidate)
    assert validated["effective_configuration"]["nested"] == {
        "candidate_parameter": [1, 2, 3]
    }


def test_selection_policy_is_recorded_not_approved():
    candidate = make_manifest()
    candidate["checkpoint_role"] = "best"
    candidate["selection_policy_id"] = "candidate_unapproved_v1"

    validated = manifest.validate_vae_checkpoint_manifest(candidate)

    assert validated["checkpoint_role"] == "best"
    assert validated["selection_policy_id"] == (
        "candidate_unapproved_v1"
    )
