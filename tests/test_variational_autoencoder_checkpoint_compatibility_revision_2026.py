"""Candidate VAE architecture compatibility regression tests.

Synthetic checkpoint metadata only. No dataset reads, model
restoration, checkpoint publication, or training.
"""

from copy import deepcopy

# Initialize Keras dependencies before importing the VAE model.
import optree
import keras
import pytest

from test_variational_autoencoder_checkpoint_manifest_revision_2026 import (
    make_manifest,
)

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_checkpoint_compatibility as compatibility,
    variational_autoencoder_checkpoint_manifest as manifest,
    variational_autoencoder_model as model,
)


DATASETS = (
    "ustc_tfc2016",
    "cicmaldroid2020",
    "malnet_image",
    "malex",
)


def compatible_manifest(dataset_id="ustc_tfc2016"):
    candidate = make_manifest(dataset_id)
    candidate["architecture_version"] = (
        compatibility.CANDIDATE_ARCHITECTURE_VERSION
    )
    return candidate


@pytest.mark.parametrize("dataset_id", DATASETS)
def test_governed_dataset_compatible(dataset_id):
    candidate = compatible_manifest(dataset_id)

    validated = (
        compatibility.validate_vae_checkpoint_architecture_compatibility(
            candidate
        )
    )

    assert validated == candidate
    assert validated is not candidate


def test_original_fixture_is_schema_valid_but_incompatible():
    candidate = make_manifest()

    assert manifest.validate_vae_checkpoint_manifest(
        candidate
    ) == candidate

    with pytest.raises(
        ValueError,
        match="unsupported VAE checkpoint architecture version",
    ):
        compatibility.validate_vae_checkpoint_architecture_compatibility(
            candidate
        )


@pytest.mark.parametrize(
    "version",
    [
        "unknown_architecture",
        "candidate_v1",
        "paper01_revision2026_conditional_vae_arch_v2",
    ],
)
def test_unsupported_architecture_version_rejected(version):
    candidate = compatible_manifest()
    candidate["architecture_version"] = version

    with pytest.raises(ValueError):
        compatibility.validate_vae_checkpoint_architecture_compatibility(
            candidate
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("image_shape", [12, 12, 1]),
        ("latent_dim", 64),
        ("encoder_dense_units", 128),
        ("decoder_feature_channels", 32),
    ],
)
def test_invalid_architecture_constants_rejected(field, value):
    candidate = compatible_manifest()
    candidate[field] = value

    with pytest.raises(ValueError):
        compatibility.validate_vae_checkpoint_architecture_compatibility(
            candidate
        )


def test_dataset_class_count_mismatch_rejected():
    candidate = compatible_manifest("malex")
    candidate["class_count"] = 9

    with pytest.raises(ValueError):
        compatibility.validate_vae_checkpoint_architecture_compatibility(
            candidate
        )


def test_compatibility_result_is_detached():
    candidate = compatible_manifest()
    original = deepcopy(candidate)

    result = (
        compatibility.validate_vae_checkpoint_architecture_compatibility(
            candidate
        )
    )

    result["effective_configuration"]["batch_size"] = 999

    assert candidate == original
    assert candidate["effective_configuration"]["batch_size"] == 8


def test_implementation_constant_drift_rejected(monkeypatch):
    candidate = compatible_manifest()

    monkeypatch.setattr(
        model,
        "LATENT_DIM",
        999,
    )

    with pytest.raises(
        ValueError,
        match="VAE architecture compatibility mismatch",
    ):
        compatibility.validate_vae_checkpoint_architecture_compatibility(
            candidate
        )
