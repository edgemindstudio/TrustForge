"""Synthetic regressions for VAE publication-input validation.

Uses disposable checkpoint fixtures only. No publication or training.
"""

import copy

import optree
import keras
import pytest
import tensorflow as tf

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_checkpoint_publication_input as publication_input,
)

from tests.test_variational_autoencoder_checkpoint_loader_roundtrip_revision_2026 import (
    synthetic_bundle,
)


@pytest.fixture
def prepared(synthetic_bundle):
    directory = synthetic_bundle["directory"]
    metadata = copy.deepcopy(synthetic_bundle["metadata"])
    provenance = copy.deepcopy(synthetic_bundle["provenance"])

    names = (
        set(publication_input.STATE_FILENAMES)
        | {
            "vae_checkpoint_manifest.json",
            "vae_dataset_provenance.json",
        }
    )

    artifacts = {
        name: (directory / name).read_bytes()
        for name in names
    }

    return {
        "metadata": metadata,
        "dataset_provenance": provenance,
        "artifacts": artifacts,
    }


def test_valid_synthetic_prepared_package(prepared):
    result = publication_input.validate_vae_publication_input(
        prepared
    )

    assert result["metadata"] == prepared["metadata"]
    assert result["dataset_provenance"] == prepared["dataset_provenance"]
    assert result["artifacts"] == prepared["artifacts"]


@pytest.mark.parametrize(
    "filename",
    sorted(publication_input.STATE_FILENAMES),
)
def test_corrupted_state_bytes_rejected(prepared, filename):
    raw = bytearray(prepared["artifacts"][filename])
    assert raw

    raw[-1] ^= 1
    prepared["artifacts"][filename] = bytes(raw)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        publication_input.validate_vae_publication_input(
            prepared
        )


def test_noncanonical_manifest_rejected(prepared):
    name = "vae_checkpoint_manifest.json"

    prepared["artifacts"][name] += b" "

    with pytest.raises(
        ValueError,
        match="noncanonical publication manifest",
    ):
        publication_input.validate_vae_publication_input(
            prepared
        )


def test_noncanonical_provenance_rejected(prepared):
    name = "vae_dataset_provenance.json"

    prepared["artifacts"][name] += b" "

    with pytest.raises(
        ValueError,
        match="noncanonical publication provenance",
    ):
        publication_input.validate_vae_publication_input(
            prepared
        )


def test_extra_artifact_rejected(prepared):
    prepared["artifacts"]["unexpected.bin"] = b"unexpected"

    with pytest.raises(
        ValueError,
        match="publication artifact inventory mismatch",
    ):
        publication_input.validate_vae_publication_input(
            prepared
        )


def test_missing_artifact_rejected(prepared):
    del prepared["artifacts"]["vae_encoder_state.npz"]

    with pytest.raises(
        ValueError,
        match="publication artifact inventory mismatch",
    ):
        publication_input.validate_vae_publication_input(
            prepared
        )


def test_invalid_prepared_type_rejected():
    with pytest.raises(TypeError):
        publication_input.validate_vae_publication_input(
            None
        )


def test_extra_prepared_field_rejected(prepared):
    prepared["unexpected"] = True

    with pytest.raises(
        ValueError,
        match="prepared publication field mismatch",
    ):
        publication_input.validate_vae_publication_input(
            prepared
        )


def test_nonbytes_artifact_rejected(prepared):
    prepared["artifacts"]["vae_encoder_state.npz"] = "invalid"

    with pytest.raises(
        ValueError,
        match="publication artifacts must be nonempty bytes",
    ):
        publication_input.validate_vae_publication_input(
            prepared
        )
