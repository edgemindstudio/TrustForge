"""Descriptor-backed synthetic VAE pre-commit verification regressions.

Uses existing synthetic checkpoint fixtures and disposable directories.
No authoritative publication or real-data model training.
"""

import copy
import os

import optree
import keras
import pytest
import tensorflow as tf

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_checkpoint_precommit_verification as adapter,
    variational_autoencoder_checkpoint_directory_anchor as anchor,
    variational_autoencoder_checkpoint_publication_input as publication_input,
)

from tests.test_variational_autoencoder_checkpoint_loader_roundtrip_revision_2026 import (
    synthetic_bundle,
    DATASET_ID,
    SEED,
    CLASS_MAPPING,
)


@pytest.fixture
def prepared(synthetic_bundle):
    directory = synthetic_bundle["directory"]

    names = (
        set(publication_input.STATE_FILENAMES)
        | {
            "vae_checkpoint_manifest.json",
            "vae_dataset_provenance.json",
        }
    )

    package = {
        "metadata": copy.deepcopy(synthetic_bundle["metadata"]),
        "dataset_provenance": copy.deepcopy(
            synthetic_bundle["provenance"]
        ),
        "artifacts": {
            name: (directory / name).read_bytes()
            for name in names
        },
    }

    publication_input.validate_vae_publication_input(package)

    return directory, package


def verify(fd, package, *, seed=SEED, class_mapping=CLASS_MAPPING):
    return adapter.verify_precommit_vae_generation(
        fd,
        prepared=package,
        requested_dataset_id=DATASET_ID,
        requested_experimental_seed=seed,
        requested_class_mapping=class_mapping,
    )


def test_valid_checkpoint_verified_with_real_descriptor(prepared):
    directory, package = prepared

    before = {
        name: (directory / name).read_bytes()
        for name in package["artifacts"]
    }

    with anchor.open_checkpoint_directory(directory) as fd:
        result = verify(fd, package)

        assert result["metadata"] == package["metadata"]
        assert result["dataset_provenance"] == (
            package["dataset_provenance"]
        )

        for name, raw in result["verified_state_bytes"].items():
            assert raw == package["artifacts"][name]

        os.fstat(fd)

    after = {
        name: (directory / name).read_bytes()
        for name in package["artifacts"]
    }

    assert before == after


def test_modified_manifest_rejected(prepared):
    directory, package = prepared

    path = directory / "vae_checkpoint_manifest.json"
    path.write_bytes(path.read_bytes() + b" ")

    with anchor.open_checkpoint_directory(directory) as fd:
        with pytest.raises(ValueError, match="manifest bytes mismatch"):
            verify(fd, package)


def test_modified_provenance_rejected(prepared):
    directory, package = prepared

    path = directory / "vae_dataset_provenance.json"
    path.write_bytes(path.read_bytes() + b" ")

    with anchor.open_checkpoint_directory(directory) as fd:
        with pytest.raises(ValueError):
            verify(fd, package)


@pytest.mark.parametrize(
    "filename",
    ["vae_encoder_state.npz", "vae_decoder_state.npz"],
)
def test_modified_state_bytes_rejected(prepared, filename):
    directory, package = prepared

    path = directory / filename
    path.write_bytes(path.read_bytes() + b"corruption")

    with anchor.open_checkpoint_directory(directory) as fd:
        with pytest.raises(ValueError):
            verify(fd, package)


def test_seed_mismatch_rejected(prepared):
    directory, package = prepared

    with anchor.open_checkpoint_directory(directory) as fd:
        with pytest.raises(ValueError, match="experimental seed mismatch"):
            verify(fd, package, seed=SEED + 1)


def test_class_mapping_mismatch_rejected(prepared):
    directory, package = prepared

    with anchor.open_checkpoint_directory(directory) as fd:
        with pytest.raises(ValueError, match="ordered class mapping mismatch"):
            verify(
                fd,
                package,
                class_mapping=list(reversed(CLASS_MAPPING)),
            )


def test_publication_lock_rejected(prepared):
    directory, package = prepared

    (directory / ".vae_checkpoint_publication.lock").mkdir()

    with anchor.open_checkpoint_directory(directory) as fd:
        with pytest.raises(RuntimeError, match="publication incomplete"):
            verify(fd, package)


def test_symlink_manifest_rejected(prepared):
    directory, package = prepared

    manifest = directory / "vae_checkpoint_manifest.json"
    saved = directory / "saved_manifest.json"

    manifest.rename(saved)
    manifest.symlink_to(saved.name)

    with anchor.open_checkpoint_directory(directory) as fd:
        with pytest.raises(ValueError):
            verify(fd, package)


def test_caller_descriptor_preserved_on_rejection(prepared):
    directory, package = prepared

    with anchor.open_checkpoint_directory(directory) as fd:
        with pytest.raises(ValueError, match="experimental seed mismatch"):
            verify(fd, package, seed=SEED + 1)

        os.fstat(fd)
