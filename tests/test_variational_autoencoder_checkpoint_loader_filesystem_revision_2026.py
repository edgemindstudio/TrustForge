"""Filesystem rejection tests for the governed VAE loader.

Uses disposable synthetic fixtures only.
No real checkpoint loading or training.
"""

import optree
import keras
import pytest
import tensorflow as tf

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_checkpoint_loader as loader,
)

from tests.test_variational_autoencoder_checkpoint_loader_rejection_revision_2026 import (
    checkpoint_bundle,
    invoke,
)


@pytest.mark.parametrize(
    "filename",
    [
        loader.MANIFEST_FILENAME,
        loader.VAE_PROVENANCE_FILENAME,
        "vae_encoder_state.npz",
        "vae_decoder_state.npz",
    ],
)
def test_symlinked_checkpoint_file_rejected(
    checkpoint_bundle,
    filename,
):
    original = checkpoint_bundle / filename
    target = checkpoint_bundle / (filename + ".original")

    original.rename(target)
    original.symlink_to(target.name)

    with pytest.raises((ValueError, RuntimeError, OSError)):
        invoke(checkpoint_bundle)


def test_symlinked_checkpoint_directory_rejected(
    checkpoint_bundle,
    tmp_path,
):
    alias = tmp_path.parent / (
        tmp_path.name + "_symlink_alias"
    )

    assert not alias.exists()
    alias.symlink_to(checkpoint_bundle, target_is_directory=True)

    with pytest.raises(
        ValueError,
        match="invalid VAE checkpoint directory",
    ):
        invoke(alias)


def test_unexpected_state_archive_rejected(
    checkpoint_bundle,
):
    unexpected = (
        checkpoint_bundle / "vae_unexpected_state.npz"
    )

    unexpected.write_bytes(b"synthetic-unexpected-state")

    with pytest.raises(
        ValueError,
        match="state-file inventory mismatch",
    ):
        invoke(checkpoint_bundle)
