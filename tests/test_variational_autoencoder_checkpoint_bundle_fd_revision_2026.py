"""Direct descriptor-native VAE checkpoint verification regressions.

Uses disposable synthetic checkpoint artifacts only.
Does not publish checkpoints or execute real-data training.
"""

import os

import optree
import keras
import pytest
import tensorflow as tf

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_checkpoint_anchored_bundle as reader,
    variational_autoencoder_checkpoint_directory_anchor as anchor,
)

from tests.test_variational_autoencoder_checkpoint_loader_roundtrip_revision_2026 import (
    synthetic_bundle,
    DATASET_ID,
    SEED,
    CLASS_MAPPING,
)


def read_fd(fd, *, seed=SEED):
    return reader.read_anchored_vae_checkpoint_bundle_fd(
        fd,
        requested_dataset_id=DATASET_ID,
        requested_experimental_seed=seed,
        requested_class_mapping=CLASS_MAPPING,
    )


def test_descriptor_native_verification_succeeds(synthetic_bundle):
    with anchor.open_checkpoint_directory(
        synthetic_bundle["directory"]
    ) as fd:
        result = read_fd(fd)

        assert result["metadata"] == synthetic_bundle["metadata"]
        assert result["dataset_provenance"] == (
            synthetic_bundle["provenance"]
        )
        assert set(result["verified_state_bytes"]) == {
            "vae_encoder_state.npz",
            "vae_decoder_state.npz",
        }

        os.fstat(fd)


def test_caller_descriptor_remains_open_after_rejection(
    synthetic_bundle,
):
    with anchor.open_checkpoint_directory(
        synthetic_bundle["directory"]
    ) as fd:
        with pytest.raises(
            ValueError,
            match="experimental seed mismatch",
        ):
            read_fd(fd, seed=SEED + 1)

        os.fstat(fd)


@pytest.mark.parametrize(
    "invalid_fd",
    [None, "3", 1.5, True],
)
def test_invalid_descriptor_type_rejected(invalid_fd):
    with pytest.raises(
        TypeError,
        match="directory FD must be an integer",
    ):
        read_fd(invalid_fd)


def test_closed_descriptor_rejected(synthetic_bundle):
    fd = os.open(
        synthetic_bundle["directory"],
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    os.close(fd)

    with pytest.raises(
        ValueError,
        match="invalid checkpoint directory FD",
    ):
        read_fd(fd)


def test_regular_file_descriptor_rejected(tmp_path):
    path = tmp_path / "ordinary_file"
    path.write_bytes(b"not-a-checkpoint")

    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        with pytest.raises(
            ValueError,
            match="checkpoint FD is not a directory",
        ):
            read_fd(fd)

        os.fstat(fd)
    finally:
        os.close(fd)


def test_directory_replacement_does_not_redirect_fd(
    synthetic_bundle,
):
    original = synthetic_bundle["directory"]

    with anchor.open_checkpoint_directory(original) as fd:
        moved = original.with_name(original.name + "_saved")
        original.rename(moved)

        original.mkdir()
        (original / reader.MANIFEST_FILENAME).write_bytes(
            b"malicious replacement"
        )

        result = read_fd(fd)

        assert result["metadata"] == synthetic_bundle["metadata"]
        assert (
            original / reader.MANIFEST_FILENAME
        ).read_bytes() == b"malicious replacement"


def test_symlinked_manifest_rejected_via_fd(synthetic_bundle):
    root = synthetic_bundle["directory"]
    original = root / reader.MANIFEST_FILENAME
    saved = root / "manifest_original.json"

    original.rename(saved)
    original.symlink_to(saved.name)

    with anchor.open_checkpoint_directory(root) as fd:
        with pytest.raises(ValueError):
            read_fd(fd)

        os.fstat(fd)
