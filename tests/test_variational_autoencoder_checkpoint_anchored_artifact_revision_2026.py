"""Descriptor-anchored VAE state-artifact verification tests.

Uses disposable synthetic fixtures only.
No real datasets, training, or checkpoint publication.
"""

import hashlib

import optree
import keras
import pytest
import tensorflow as tf

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_checkpoint_directory_anchor as anchor,
    variational_autoencoder_checkpoint_anchored_artifact as verifier,
)

from tests.test_variational_autoencoder_checkpoint_artifact_revision_2026 import (
    make_manifest,
    ENCODER_NAME,
    DECODER_NAME,
    ENCODER_BYTES,
    DECODER_BYTES,
)


@pytest.fixture
def bundle(tmp_path):
    metadata = make_manifest()

    (tmp_path / ENCODER_NAME).write_bytes(ENCODER_BYTES)
    (tmp_path / DECODER_NAME).write_bytes(DECODER_BYTES)

    return tmp_path, metadata


def verify(root, metadata):
    with anchor.open_checkpoint_directory(root) as fd:
        return verifier.verify_anchored_vae_checkpoint_state_artifacts(
            fd,
            metadata,
        )


def test_valid_state_artifacts_return_exact_bytes(bundle):
    root, metadata = bundle
    verified = verify(root, metadata)

    assert verified == {
        ENCODER_NAME: ENCODER_BYTES,
        DECODER_NAME: DECODER_BYTES,
    }


@pytest.mark.parametrize(
    "filename",
    [ENCODER_NAME, DECODER_NAME],
)
def test_corrupted_state_artifact_rejected(bundle, filename):
    root, metadata = bundle
    path = root / filename

    path.write_bytes(path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify(root, metadata)


@pytest.mark.parametrize(
    "filename",
    [ENCODER_NAME, DECODER_NAME],
)
def test_symlinked_state_artifact_rejected(bundle, filename):
    root, metadata = bundle
    path = root / filename
    target = root / (filename + ".original")

    path.rename(target)
    path.symlink_to(target.name)

    with pytest.raises(ValueError):
        verify(root, metadata)


def test_unexpected_state_file_rejected(bundle):
    root, metadata = bundle
    (root / "vae_extra_state.npz").write_bytes(b"extra")

    with pytest.raises(ValueError, match="inventory mismatch"):
        verify(root, metadata)


def test_publication_lock_rejected(bundle):
    root, metadata = bundle
    (root / verifier.PUBLICATION_LOCK).mkdir()

    with pytest.raises(RuntimeError, match="publication incomplete"):
        verify(root, metadata)


def test_directory_path_replacement_preserves_verified_identity(
    bundle,
):
    root, metadata = bundle

    with anchor.open_checkpoint_directory(root) as fd:
        moved = root.with_name(root.name + "_moved")
        root.rename(moved)

        root.mkdir()
        (root / ENCODER_NAME).write_bytes(b"replacement")
        (root / DECODER_NAME).write_bytes(b"replacement")

        verified = verifier.verify_anchored_vae_checkpoint_state_artifacts(
            fd,
            metadata,
        )

    assert verified[ENCODER_NAME] == ENCODER_BYTES
    assert verified[DECODER_NAME] == DECODER_BYTES


def test_lock_appearing_during_verification_rejected(
    bundle,
    monkeypatch,
):
    root, metadata = bundle
    original_read = verifier.read_anchored_regular_file
    calls = []

    def read_and_inject_lock(directory_fd, name, *, max_bytes):
        raw = original_read(
            directory_fd,
            name,
            max_bytes=max_bytes,
        )

        calls.append(name)

        if len(calls) == 1:
            (root / verifier.PUBLICATION_LOCK).mkdir()

        return raw

    monkeypatch.setattr(
        verifier,
        "read_anchored_regular_file",
        read_and_inject_lock,
    )

    with pytest.raises(RuntimeError, match="publication incomplete"):
        verify(root, metadata)

    assert len(calls) == 2
