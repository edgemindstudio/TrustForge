"""Checkpoint v3 state-file integrity regression tests."""

import hashlib
import os

import pytest

from trustforge.models.paper01_revision_2026.gan_checkpoint_v3_state_integrity import (
    sha256_state_file_securely,
    verify_gan_checkpoint_v3_state_files,
)


@pytest.fixture
def prepared(tmp_path):
    state = {
        "model_state.ckpt.index": b"fixture-index",
        "model_state.ckpt.data-00000-of-00001": b"fixture-state",
    }

    for name, raw in state.items():
        (tmp_path / name).write_bytes(raw)

    inventory = {
        name: hashlib.sha256(raw).hexdigest()
        for name, raw in state.items()
    }

    return tmp_path, inventory


def test_valid_state_files(prepared):
    root, inventory = prepared
    assert verify_gan_checkpoint_v3_state_files(
        root, inventory
    )


def test_exact_hash(prepared):
    root, inventory = prepared
    assert sha256_state_file_securely(
        root / "model_state.ckpt.index"
    ) == inventory["model_state.ckpt.index"]


def test_modified_state_rejected(prepared):
    root, inventory = prepared
    path = root / "model_state.ckpt.index"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="integrity failure"):
        verify_gan_checkpoint_v3_state_files(root, inventory)


def test_missing_state_rejected(prepared):
    root, inventory = prepared
    (root / "model_state.ckpt.index").unlink()
    with pytest.raises(ValueError, match="inventory mismatch"):
        verify_gan_checkpoint_v3_state_files(root, inventory)


def test_extra_state_rejected(prepared):
    root, inventory = prepared
    (root / "model_state.ckpt.extra").write_bytes(b"extra")
    with pytest.raises(ValueError, match="inventory mismatch"):
        verify_gan_checkpoint_v3_state_files(root, inventory)


def test_symlink_state_rejected(prepared):
    root, inventory = prepared
    path = root / "model_state.ckpt.index"
    target = root / "target"
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(ValueError, match="unsafe"):
        verify_gan_checkpoint_v3_state_files(root, inventory)


def test_empty_state_rejected(prepared):
    root, inventory = prepared
    (root / "model_state.ckpt.index").write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        verify_gan_checkpoint_v3_state_files(root, inventory)


def test_invalid_digest_rejected(prepared):
    root, inventory = prepared
    inventory["model_state.ckpt.index"] = "invalid"
    with pytest.raises(ValueError, match="SHA-256"):
        verify_gan_checkpoint_v3_state_files(root, inventory)


def test_invalid_filename_rejected(prepared):
    root, inventory = prepared
    inventory["../outside"] = "a" * 64
    with pytest.raises(ValueError, match="name"):
        verify_gan_checkpoint_v3_state_files(root, inventory)


def test_missing_directory_rejected(prepared):
    root, inventory = prepared
    with pytest.raises(ValueError):
        verify_gan_checkpoint_v3_state_files(
            root / "missing", inventory
        )


@pytest.mark.skipif(
    not hasattr(os, "mkfifo"),
    reason="POSIX FIFO required",
)
def test_fifo_state_rejected_without_blocking(prepared):
    root, inventory = prepared
    path = root / "model_state.ckpt.index"
    path.unlink()
    os.mkfifo(path)
    with pytest.raises(ValueError, match="regular file"):
        verify_gan_checkpoint_v3_state_files(root, inventory)
