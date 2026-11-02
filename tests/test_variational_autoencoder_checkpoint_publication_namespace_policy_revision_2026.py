"""Synthetic regressions for private VAE publication namespace policy."""

import os
import stat
from types import SimpleNamespace
from unittest import mock

import pytest

from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_publication_namespace_policy import (
    validate_private_vae_generation_parent,
)


def test_private_directory_accepted(tmp_path):
    parent = tmp_path / "generations"
    parent.mkdir(mode=0o700)
    parent.chmod(0o700)

    fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        result = validate_private_vae_generation_parent(fd)
        info = os.fstat(fd)

        assert result["device"] == info.st_dev
        assert result["inode"] == info.st_ino
        assert result["owner_uid"] == os.geteuid()
        assert result["mode"] == 0o700
    finally:
        os.close(fd)


@pytest.mark.parametrize("invalid", [None, True, False, 1.5, "3"])
def test_noninteger_descriptor_rejected(invalid):
    with pytest.raises(TypeError):
        validate_private_vae_generation_parent(invalid)


def test_closed_descriptor_rejected(tmp_path):
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    os.close(fd)

    with pytest.raises(ValueError, match="invalid generation parent FD"):
        validate_private_vae_generation_parent(fd)


def test_regular_file_descriptor_rejected(tmp_path):
    path = tmp_path / "ordinary_file"
    path.write_bytes(b"synthetic")

    fd = os.open(path, os.O_RDONLY)
    try:
        with pytest.raises(ValueError, match="not a directory"):
            validate_private_vae_generation_parent(fd)
    finally:
        os.close(fd)


@pytest.mark.parametrize("mode", [0o750, 0o770, 0o705, 0o777])
def test_nonprivate_directory_rejected(tmp_path, mode):
    parent = tmp_path / "generations"
    parent.mkdir()
    parent.chmod(mode)

    fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(ValueError, match="not private"):
            validate_private_vae_generation_parent(fd)
    finally:
        os.close(fd)


def test_foreign_owner_rejected_with_synthetic_stat(tmp_path):
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        actual = os.fstat(fd)
        simulated = SimpleNamespace(
            st_mode=actual.st_mode,
            st_uid=os.geteuid() + 1,
            st_dev=actual.st_dev,
            st_ino=actual.st_ino,
        )

        with mock.patch(
            "trustforge.models.paper01_revision_2026."
            "variational_autoencoder_checkpoint_publication_namespace_policy."
            "os.fstat",
            return_value=simulated,
        ):
            with pytest.raises(ValueError, match="ownership mismatch"):
                validate_private_vae_generation_parent(fd)
    finally:
        os.close(fd)


def test_private_policy_does_not_mutate_directory(tmp_path):
    parent = tmp_path / "generations"
    parent.mkdir()
    parent.chmod(0o700)

    before = parent.stat()

    fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        validate_private_vae_generation_parent(fd)
    finally:
        os.close(fd)

    after = parent.stat()

    assert before.st_dev == after.st_dev
    assert before.st_ino == after.st_ino
    assert stat.S_IMODE(before.st_mode) == stat.S_IMODE(after.st_mode)
    assert list(parent.iterdir()) == []
