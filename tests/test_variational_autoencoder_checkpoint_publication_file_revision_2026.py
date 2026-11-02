"""Exclusive VAE publication-file writer regressions.

All writes occur inside pytest-managed disposable directories.
No authoritative checkpoints, generation commits, or training.
"""

import os

import pytest

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_checkpoint_publication_file as writer,
)


@pytest.fixture
def directory_fd(tmp_path):
    fd = os.open(
        tmp_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    try:
        yield fd, tmp_path
    finally:
        os.close(fd)


def test_exclusive_write_preserves_exact_bytes(directory_fd):
    fd, directory = directory_fd
    raw = b"synthetic-vae-artifact"

    result = writer.write_exclusive_vae_artifact(
        fd, "artifact.bin", raw
    )

    assert (directory / "artifact.bin").read_bytes() == raw
    assert result["filename"] == "artifact.bin"
    assert result["size_bytes"] == len(raw)


def test_existing_file_collision_preserves_original(directory_fd):
    fd, directory = directory_fd
    original = directory / "artifact.bin"
    original.write_bytes(b"original")

    with pytest.raises(FileExistsError):
        writer.write_exclusive_vae_artifact(
            fd, "artifact.bin", b"replacement"
        )

    assert original.read_bytes() == b"original"


@pytest.mark.parametrize(
    "name",
    [
        "",
        ".",
        "..",
        "../escape",
        "nested/file",
        "bad\x00name",
    ],
)
def test_invalid_basename_rejected(directory_fd, name):
    fd, _ = directory_fd

    with pytest.raises(ValueError, match="invalid publication artifact basename"):
        writer.write_exclusive_vae_artifact(
            fd, name, b"synthetic"
        )


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        None,
        "text",
        bytearray(b"data"),
    ],
)
def test_invalid_artifact_bytes_rejected(directory_fd, raw):
    fd, _ = directory_fd

    with pytest.raises(ValueError, match="nonempty bytes"):
        writer.write_exclusive_vae_artifact(
            fd, "artifact.bin", raw
        )


def test_non_directory_descriptor_rejected(tmp_path):
    path = tmp_path / "regular.bin"
    path.write_bytes(b"test")

    fd = os.open(path, os.O_RDONLY)
    try:
        with pytest.raises(ValueError, match="not a directory"):
            writer.write_exclusive_vae_artifact(
                fd, "artifact.bin", b"synthetic"
            )
    finally:
        os.close(fd)


def test_partial_writes_complete_successfully(directory_fd, monkeypatch):
    fd, directory = directory_fd
    original_write = writer.os.write

    def partial_write(descriptor, data):
        return original_write(descriptor, data[:3])

    monkeypatch.setattr(writer.os, "write", partial_write)

    raw = b"synthetic-partial-write-payload"
    writer.write_exclusive_vae_artifact(
        fd, "partial.bin", raw
    )

    assert (directory / "partial.bin").read_bytes() == raw


def test_fsync_failure_preserves_recovery_evidence(directory_fd, monkeypatch):
    fd, directory = directory_fd

    def failed_fsync(descriptor):
        raise OSError("injected fsync failure")

    monkeypatch.setattr(writer.os, "fsync", failed_fsync)

    with pytest.raises(OSError, match="injected fsync failure"):
        writer.write_exclusive_vae_artifact(
            fd, "failed.bin", b"synthetic-recovery-evidence"
        )

    assert (
        directory / "failed.bin"
    ).read_bytes() == b"synthetic-recovery-evidence"
