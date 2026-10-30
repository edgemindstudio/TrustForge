"""Tests for checkpoint v3 descriptor-based artifact reading."""

import os

import pytest

from trustforge.models.paper01_revision_2026.gan_checkpoint_v3_secure_io import (
    read_regular_file_securely,
)


def test_regular_file_accepted(tmp_path):
    path = tmp_path / "evidence.json"
    path.write_bytes(b"validated-fixture")
    assert read_regular_file_securely(
        path, max_bytes=1024
    ) == b"validated-fixture"


def test_symlink_rejected(tmp_path):
    target = tmp_path / "target"
    target.write_bytes(b"secret")
    link = tmp_path / "link"
    link.symlink_to(target)

    with pytest.raises(ValueError):
        read_regular_file_securely(link, max_bytes=1024)


def test_empty_file_rejected(tmp_path):
    path = tmp_path / "empty"
    path.write_bytes(b"")

    with pytest.raises(ValueError, match="size"):
        read_regular_file_securely(path, max_bytes=1024)


def test_oversized_file_rejected(tmp_path):
    path = tmp_path / "large"
    path.write_bytes(b"x" * 1025)

    with pytest.raises(ValueError, match="size"):
        read_regular_file_securely(path, max_bytes=1024)


def test_missing_file_rejected(tmp_path):
    with pytest.raises(ValueError, match="securely open"):
        read_regular_file_securely(
            tmp_path / "missing", max_bytes=1024
        )


def test_directory_rejected(tmp_path):
    with pytest.raises(ValueError, match="regular file"):
        read_regular_file_securely(tmp_path, max_bytes=1024)


def test_invalid_size_limit_rejected(tmp_path):
    path = tmp_path / "fixture"
    path.write_bytes(b"x")

    for limit in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            read_regular_file_securely(
                path, max_bytes=limit
            )


@pytest.mark.skipif(
    not hasattr(os, "mkfifo"),
    reason="POSIX FIFO required",
)
def test_fifo_rejected_without_blocking(tmp_path):
    path = tmp_path / "fifo"
    os.mkfifo(path)

    with pytest.raises(ValueError):
        read_regular_file_securely(path, max_bytes=1024)
