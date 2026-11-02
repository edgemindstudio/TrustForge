"""Descriptor-anchored checkpoint access security regressions.

Uses disposable filesystem fixtures only.
No checkpoint loading, real datasets, or training.
"""

import os
from pathlib import Path

import pytest

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_checkpoint_directory_anchor as anchor,
)


def test_anchored_regular_file_read(tmp_path):
    payload = b"synthetic-checkpoint-payload"
    (tmp_path / "state.npz").write_bytes(payload)

    with anchor.open_checkpoint_directory(tmp_path) as fd:
        assert "state.npz" in anchor.anchored_member_names(fd)
        assert anchor.anchored_member_exists(fd, "state.npz")

        actual = anchor.read_anchored_regular_file(
            fd,
            "state.npz",
            max_bytes=1024,
        )

    assert actual == payload


def test_anchored_symlinked_file_rejected(tmp_path):
    (tmp_path / "target.npz").write_bytes(b"synthetic")
    (tmp_path / "state.npz").symlink_to("target.npz")

    with anchor.open_checkpoint_directory(tmp_path) as fd:
        with pytest.raises(ValueError):
            anchor.read_anchored_regular_file(
                fd,
                "state.npz",
                max_bytes=1024,
            )


@pytest.mark.parametrize(
    "name",
    [
        "",
        ".",
        "..",
        "../state.npz",
        "nested/state.npz",
        "/tmp/state.npz",
        "state\x00.npz",
    ],
)
def test_invalid_member_basename_rejected(tmp_path, name):
    with anchor.open_checkpoint_directory(tmp_path) as fd:
        with pytest.raises(ValueError):
            anchor.read_anchored_regular_file(
                fd,
                name,
                max_bytes=1024,
            )


def test_directory_rename_preserves_anchor_identity(tmp_path):
    original = tmp_path / "checkpoint"
    original.mkdir()

    (original / "state.npz").write_bytes(b"original-state")

    with anchor.open_checkpoint_directory(original) as fd:
        moved = tmp_path / "checkpoint-moved"
        original.rename(moved)

        original.mkdir()
        (original / "state.npz").write_bytes(
            b"replacement-state"
        )

        anchored = anchor.read_anchored_regular_file(
            fd,
            "state.npz",
            max_bytes=1024,
        )

    assert anchored == b"original-state"
    assert (
        original / "state.npz"
    ).read_bytes() == b"replacement-state"


def test_anchored_file_size_limit(tmp_path):
    (tmp_path / "state.npz").write_bytes(b"0123456789")

    with anchor.open_checkpoint_directory(tmp_path) as fd:
        with pytest.raises(ValueError, match="size"):
            anchor.read_anchored_regular_file(
                fd,
                "state.npz",
                max_bytes=5,
            )


def test_directory_entry_rejected_as_file(tmp_path):
    (tmp_path / "state.npz").mkdir()

    with anchor.open_checkpoint_directory(tmp_path) as fd:
        with pytest.raises(ValueError, match="not a regular file"):
            anchor.read_anchored_regular_file(
                fd,
                "state.npz",
                max_bytes=1024,
            )


def test_symlinked_checkpoint_directory_rejected(tmp_path):
    real = tmp_path / "real"
    real.mkdir()

    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)

    with pytest.raises(ValueError):
        with anchor.open_checkpoint_directory(alias):
            pytest.fail("symlinked directory was opened")
