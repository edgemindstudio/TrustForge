"""Synthetic private-parent enforcement regressions."""

import os
from types import SimpleNamespace
from unittest import mock

import pytest

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_checkpoint_publication_reservation as reservation,
)

GENERATION_ID = "c" * 32


@pytest.mark.parametrize("mode", [0o750, 0o705, 0o777])
def test_nonprivate_parent_rejected_before_creation(tmp_path, mode):
    parent = tmp_path / "generations"
    parent.mkdir()
    parent.chmod(mode)

    fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)

    try:
        with pytest.raises(ValueError, match="not private"):
            reservation.reserve_vae_generation_directory(
                fd, GENERATION_ID
            )

        assert not (parent / GENERATION_ID).exists()
    finally:
        os.close(fd)


def test_foreign_owner_rejected_before_creation(tmp_path):
    parent = tmp_path / "generations"
    parent.mkdir()
    parent.chmod(0o700)

    fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)

    actual_fstat = os.fstat

    def simulated_fstat(descriptor):
        info = actual_fstat(descriptor)

        if descriptor == fd:
            return SimpleNamespace(
                st_mode=info.st_mode,
                st_uid=os.geteuid() + 1,
                st_dev=info.st_dev,
                st_ino=info.st_ino,
            )

        return info

    try:
        with mock.patch.object(
            reservation.os,
            "fstat",
            side_effect=simulated_fstat,
        ):
            with pytest.raises(ValueError, match="ownership mismatch"):
                reservation.reserve_vae_generation_directory(
                    fd, GENERATION_ID
                )

        assert not (parent / GENERATION_ID).exists()
    finally:
        os.close(fd)


def test_private_parent_reservation_still_succeeds(tmp_path):
    parent = tmp_path / "generations"
    parent.mkdir()
    parent.chmod(0o700)

    fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)

    try:
        child_fd = reservation.reserve_vae_generation_directory(
            fd, GENERATION_ID
        )

        try:
            assert (parent / GENERATION_ID).is_dir()
        finally:
            os.close(child_fd)
    finally:
        os.close(fd)
