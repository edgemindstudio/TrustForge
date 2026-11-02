"""Synthetic VAE generation-directory reservation regressions.

Only pytest-managed disposable directories are modified.
No authoritative checkpoint publication or model training.
"""

import os
import stat

import pytest

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_checkpoint_publication_reservation as reservation,
)


GENERATION_ID = "a" * 32


@pytest.fixture
def generations_fd(tmp_path):
    generations = tmp_path / "generations"
    generations.mkdir(mode=0o700)
    generations.chmod(0o700)

    fd = os.open(
        generations,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )

    try:
        yield fd, generations
    finally:
        os.close(fd)


def test_reservation_creates_anchored_directory(generations_fd):
    fd, generations = generations_fd

    child_fd = reservation.reserve_vae_generation_directory(
        fd, GENERATION_ID
    )

    try:
        assert stat.S_ISDIR(os.fstat(child_fd).st_mode)
        assert (generations / GENERATION_ID).is_dir()
    finally:
        os.close(child_fd)


def test_existing_generation_collision_preserves_entry(generations_fd):
    fd, generations = generations_fd
    original = generations / GENERATION_ID
    original.mkdir()
    marker = original / "existing.txt"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError):
        reservation.reserve_vae_generation_directory(
            fd, GENERATION_ID
        )

    assert marker.read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize(
    "generation_id",
    [
        "",
        "ABCDEF0123456789ABCDEF0123456789",
        "../escape",
        "g" * 32,
        "a" * 31,
        None,
    ],
)
def test_invalid_generation_id_rejected(
    generations_fd,
    generation_id,
):
    fd, _ = generations_fd

    with pytest.raises(ValueError, match="invalid VAE checkpoint generation ID"):
        reservation.reserve_vae_generation_directory(
            fd, generation_id
        )


def test_non_directory_parent_descriptor_rejected(tmp_path):
    regular = tmp_path / "regular.bin"
    regular.write_bytes(b"test")

    fd = os.open(regular, os.O_RDONLY)

    try:
        with pytest.raises(ValueError, match="not a directory"):
            reservation.reserve_vae_generation_directory(
                fd, GENERATION_ID
            )
    finally:
        os.close(fd)


def test_fsync_failure_preserves_reserved_directory(
    generations_fd,
    monkeypatch,
):
    fd, generations = generations_fd

    def injected_failure(descriptor):
        raise OSError("injected generation fsync failure")

    with monkeypatch.context() as patch:
        patch.setattr(
            reservation.os,
            "fsync",
            injected_failure,
        )

        with pytest.raises(
            OSError,
            match="injected generation fsync failure",
        ):
            reservation.reserve_vae_generation_directory(
                fd, GENERATION_ID
            )

    assert (generations / GENERATION_ID).is_dir()


def test_existing_symlink_collision_preserved(
    generations_fd,
    tmp_path,
):
    fd, generations = generations_fd

    target = tmp_path / "target"
    target.mkdir()

    link = generations / GENERATION_ID
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(FileExistsError):
        reservation.reserve_vae_generation_directory(
            fd, GENERATION_ID
        )

    assert link.is_symlink()
    assert link.resolve() == target
