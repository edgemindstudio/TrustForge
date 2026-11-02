"""Descriptor-anchored VAE publication namespace regressions.

Uses disposable directories only. Does not publish checkpoints,
access real datasets, or execute training.
"""

import os

import pytest

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_checkpoint_publication_anchor as anchor,
)


GENERATION_ID = "0123456789abcdef0123456789abcdef"


def create_namespace(root):
    root.mkdir()
    generations = root / "generations"
    commits = root / "commits"
    generations.mkdir()
    commits.mkdir()
    generation = generations / GENERATION_ID
    generation.mkdir()
    return generations, commits, generation


def test_publication_namespace_opens_expected_directories(tmp_path):
    root = tmp_path / "publication"
    generations, commits, generation = create_namespace(root)

    expected = {
        "root_fd": root,
        "generations_fd": generations,
        "commits_fd": commits,
        "generation_fd": generation,
    }

    with anchor.open_vae_publication_namespace(
        root,
        generation_id=GENERATION_ID,
    ) as descriptors:
        assert set(descriptors) == set(expected)

        for name, directory in expected.items():
            assert os.fstat(descriptors[name]).st_ino == (
                directory.stat().st_ino
            )
            assert os.fstat(descriptors[name]).st_dev == (
                directory.stat().st_dev
            )


def test_descriptors_closed_after_success(tmp_path):
    root = tmp_path / "publication"
    create_namespace(root)

    with anchor.open_vae_publication_namespace(
        root,
        generation_id=GENERATION_ID,
    ) as descriptors:
        numbers = tuple(descriptors.values())

    for fd in numbers:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_generation_descriptor_preserves_inode_after_rename(tmp_path):
    root = tmp_path / "publication"
    _, _, generation = create_namespace(root)

    with anchor.open_vae_publication_namespace(
        root,
        generation_id=GENERATION_ID,
    ) as descriptors:
        original = os.fstat(descriptors["generation_fd"])
        renamed = generation.with_name("renamed_generation")
        generation.rename(renamed)

        anchored = os.fstat(descriptors["generation_fd"])

        assert anchored.st_ino == original.st_ino
        assert anchored.st_dev == original.st_dev
        assert renamed.stat().st_ino == original.st_ino


@pytest.mark.parametrize(
    "entry",
    ["generations", "commits"],
)
def test_symlinked_namespace_directory_rejected(tmp_path, entry):
    root = tmp_path / "publication"
    generations, commits, _ = create_namespace(root)

    target = generations if entry == "generations" else commits
    replacement = root / (entry + "_actual")
    target.rename(replacement)
    target.symlink_to(replacement, target_is_directory=True)

    with pytest.raises(
        ValueError,
        match="cannot securely open publication directory",
    ):
        with anchor.open_vae_publication_namespace(
            root,
            generation_id=GENERATION_ID,
        ):
            pass


def test_symlinked_generation_rejected(tmp_path):
    root = tmp_path / "publication"
    generations, _, generation = create_namespace(root)

    actual = generations / "actual_generation"
    generation.rename(actual)
    generation.symlink_to(actual, target_is_directory=True)

    with pytest.raises(
        ValueError,
        match="cannot securely open publication directory",
    ):
        with anchor.open_vae_publication_namespace(
            root,
            generation_id=GENERATION_ID,
        ):
            pass


def test_symlinked_root_rejected(tmp_path):
    actual = tmp_path / "actual_root"
    create_namespace(actual)

    alias = tmp_path / "root_alias"
    alias.symlink_to(actual, target_is_directory=True)

    with pytest.raises(
        ValueError,
        match="invalid VAE checkpoint directory",
    ):
        with anchor.open_vae_publication_namespace(
            alias,
            generation_id=GENERATION_ID,
        ):
            pass


@pytest.mark.parametrize(
    "entry",
    ["generations", "commits"],
)
def test_missing_namespace_directory_rejected(tmp_path, entry):
    root = tmp_path / "publication"
    generations, commits, _ = create_namespace(root)

    target = generations if entry == "generations" else commits
    target.rmdir() if entry == "commits" else None

    if entry == "generations":
        # The directory contains a generation, so remove only
        # the disposable fixture's empty generation first.
        (generations / GENERATION_ID).rmdir()
        generations.rmdir()

    with pytest.raises(
        ValueError,
        match="cannot securely open publication directory",
    ):
        with anchor.open_vae_publication_namespace(
            root,
            generation_id=GENERATION_ID,
        ):
            pass


def test_missing_generation_rejected(tmp_path):
    root = tmp_path / "publication"
    _, _, generation = create_namespace(root)
    generation.rmdir()

    with pytest.raises(
        ValueError,
        match="cannot securely open publication directory",
    ):
        with anchor.open_vae_publication_namespace(
            root,
            generation_id=GENERATION_ID,
        ):
            pass


def test_invalid_generation_id_rejected_before_open(tmp_path):
    with pytest.raises(ValueError, match="generation ID"):
        with anchor.open_vae_publication_namespace(
            tmp_path / "nonexistent",
            generation_id="../escape",
        ):
            pass


def test_descriptors_closed_after_exception(tmp_path):
    root = tmp_path / "publication"
    create_namespace(root)

    numbers = ()

    with pytest.raises(RuntimeError, match="synthetic interruption"):
        with anchor.open_vae_publication_namespace(
            root,
            generation_id=GENERATION_ID,
        ) as descriptors:
            numbers = tuple(descriptors.values())
            raise RuntimeError("synthetic interruption")

    assert len(numbers) == 4

    for fd in numbers:
        with pytest.raises(OSError):
            os.fstat(fd)
