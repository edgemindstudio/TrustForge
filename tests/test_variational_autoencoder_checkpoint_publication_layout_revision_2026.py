"""Pure VAE checkpoint publication-layout regressions.

Tests namespace derivation and identity binding without filesystem
publication, training, or real dataset access.
"""

import pytest

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_checkpoint_publication_layout as layout,
    variational_autoencoder_checkpoint_commit_record as commit,
)


GENERATION_ID = "0123456789abcdef0123456789abcdef"
OTHER_ID = "fedcba9876543210fedcba9876543210"


def valid_record(generation_id=GENERATION_ID):
    return commit.build_vae_commit_record(
        generation_id=generation_id,
        manifest_bytes=b'{"synthetic":"manifest"}\n',
    )


def test_generation_path_is_canonical():
    result = layout.generation_relative_path(GENERATION_ID)

    assert result.as_posix() == (
        "generations/" + GENERATION_ID
    )
    assert not result.is_absolute()
    assert ".." not in result.parts


def test_commit_path_is_canonical():
    result = layout.commit_relative_path(GENERATION_ID)

    assert result.as_posix() == (
        "commits/" + GENERATION_ID + ".commit.json"
    )
    assert not result.is_absolute()
    assert ".." not in result.parts


def test_matching_layout_identity_accepted():
    result = layout.validate_publication_layout_identity(
        generation_id=GENERATION_ID,
        record=valid_record(),
    )

    assert result["generation_relative_path"] == (
        layout.generation_relative_path(GENERATION_ID)
    )
    assert result["commit_relative_path"] == (
        layout.commit_relative_path(GENERATION_ID)
    )


def test_mismatched_layout_identity_rejected():
    with pytest.raises(
        ValueError,
        match="generation identity mismatch",
    ):
        layout.validate_publication_layout_identity(
            generation_id=GENERATION_ID,
            record=valid_record(OTHER_ID),
        )


@pytest.mark.parametrize(
    "generation_id",
    [
        "../" + "a" * 29,
        "../../etc/passwd",
        "/tmp/escape",
        "a/b",
        "A" * 32,
        "",
        "a" * 31,
        "g" * 32,
    ],
)
def test_invalid_generation_path_rejected(generation_id):
    with pytest.raises(ValueError, match="generation ID"):
        layout.generation_relative_path(generation_id)


@pytest.mark.parametrize(
    "generation_id",
    [
        "../" + "a" * 29,
        "../../etc/passwd",
        "/tmp/escape",
        "a/b",
        "A" * 32,
        "",
        "a" * 31,
        "g" * 32,
    ],
)
def test_invalid_commit_path_rejected(generation_id):
    with pytest.raises(ValueError, match="generation ID"):
        layout.commit_relative_path(generation_id)


def test_non_dictionary_commit_record_rejected():
    with pytest.raises(
        TypeError,
        match="commit record must be a dictionary",
    ):
        layout.validate_publication_layout_identity(
            generation_id=GENERATION_ID,
            record=[],
        )


def test_missing_record_generation_identity_rejected():
    record = valid_record()
    record.pop("generation_id")

    with pytest.raises(
        ValueError,
        match="generation identity mismatch",
    ):
        layout.validate_publication_layout_identity(
            generation_id=GENERATION_ID,
            record=record,
        )
