"""Pure publication layout for candidate VAE checkpoint generations.

Defines relative publication paths only. Does not create, verify,
authenticate, publish, or authorize checkpoint generations.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_commit_record import (
    validate_vae_generation_id,
)


GENERATIONS_DIRECTORY = "generations"
COMMITS_DIRECTORY = "commits"
COMMIT_SUFFIX = ".commit.json"


def generation_relative_path(generation_id):
    """Return the canonical relative generation-directory path."""
    validated = validate_vae_generation_id(generation_id)

    return PurePosixPath(
        GENERATIONS_DIRECTORY,
        validated,
    )


def commit_relative_path(generation_id):
    """Return the canonical relative commit-record path."""
    validated = validate_vae_generation_id(generation_id)

    return PurePosixPath(
        COMMITS_DIRECTORY,
        validated + COMMIT_SUFFIX,
    )


def validate_publication_layout_identity(
    *,
    generation_id,
    record,
):
    """Require the publication request and commit identity to agree."""
    validated = validate_vae_generation_id(generation_id)

    if type(record) is not dict:
        raise TypeError("commit record must be a dictionary")

    if record.get("generation_id") != validated:
        raise ValueError(
            "VAE publication generation identity mismatch"
        )

    return {
        "generation_relative_path": generation_relative_path(
            validated
        ),
        "commit_relative_path": commit_relative_path(
            validated
        ),
    }


__all__ = [
    "GENERATIONS_DIRECTORY",
    "COMMITS_DIRECTORY",
    "COMMIT_SUFFIX",
    "generation_relative_path",
    "commit_relative_path",
    "validate_publication_layout_identity",
]
