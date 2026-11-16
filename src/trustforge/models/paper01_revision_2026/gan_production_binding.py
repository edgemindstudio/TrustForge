"""Governed GAN production checkpoint-path binding.

Resolves one deterministic external checkpoint destination from the governed
dataset, experimental seed, accepted source revision and declared attempt.

This module plans and creates only the destination directory. It does not
construct models, read datasets, authorize training, publish checkpoints or
activate production execution.
"""

from __future__ import annotations

import re
from pathlib import Path

from trustforge.storage import validate_storage_identifier

from .experimental_seed_policy import require_governed_experimental_seed
from .gan_checkpoint_manifest import DATASET_CLASS_COUNTS


FAMILY = "gan"

ARTIFACT_ROOT = (
    Path.home() / "trustforge_revision_2026_artifacts" / "paper01"
)

SOURCE_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
RUN_IDENTITY_PATTERN = re.compile(
    r"^commit([0-9a-f]{7})_attempt([0-9]{2,})$"
)


def build_gan_run_identity(
    *,
    expected_source_commit,
    attempt,
):
    """Return the deterministic attempt identity for one GAN run.

    The identity binds the accepted source revision to one explicit numeric
    attempt. Timestamps and random identifiers are deliberately excluded.
    """
    if (
        not isinstance(expected_source_commit, str)
        or SOURCE_COMMIT_PATTERN.fullmatch(expected_source_commit) is None
    ):
        raise ValueError("expected source commit must be a full lowercase SHA-1")

    if type(attempt) is not int or attempt <= 0:
        raise ValueError("attempt must be a positive integer")

    identity = (
        f"commit{expected_source_commit[:7]}_attempt{attempt:02d}"
    )

    return validate_storage_identifier(
        identity,
        field_name="run_identity",
    )


def resolve_gan_production_checkpoint_directory(
    *,
    dataset_id,
    experimental_seed,
    run_identity,
    expected_source_commit,
):
    """Plan one GAN production checkpoint directory without creating it."""
    if dataset_id not in DATASET_CLASS_COUNTS:
        raise ValueError("unsupported governed dataset")

    seed = require_governed_experimental_seed(experimental_seed)

    if (
        not isinstance(expected_source_commit, str)
        or SOURCE_COMMIT_PATTERN.fullmatch(expected_source_commit) is None
    ):
        raise ValueError("expected source commit must be a full lowercase SHA-1")

    if not isinstance(run_identity, str):
        raise TypeError("run identity must be a string")

    identity = validate_storage_identifier(
        run_identity,
        field_name="run_identity",
    )

    match = RUN_IDENTITY_PATTERN.fullmatch(identity)

    if match is None:
        raise ValueError(
            "run identity must match commit<7 hex>_attempt<2+ digits>"
        )

    if match.group(1) != expected_source_commit[:7]:
        raise ValueError("run identity source commit does not match")

    directory = (
        ARTIFACT_ROOT
        / FAMILY
        / dataset_id
        / f"seed{seed}"
        / "checkpoints"
        / "production"
        / identity
    )

    if ".." in directory.parts:
        raise ValueError("checkpoint path must not contain parent traversal")

    return directory


def _reject_symlinked_governed_path(directory):
    """Reject symlinks anywhere from the artifact root to the destination."""
    current = directory

    while True:
        if current.is_symlink():
            raise ValueError(
                f"checkpoint path contains a symlink: {current}"
            )

        if current == ARTIFACT_ROOT or current.parent == current:
            break

        current = current.parent


def create_gan_production_checkpoint_directory(
    *,
    dataset_id,
    experimental_seed,
    run_identity,
    expected_source_commit,
):
    """Create one fresh, identity-bound checkpoint destination.

    This is a separate explicit action. It must only be called during a
    separately authorized execution attempt, never during import or read-only
    planning.
    """
    directory = resolve_gan_production_checkpoint_directory(
        dataset_id=dataset_id,
        experimental_seed=experimental_seed,
        run_identity=run_identity,
        expected_source_commit=expected_source_commit,
    )

    _reject_symlinked_governed_path(directory)

    if directory.is_symlink() or directory.exists():
        raise FileExistsError(
            f"checkpoint destination already exists: {directory}"
        )

    directory.mkdir(parents=True, exist_ok=False)

    if directory.is_symlink() or not directory.is_dir():
        raise RuntimeError("checkpoint destination is not a real directory")

    _reject_symlinked_governed_path(directory)

    return directory
