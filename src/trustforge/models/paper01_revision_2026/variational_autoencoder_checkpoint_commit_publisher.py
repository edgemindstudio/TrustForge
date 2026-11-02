"""Candidate exclusive VAE generation commit publisher.

For disposable synthetic integration development only.

Requires cooperative same-UID operation. Does not establish checkpoint
origin authenticity, immutable namespace identity, GPFS crash durability,
or authorization for real-data training or production publication.
"""

from __future__ import annotations

import os
import secrets
import stat

from .variational_autoencoder_checkpoint_commit_record import (
    build_vae_commit_record,
    encode_vae_commit_record,
    validate_vae_generation_id,
)
from .variational_autoencoder_checkpoint_manifest import (
    MANIFEST_FILENAME,
)
from .variational_autoencoder_checkpoint_publication_input import (
    validate_vae_publication_input,
)
from .variational_autoencoder_checkpoint_publication_file import (
    write_exclusive_vae_artifact,
)
from .variational_autoencoder_checkpoint_publication_layout import (
    COMMIT_SUFFIX,
)
from .variational_autoencoder_checkpoint_publication_namespace_policy import (
    validate_private_vae_generation_parent,
)
from .variational_autoencoder_checkpoint_precommit_verification import (
    verify_precommit_vae_generation,
)
from .variational_autoencoder_checkpoint_publication_policy import (
    COMMITTED_SYNCED,
)


class IndeterminateVAECommitError(RuntimeError):
    """Commit-link operation may have changed persistent namespace."""


def _require_bound_generation(
    generations_fd,
    generation_fd,
    generation_id,
):
    """Check descriptor-to-name binding at one observable instant."""
    actual = os.fstat(generation_fd)
    named = os.stat(
        generation_id,
        dir_fd=generations_fd,
        follow_symlinks=False,
    )

    if (
        not stat.S_ISDIR(actual.st_mode)
        or not stat.S_ISDIR(named.st_mode)
        or (actual.st_dev, actual.st_ino)
        != (named.st_dev, named.st_ino)
    ):
        raise ValueError("VAE generation descriptor identity mismatch")


def publish_candidate_vae_commit(
    generations_fd,
    commits_fd,
    generation_fd,
    generation_id,
    *,
    prepared,
    requested_dataset_id,
    requested_experimental_seed,
    requested_class_mapping,
):
    """Publish an exclusive marker after verifying synthetic evidence.

    Caller retains ownership of all supplied descriptors.
    Failed or interrupted staging is never deleted automatically.
    Errors at or after attempted linking have indeterminate outcome
    except a confirmed FileExistsError, which is a duplicate rejection.
    """
    generation_id = validate_vae_generation_id(generation_id)
    validated = validate_vae_publication_input(prepared)

    validate_private_vae_generation_parent(generations_fd)
    validate_private_vae_generation_parent(commits_fd)

    _require_bound_generation(
        generations_fd,
        generation_fd,
        generation_id,
    )

    verify_precommit_vae_generation(
        generation_fd,
        prepared=validated,
        requested_dataset_id=requested_dataset_id,
        requested_experimental_seed=requested_experimental_seed,
        requested_class_mapping=requested_class_mapping,
    )

    record = build_vae_commit_record(
        generation_id=generation_id,
        manifest_bytes=validated["artifacts"][MANIFEST_FILENAME],
    )
    raw = encode_vae_commit_record(record)

    final_name = generation_id + COMMIT_SUFFIX
    staged_name = (
        ".vae-staged-"
        + generation_id
        + "-"
        + secrets.token_hex(16)
        + ".tmp"
    )

    write_exclusive_vae_artifact(
        commits_fd,
        staged_name,
        raw,
    )

    # Persist the staging entry before attempting a visible commit.
    # These calls do not independently establish GPFS crash durability.
    os.fsync(commits_fd)

    link_attempted = False

    try:
        link_attempted = True
        os.link(
            staged_name,
            final_name,
            src_dir_fd=commits_fd,
            dst_dir_fd=commits_fd,
            follow_symlinks=False,
        )

    except FileExistsError:
        # An existing commit name is never overwritten. Keep the
        # staged record for explicit recovery inspection.
        raise

    except BaseException as exc:
        if link_attempted:
            raise IndeterminateVAECommitError(
                "VAE commit link result is indeterminate"
            ) from exc
        raise

    try:
        os.unlink(staged_name, dir_fd=commits_fd)
        os.fsync(commits_fd)

    except BaseException as exc:
        raise IndeterminateVAECommitError(
            "VAE commit visibility or synchronization is indeterminate"
        ) from exc

    return {
        "generation_id": generation_id,
        "commit_filename": final_name,
        "commit_record": record,
        "outcome": COMMITTED_SYNCED,
        "crash_durability_verified": False,
    }


__all__ = [
    "IndeterminateVAECommitError",
    "publish_candidate_vae_commit",
]
