"""Read-only verification of a committed VAE checkpoint generation.

Checks canonical commit evidence, namespace identity, manifest binding,
and descriptor-anchored checkpoint integrity.

Does not publish checkpoints, establish immutable filesystem contents,
authenticate the producer, or authorize scientific use.
"""

from __future__ import annotations

import hashlib
import json

from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_commit_record import (
    validate_vae_commit_record,
    validate_vae_generation_id,
    encode_vae_commit_record,
)

from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_publication_layout import (
    COMMIT_SUFFIX,
    validate_publication_layout_identity,
)

from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_publication_anchor import (
    open_vae_publication_namespace,
)

from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_directory_anchor import (
    read_anchored_regular_file,
)

from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_anchored_bundle import (
    read_anchored_vae_checkpoint_bundle_fd,
)

from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_contract import (
    MAX_MANIFEST_BYTES,
)

from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_manifest import (
    MANIFEST_FILENAME,
)


MAX_COMMIT_RECORD_BYTES = 4096


def _read_canonical_commit_record(commits_fd, generation_id):
    """Read and validate one exact commit-record entry."""
    name = generation_id + COMMIT_SUFFIX

    raw = read_anchored_regular_file(
        commits_fd,
        name,
        max_bytes=MAX_COMMIT_RECORD_BYTES,
    )

    try:
        record = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("invalid VAE commit record JSON") from exc

    validated = validate_vae_commit_record(record)

    if encode_vae_commit_record(validated) != raw:
        raise ValueError("noncanonical VAE commit record")

    validate_publication_layout_identity(
        generation_id=generation_id,
        record=validated,
    )

    return validated, raw


def _read_bound_manifest(generation_fd, expected_digest):
    """Read exact anchored manifest bytes and check commit binding."""
    raw = read_anchored_regular_file(
        generation_fd,
        MANIFEST_FILENAME,
        max_bytes=MAX_MANIFEST_BYTES,
    )

    actual_digest = hashlib.sha256(raw).hexdigest()

    if actual_digest != expected_digest:
        raise ValueError(
            "VAE committed generation manifest SHA-256 mismatch"
        )

    return raw


def verify_committed_vae_generation(
    publication_root,
    *,
    generation_id,
    requested_dataset_id,
    requested_experimental_seed,
    requested_class_mapping,
):
    """Verify an existing committed generation without filesystem writes.

    Successful verification establishes the tested publication-evidence
    relationship and checkpoint byte integrity, not scientific approval,
    independent origin authentication, or a filesystem snapshot.
    """
    generation_id = validate_vae_generation_id(generation_id)

    with open_vae_publication_namespace(
        publication_root,
        generation_id=generation_id,
    ) as namespace:
        commits_fd = namespace["commits_fd"]
        generation_fd = namespace["generation_fd"]

        record, first_commit_bytes = _read_canonical_commit_record(
            commits_fd,
            generation_id,
        )

        first_manifest_bytes = _read_bound_manifest(
            generation_fd,
            record["manifest_sha256"],
        )

        bundle = read_anchored_vae_checkpoint_bundle_fd(
            generation_fd,
            requested_dataset_id=requested_dataset_id,
            requested_experimental_seed=requested_experimental_seed,
            requested_class_mapping=requested_class_mapping,
        )

        second_manifest_bytes = _read_bound_manifest(
            generation_fd,
            record["manifest_sha256"],
        )

        second_record, second_commit_bytes = (
            _read_canonical_commit_record(
                commits_fd,
                generation_id,
            )
        )

        if first_manifest_bytes != second_manifest_bytes:
            raise RuntimeError(
                "VAE committed generation manifest changed"
            )

        if (
            first_commit_bytes != second_commit_bytes
            or record != second_record
        ):
            raise RuntimeError(
                "VAE committed generation record changed"
            )

        return {
            "generation_id": generation_id,
            "commit_record": record,
            "metadata": bundle["metadata"],
            "dataset_provenance": bundle["dataset_provenance"],
            "verified_state_bytes": bundle["verified_state_bytes"],
        }


__all__ = [
    "MAX_COMMIT_RECORD_BYTES",
    "verify_committed_vae_generation",
]
