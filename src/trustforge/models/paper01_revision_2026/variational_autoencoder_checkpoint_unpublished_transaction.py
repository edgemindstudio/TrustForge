"""Prepare an unpublished VAE checkpoint generation.

Restricted to synthetic development under the documented cooperative
same-UID threat model. This operation creates no commit marker and does
not authorize real-data training or authoritative publication.

Directory substitution and crash durability remain unresolved.
"""

from __future__ import annotations

import os

from .variational_autoencoder_checkpoint_contract import (
    _validate_requested_identity,
)
from .variational_autoencoder_checkpoint_manifest import (
    MANIFEST_FILENAME,
)
from .variational_autoencoder_checkpoint_preparation import (
    VAE_PROVENANCE_FILENAME,
)
from .variational_autoencoder_checkpoint_publication_input import (
    STATE_FILENAMES,
    validate_vae_publication_input,
)
from .variational_autoencoder_checkpoint_publication_reservation import (
    reserve_vae_generation_directory,
)
from .variational_autoencoder_checkpoint_publication_file import (
    write_exclusive_vae_artifact,
)
from .variational_autoencoder_checkpoint_precommit_verification import (
    verify_precommit_vae_generation,
)


def prepare_unpublished_vae_generation(
    generations_fd,
    generation_id,
    *,
    prepared,
    requested_dataset_id,
    requested_experimental_seed,
    requested_class_mapping,
):
    """Write and verify one generation without publishing a commit.

    Returns a dictionary containing the caller-owned generation FD and
    verified evidence. The caller must close the returned FD.

    On failure, the coordinator closes its FD but never removes a
    possibly created generation or an artifact requiring recovery.
    """
    validated = validate_vae_publication_input(prepared)

    # Reject inconsistent requests before any filesystem mutation.
    _validate_requested_identity(
        validated['metadata'],
        dataset_id=requested_dataset_id,
        experimental_seed=requested_experimental_seed,
        class_mapping=requested_class_mapping,
    )

    generation_fd = None

    try:
        generation_fd = reserve_vae_generation_directory(
            generations_fd,
            generation_id,
        )

        # The manifest is written last, but its presence is NOT
        # evidence of authoritative publication.
        write_order = (
            *sorted(STATE_FILENAMES),
            VAE_PROVENANCE_FILENAME,
            MANIFEST_FILENAME,
        )

        if set(write_order) != set(validated["artifacts"]):
            raise ValueError(
                "unpublished generation artifact inventory mismatch"
            )

        written = {}

        for name in write_order:
            written[name] = write_exclusive_vae_artifact(
                generation_fd,
                name,
                validated["artifacts"][name],
            )

        # Artifact writers synchronize individual files. This sync
        # requests persistence of generation-directory entries.
        # It is not a GPFS crash-durability guarantee.
        os.fsync(generation_fd)

        verified = verify_precommit_vae_generation(
            generation_fd,
            prepared=validated,
            requested_dataset_id=requested_dataset_id,
            requested_experimental_seed=requested_experimental_seed,
            requested_class_mapping=requested_class_mapping,
        )

        result = {
            "generation_id": generation_id,
            "generation_fd": generation_fd,
            "written_artifacts": written,
            "verified": verified,
            "publication_status": "UNPUBLISHED",
        }

        # Explicit ownership transfer only after verification.
        generation_fd = None
        return result

    finally:
        if generation_fd is not None:
            os.close(generation_fd)


__all__ = ["prepare_unpublished_vae_generation"]
