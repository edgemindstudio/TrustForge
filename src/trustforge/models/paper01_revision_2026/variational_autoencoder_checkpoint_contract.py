"""Shared candidate VAE checkpoint-loading contracts.

Defines resource limits and requested experimental identity.
Does not authorize checkpoint use or real-data training.
"""

MAX_MANIFEST_BYTES = 1024 * 1024
MAX_PROVENANCE_BYTES = 1024 * 1024
PUBLICATION_LOCK = ".vae_checkpoint_publication.lock"


def _validate_requested_identity(
    metadata,
    *,
    dataset_id,
    experimental_seed,
    class_mapping,
):
    if type(dataset_id) is not str:
        raise TypeError(
            "requested dataset ID must be a string"
        )

    if type(experimental_seed) is not int:
        raise TypeError(
            "requested experimental seed must be an integer"
        )

    if type(class_mapping) not in (tuple, list):
        raise TypeError(
            "requested class mapping must be a sequence"
        )

    if metadata["dataset_id"] != dataset_id:
        raise ValueError(
            "VAE checkpoint dataset identity mismatch"
        )

    if metadata["experimental_seed"] != experimental_seed:
        raise ValueError(
            "VAE checkpoint experimental seed mismatch"
        )

    if metadata["class_mapping"] != list(class_mapping):
        raise ValueError(
            "VAE checkpoint ordered class mapping mismatch"
        )


__all__ = [
    "MAX_MANIFEST_BYTES",
    "MAX_PROVENANCE_BYTES",
    "PUBLICATION_LOCK",
    "_validate_requested_identity",
]
