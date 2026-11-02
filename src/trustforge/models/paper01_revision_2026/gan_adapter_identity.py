"""Dataset-to-adapter identity binding for revision-2026 GAN.

This checks the actual governed adapter type. It does not replace
source-file provenance or the adapter's own data-contract checks.
"""

from __future__ import annotations

from trustforge.data.paper01_revision_2026 import (
    CICMalDroidRevision2026,
    MaleXRevision2026,
    MalNetImageRevision2026,
    USTCRevision2026,
)


DATASET_ADAPTER_TYPES = {
    "ustc_tfc2016": USTCRevision2026,
    "cicmaldroid2020": CICMalDroidRevision2026,
    "malnet_image": MalNetImageRevision2026,
    "malex": MaleXRevision2026,
}


def require_gan_adapter_identity(*, dataset_id, adapter):
    """Reject unsupported IDs and mismatched governed adapter types."""
    if not isinstance(dataset_id, str):
        raise TypeError("dataset_id must be a string")

    expected_type = DATASET_ADAPTER_TYPES.get(dataset_id)

    if expected_type is None:
        raise ValueError("unsupported governed dataset ID")

    if type(adapter) is not expected_type:
        raise TypeError(
            "dataset adapter identity mismatch: "
            f"{dataset_id} requires {expected_type.__name__}, "
            f"received {type(adapter).__name__}"
        )

    return expected_type.__name__
