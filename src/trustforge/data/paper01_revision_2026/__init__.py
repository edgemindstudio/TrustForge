"""Dataset adapters for the governed Paper 1 revision-2026 benchmark."""

from .cicmaldroid import (
    CICMalDroidRevision2026,
    CICPartitionIndices,
    CICSourceArrays,
    CIC_NUM_CLASSES,
)
from .malex import (
    MALEX_NUM_CLASSES,
    MaleXPartitionIndices,
    MaleXRevision2026,
)
from .malnet_image import (
    MALNET_NUM_CLASSES,
    MalNetImageRevision2026,
    MalNetPreparedSplit,
)
from .malnet_preparation import (
    MALNET_PROVIDER_COUNTS,
    MALNET_PROVIDER_MEMBERS,
    MALNET_SPLIT_ARCHIVE_SHA256,
    MalNetPreparedSplitRecord,
    prepare_malnet_index,
)
from .ustc import (
    USTCRevision2026,
    USTCPartitionIndices,
    USTCSourceArrays,
    USTC_NUM_CLASSES,
)


__all__ = [
    "CICMalDroidRevision2026",
    "CICPartitionIndices",
    "CICSourceArrays",
    "CIC_NUM_CLASSES",
    "MALEX_NUM_CLASSES",
    "MaleXPartitionIndices",
    "MaleXRevision2026",
    "MALNET_NUM_CLASSES",
    "MALNET_PROVIDER_COUNTS",
    "MALNET_PROVIDER_MEMBERS",
    "MALNET_SPLIT_ARCHIVE_SHA256",
    "MalNetImageRevision2026",
    "MalNetPreparedSplit",
    "MalNetPreparedSplitRecord",
    "prepare_malnet_index",
    "USTCRevision2026",
    "USTCPartitionIndices",
    "USTCSourceArrays",
    "USTC_NUM_CLASSES",
]
