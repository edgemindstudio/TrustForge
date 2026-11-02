"""Authoritative class ordering for revision-2026 GAN runs.

USTC and CIC numeric strings are computational identifiers.
MalNet and MaleX use the governed adapters' ordered label metadata.
"""

from __future__ import annotations

from collections.abc import Mapping

from trustforge.models.paper01_revision_2026.gan_checkpoint_manifest import (
    DATASET_CLASS_COUNTS,
)


def resolve_gan_class_mapping(*, dataset_id, adapter):
    if dataset_id not in DATASET_CLASS_COUNTS:
        raise ValueError("unsupported governed dataset")

    count = DATASET_CLASS_COUNTS[dataset_id]

    if dataset_id in ("ustc_tfc2016", "cicmaldroid2020"):
        return tuple(str(index) for index in range(count))

    if getattr(adapter, "num_classes", None) != count:
        raise ValueError("adapter class count does not match dataset")

    id_to_class = getattr(adapter, "id_to_class", None)
    class_to_id = getattr(adapter, "class_to_id", None)

    if not isinstance(id_to_class, Mapping):
        raise TypeError("adapter lacks authoritative id_to_class mapping")

    if not isinstance(class_to_id, Mapping):
        raise TypeError("adapter lacks authoritative class_to_id mapping")

    if set(id_to_class) != set(range(count)):
        raise ValueError("adapter class IDs are incomplete or unexpected")

    labels = tuple(id_to_class[index] for index in range(count))

    if (
        any(
            not isinstance(label, str)
            or not label
            or label != label.strip()
            for label in labels
        )
        or len(set(labels)) != count
    ):
        raise ValueError("invalid authoritative class labels")

    if class_to_id != {
        label: index for index, label in enumerate(labels)
    }:
        raise ValueError("adapter forward/reverse mappings disagree")

    if dataset_id == "malex" and labels != ("benign", "malware"):
        raise ValueError("MaleX authoritative class ordering changed")

    return labels
