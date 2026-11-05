"""Compute-node AR/MAF adapter and provenance preflight.

Does not authorize training, construct models or publish checkpoints.
The caller must pass this same adapter instance to subsequent execution.
"""

import hashlib
import os
import socket

from .ar_maf_execution_inputs import (
    DATASET_CLASSES, SEEDS, require_integer,
)
from .ar_maf_checkpoint_identity import (
    SELECTED_RECIPE_SHA256, build_checkpoint_identity,
)

EXPECTED_SPLIT_COUNTS = {
    "ustc_tfc2016": (72900, 8100),
    "cicmaldroid2020": (8117, 1158),
    "malnet_image": (883416, 126202),
    "malex": (731049, 104435),
}


def adapter_class_mapping(*, dataset_id, adapter):
    """Return labels in model-ID order after adapter-type verification."""
    if dataset_id not in DATASET_CLASSES:
        raise ValueError("unsupported dataset")
    count = DATASET_CLASSES[dataset_id]
    if dataset_id in ("ustc_tfc2016", "cicmaldroid2020"):
        # Explicit numeric model IDs; no inferred semantic class names.
        return tuple(str(i) for i in range(count))

    observed_count = require_integer(adapter.num_classes, "num_classes")
    mapping = adapter.class_to_id
    if observed_count != count or type(mapping) is not dict:
        raise ValueError("adapter class inventory mismatch")
    if len(mapping) != count:
        raise ValueError("adapter mapping count mismatch")

    ordered = [None] * count
    for label, raw_id in mapping.items():
        if type(label) is not str or not label.strip():
            raise ValueError("invalid adapter class label")
        class_id = require_integer(raw_id, "class_id")
        if not 0 <= class_id < count or ordered[class_id] is not None:
            raise ValueError("adapter class IDs must be unique and contiguous")
        ordered[class_id] = label
    if any(label is None for label in ordered):
        raise ValueError("incomplete adapter mapping")
    if dataset_id == "malex" and ordered != ["benign", "malware"]:
        raise ValueError("MaleX semantic mapping mismatch")
    return tuple(ordered)


def preflight_dataset(
    *, family, dataset_id, experimental_seed, adapter, recipe_raw,
):
    """Verify the selected adapter and construct its declared checkpoint identity."""
    if family not in ("ar", "maf") or dataset_id not in DATASET_CLASSES:
        raise ValueError("unsupported family or dataset")
    seed = require_integer(experimental_seed, "experimental_seed")
    if seed not in SEEDS:
        raise ValueError("unsupported experimental seed")
    if (
        type(recipe_raw) is not bytes
        or hashlib.sha256(recipe_raw).hexdigest() != SELECTED_RECIPE_SHA256
    ):
        raise ValueError("selected recipe identity mismatch")
    if (
        not os.environ.get("SLURM_JOB_ID")
        or not os.environ.get("SLURM_JOB_NODELIST")
        or socket.gethostname().split(".")[0].startswith("login")
    ):
        raise RuntimeError("allocated compute-node preflight required")

    # Lazy imports avoid importing dataset/TensorFlow modules on module import.
    from .gan_adapter_identity import require_gan_adapter_identity
    from .gan_dataset_provenance import require_gan_dataset_provenance
    from .gan_v3_provenance_adapter import build_gan_v3_provenance

    adapter_type = require_gan_adapter_identity(
        dataset_id=dataset_id, adapter=adapter,
    )
    mapping = adapter_class_mapping(dataset_id=dataset_id, adapter=adapter)
    counts = tuple(
        require_integer(adapter.split_size(split), split + "_size")
        for split in ("train", "validation")
    )
    if counts != EXPECTED_SPLIT_COUNTS[dataset_id]:
        raise ValueError("governed train/validation counts mismatch")

    verified = require_gan_dataset_provenance(
        dataset_id=dataset_id, adapter=adapter,
    )
    provenance = build_gan_v3_provenance(
        dataset_id=dataset_id, verified_preflight=verified,
    )
    identity = build_checkpoint_identity(
        family=family, dataset_id=dataset_id, experimental_seed=seed,
        class_mapping=mapping, recipe_raw=recipe_raw,
        dataset_provenance=provenance,
    )
    return {
        "identity": identity,
        "adapter_type": adapter_type,
        "training_examples": counts[0],
        "validation_examples": counts[1],
    }
