"""AR/MAF input mechanics; no execution or dataset authorization."""
import numpy as np

from .variational_autoencoder_data_integration import canonical_vae_batch
from .variational_autoencoder_traversal import (
    training_position_batches,
    validation_position_batches,
)

DATASET_CLASSES = {
    "ustc_tfc2016": 9,
    "cicmaldroid2020": 5,
    "malnet_image": 47,
    "malex": 2,
}
DATASETS = tuple(DATASET_CLASSES)
FAMILIES = ("ar", "maf")
SEEDS = tuple(range(42, 49))

def require_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    return int(value)

def synthesis_batch_seed(*, dataset_id, family, experimental_seed,
                         class_id, batch_index):
    """Assign a seed for one fixed eight-sample production batch."""
    if dataset_id not in DATASETS or family not in FAMILIES:
        raise ValueError("unsupported dataset or family")
    seed = require_integer(experimental_seed, "experimental_seed")
    cls = require_integer(class_id, "class_id")
    batch = require_integer(batch_index, "batch_index")
    if seed not in SEEDS:
        raise ValueError("experimental seed outside selected recipe")
    if not 0 <= cls < DATASET_CLASSES[dataset_id]:
        raise ValueError("class outside authoritative dataset universe")
    if not 0 <= batch < 250:
        raise ValueError("batch index outside 2000/class, batch-eight plan")
    return 1 + (
        (((SEEDS.index(seed)*4 + DATASETS.index(dataset_id))*2
          + FAMILIES.index(family))*47 + cls)*250 + batch
    )

def canonical_training_batches(*, adapter, dataset_id, experimental_seed,
                               epoch):
    """Yield positions/images/one-hot labels; caller verifies provenance."""
    seed = require_integer(experimental_seed, "experimental_seed")
    epoch = require_integer(epoch, "epoch")
    if dataset_id not in DATASET_CLASSES:
        raise ValueError("unsupported dataset")
    if seed not in SEEDS or epoch not in (0, 1):
        raise ValueError("seed or epoch outside selected recipe")
    for positions in training_position_batches(
        adapter=adapter, batch_size=64, experimental_seed=seed, epoch=epoch
    ):
        images, labels = canonical_vae_batch(
            adapter=adapter, split="train", logical_positions=positions,
            num_classes=DATASET_CLASSES[dataset_id],
        )
        yield positions, images, labels

def canonical_validation_batches(*, adapter, dataset_id):
    """Yield the complete ordered validation split; never accepts test."""
    if dataset_id not in DATASET_CLASSES:
        raise ValueError("unsupported dataset")
    for positions in validation_position_batches(adapter=adapter, batch_size=64):
        images, labels = canonical_vae_batch(
            adapter=adapter, split="validation", logical_positions=positions,
            num_classes=DATASET_CLASSES[dataset_id],
        )
        yield positions, images, labels
