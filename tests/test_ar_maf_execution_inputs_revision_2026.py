import numpy as np
import pytest

from trustforge.models.paper01_revision_2026.ar_maf_execution_inputs import (
    canonical_training_batches,
    canonical_validation_batches,
    synthesis_batch_seed,
)

class Adapter:
    def __init__(self, defect=None):
        self.defect = defect
        self.requested = []

    def split_size(self, split):
        if split not in ("train", "validation"):
            raise AssertionError("test partition accessed")
        return 130 if split == "train" else 65

    def canonical_batch(self, split, positions):
        self.requested.append(split)
        n = len(positions)
        images = np.zeros((n, 40, 40, 3), dtype=np.float32)
        labels = np.eye(2, dtype=np.float32)[positions % 2]
        if self.defect == "shape":
            images = images[:, :39]
        elif self.defect == "nonfinite":
            images[0, 0, 0, 0] = np.nan
        elif self.defect == "soft_labels":
            labels[0] = [0.5, 0.5]
        elif self.defect == "multi_hot":
            labels[0] = [1, 1]
        elif self.defect == "class_width":
            labels = np.pad(labels, ((0, 0), (0, 1)))
        return images, labels

def training(adapter, seed=42, epoch=0):
    return list(canonical_training_batches(
        adapter=adapter, dataset_id="malex",
        experimental_seed=seed, epoch=epoch,
    ))

def test_complete_reproducible_training_and_distinct_epoch_identity():
    adapter = Adapter()
    batches = training(adapter)
    positions = np.concatenate([b[0] for b in batches])
    assert [len(b[0]) for b in batches] == [64, 64, 2]
    np.testing.assert_array_equal(np.sort(positions), np.arange(130))
    expected = np.random.default_rng(
        np.random.SeedSequence([42, 0])
    ).permutation(130)
    np.testing.assert_array_equal(positions, expected)
    np.testing.assert_array_equal(
        positions, np.concatenate([b[0] for b in training(Adapter())])
    )
    a = np.concatenate([b[0] for b in training(Adapter(), 42, 1)])
    b = np.concatenate([b[0] for b in training(Adapter(), 43, 0)])
    assert not np.array_equal(a, b)
    assert set(adapter.requested) == {"train"}

def test_complete_ordered_validation_with_partial_batch():
    adapter = Adapter()
    batches = list(canonical_validation_batches(
        adapter=adapter, dataset_id="malex",
    ))
    assert [len(b[0]) for b in batches] == [64, 1]
    np.testing.assert_array_equal(
        np.concatenate([b[0] for b in batches]), np.arange(65)
    )
    assert set(adapter.requested) == {"validation"}

@pytest.mark.parametrize("defect", [
    "shape", "nonfinite", "soft_labels", "multi_hot", "class_width",
])
def test_malformed_batches_rejected(defect):
    with pytest.raises((ValueError, TypeError)):
        training(Adapter(defect))

@pytest.mark.parametrize("seed,epoch", [
    (True, 0), (42.0, 0), (41, 0), (49, 0),
    (42, True), (42, 2), (42, -1),
])
def test_invalid_training_identity_rejected(seed, epoch):
    with pytest.raises((ValueError, TypeError)):
        training(Adapter(), seed, epoch)

@pytest.mark.parametrize("change", [
    {"dataset_id": "unknown"}, {"family": "gan"},
    {"experimental_seed": True}, {"experimental_seed": 49},
    {"class_id": 2}, {"class_id": -1}, {"class_id": 0.0},
    {"batch_index": -1}, {"batch_index": 250},
    {"batch_index": True},
])
def test_invalid_synthesis_identity_rejected(change):
    arguments = dict(
        dataset_id="malex", family="ar", experimental_seed=42,
        class_id=0, batch_index=0,
    )
    arguments.update(change)
    with pytest.raises((ValueError, TypeError)):
        synthesis_batch_seed(**arguments)

def test_synthesis_assignment_covers_grid_without_collisions():
    assigned = set()
    ar_max = 0
    for dataset, count in (
        ("ustc_tfc2016", 9), ("cicmaldroid2020", 5),
        ("malnet_image", 47), ("malex", 2),
    ):
        for family in ("ar", "maf"):
            for seed in range(42, 49):
                for cls in range(count):
                    for batch in range(250):
                        value = synthesis_batch_seed(
                            dataset_id=dataset, family=family,
                            experimental_seed=seed, class_id=cls,
                            batch_index=batch,
                        )
                        assert value not in assigned
                        assigned.add(value)
                        if family == "ar":
                            ar_max = max(ar_max, value)
    assert len(assigned) == 220500
    assert max(assigned) == 646750
    # AR pixel offsets occupy disjoint intervals of the first seed word.
    assert max(assigned) < 1000003
    assert ar_max + 1599 * 1000003 < 2**31
