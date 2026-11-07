"""Batch validation and lazy-access checks without TensorFlow or real data."""

import numpy as np
import pytest

from trustforge.evaluation.paper01_inputs import RealSplitView


class Adapter:
    def __init__(self, size=257, fault=None):
        self.size = size
        self.fault = fault
        self.requests = []

    def split_size(self, split):
        return self.size

    def canonical_batch(self, split, positions):
        self.requests.append((split, positions.copy()))
        images = np.zeros((len(positions), 40, 40, 3), dtype=np.float32)
        labels = np.eye(5, dtype=np.float32)[positions % 5]
        if self.fault == "nonfinite":
            images[0, 0, 0, 0] = np.nan
        elif self.fault == "range":
            images[0, 0, 0, 0] = 2
        elif self.fault == "soft_labels":
            labels[0] = 0.2
        elif self.fault == "mutate":
            positions[0] += 1
        return images, labels


def view(adapter, split="validation"):
    return RealSplitView(
        adapter=adapter, dataset_id="cicmaldroid2020",
        split=split, provenance_sha256="a" * 64,
    )


def test_complete_ordered_traversal_retains_partial_batch():
    adapter = Adapter()
    batches = list(view(adapter).iter_ordered())
    assert [len(batch[0]) for batch in batches] == [128, 128, 1]
    np.testing.assert_array_equal(
        np.concatenate([batch[0] for batch in batches]), np.arange(257),
    )
    assert all(split == "validation" for split, _ in adapter.requests)


def test_large_view_is_lazy_and_bounds_image_access():
    adapter = Adapter(size=1_000_000_000)
    stream = view(adapter).iter_ordered()
    assert adapter.requests == []
    positions, images, _ = next(stream)
    assert len(adapter.requests) == 1
    assert len(positions) == 128
    assert images.shape == (128, 40, 40, 3)


@pytest.mark.parametrize("fault", ["nonfinite", "range", "soft_labels", "mutate"])
def test_invalid_adapter_output_rejected(fault):
    with pytest.raises((ValueError, RuntimeError)):
        view(Adapter(fault=fault)).batch(np.arange(3, dtype=np.int64))


@pytest.mark.parametrize("positions", [
    np.array([0, 0], dtype=np.int64),
    np.array([-1], dtype=np.int64),
    np.array([257], dtype=np.int64),
    np.arange(129, dtype=np.int64),
    np.arange(3, dtype=np.int32),
])
def test_invalid_positions_rejected_before_adapter_access(positions):
    adapter = Adapter()
    with pytest.raises(ValueError):
        view(adapter).batch(positions)
    assert adapter.requests == []


def test_split_identity_and_changed_size():
    adapter = Adapter()
    selected = view(adapter, split="test")
    assert selected.identity["split"] == "test"
    selected.identity["split"] = "train"
    assert selected.identity["split"] == "test"
    adapter.size = 258
    with pytest.raises(RuntimeError, match="size changed"):
        selected.batch(np.arange(3, dtype=np.int64))
