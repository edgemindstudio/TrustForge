"""Combined-view checks using synthetic adapters and pool files."""

import numpy as np
import pytest

from test_paper01_cic_pool import pool
from trustforge.evaluation.paper01_inputs import RealSplitView
from trustforge.evaluation.paper01_combined_inputs import CombinedTrainingView


class Adapter:
    def __init__(self):
        self.requests = []

    def split_size(self, split):
        return 257

    def canonical_batch(self, split, positions):
        self.requests.append((split, positions.copy()))
        images = np.ones((len(positions), 40, 40, 3), dtype=np.float32)
        labels = np.eye(5, dtype=np.float32)[positions % 5]
        return images, labels


def real_view(adapter, split="train"):
    return RealSplitView(
        adapter=adapter, dataset_id="cicmaldroid2020", split=split,
        provenance_sha256="a" * 64,
    )


def test_mixed_positions_preserve_order_and_origin(pool):
    _, _, open_pool = pool
    adapter = Adapter()
    combined = CombinedTrainingView(
        real=real_view(adapter), synthetic=open_pool(),
    )
    assert adapter.requests == []
    positions = np.array([257, 256, 2257, 0], dtype=np.int64)
    observed, images, labels = combined.batch(positions)
    np.testing.assert_array_equal(observed, positions)
    np.testing.assert_array_equal(images[:, 0, 0, 0], [0, 1, 0, 1])
    np.testing.assert_array_equal(labels.argmax(axis=1), [0, 1, 1, 0])
    np.testing.assert_array_equal(adapter.requests[0][1], [256, 0])
    assert combined.sample_identity(256)["source"] == "real"
    assert combined.sample_identity(257)["logical_position"] == 0


@pytest.mark.parametrize("split", ["validation", "test"])
def test_nontraining_real_split_rejected(pool, split):
    _, _, open_pool = pool
    with pytest.raises(ValueError, match="training split"):
        CombinedTrainingView(
            real=real_view(Adapter(), split), synthetic=open_pool(),
        )


def test_incomplete_pool_blocks_combined_view(pool):
    root, _, open_pool = pool
    synthetic = open_pool()
    (root / "c4_b249.npy").unlink()
    with pytest.raises(FileNotFoundError):
        CombinedTrainingView(real=real_view(Adapter()), synthetic=synthetic)


def test_ordered_traversal_complete_and_bounded(pool):
    _, _, open_pool = pool
    adapter = Adapter()
    combined = CombinedTrainingView(
        real=real_view(adapter), synthetic=open_pool(),
    )
    expected_start = 0
    for positions, images, labels in combined.iter_ordered():
        assert 1 <= len(positions) <= 128
        np.testing.assert_array_equal(
            positions, np.arange(expected_start, expected_start + len(positions)),
        )
        expected_start += len(positions)
    assert expected_start == combined.size == 10257
    assert all(split == "train" for split, _ in adapter.requests)
    assert all(len(positions) <= 128 for _, positions in adapter.requests)


def test_identity_copy_and_invalid_request(pool):
    _, _, open_pool = pool
    combined = CombinedTrainingView(
        real=real_view(Adapter()), synthetic=open_pool(),
    )
    copied = combined.identity
    copied["real"]["split"] = "test"
    assert combined.identity["real"]["split"] == "train"
    with pytest.raises(ValueError):
        combined.batch(np.array([0, 0], dtype=np.int64))
    with pytest.raises(ValueError):
        combined.batch(np.arange(129, dtype=np.int64))
