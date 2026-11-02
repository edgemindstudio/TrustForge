"""VAE logical-position traversal regression tests."""

import numpy as np
import pytest

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_traversal as traversal,
)


class SyntheticAdapter:
    """Minimal adapter that records requested split sizes."""

    def __init__(self, train_size=23, validation_size=11):
        self.sizes = {
            "train": train_size,
            "validation": validation_size,
        }
        self.requests = []

    def split_size(self, split):
        self.requests.append(split)

        if split not in self.sizes:
            raise AssertionError(
                f"unauthorized split requested: {split}"
            )

        return self.sizes[split]


def flatten(batches):
    return np.concatenate(list(batches))


@pytest.mark.parametrize(
    "size,batch_size",
    [(1, 1), (7, 3), (23, 5), (24, 6), (25, 100)],
)
def test_training_complete_epoch_coverage(size, batch_size):
    adapter = SyntheticAdapter(train_size=size)

    batches = list(
        traversal.training_position_batches(
            adapter=adapter,
            batch_size=batch_size,
            experimental_seed=42,
            epoch=0,
        )
    )

    positions = flatten(batches)

    assert positions.dtype == np.int64
    assert len(positions) == size
    np.testing.assert_array_equal(
        np.sort(positions),
        np.arange(size, dtype=np.int64),
    )

    assert all(0 < len(batch) <= batch_size for batch in batches)
    assert adapter.requests == ["train"]


def test_training_replay_is_deterministic():
    adapter = SyntheticAdapter(train_size=100)

    def collect():
        return flatten(
            traversal.training_position_batches(
                adapter=adapter,
                batch_size=13,
                experimental_seed=42,
                epoch=3,
            )
        )

    np.testing.assert_array_equal(collect(), collect())


def test_training_epoch_changes_order():
    adapter = SyntheticAdapter(train_size=100)

    first = flatten(
        traversal.training_position_batches(
            adapter=adapter,
            batch_size=13,
            experimental_seed=42,
            epoch=0,
        )
    )

    second = flatten(
        traversal.training_position_batches(
            adapter=adapter,
            batch_size=13,
            experimental_seed=42,
            epoch=1,
        )
    )

    assert not np.array_equal(first, second)


def test_training_seed_changes_order():
    adapter = SyntheticAdapter(train_size=100)

    first = flatten(
        traversal.training_position_batches(
            adapter=adapter,
            batch_size=13,
            experimental_seed=42,
            epoch=0,
        )
    )

    second = flatten(
        traversal.training_position_batches(
            adapter=adapter,
            batch_size=13,
            experimental_seed=43,
            epoch=0,
        )
    )

    assert not np.array_equal(first, second)


@pytest.mark.parametrize(
    "size,batch_size",
    [(1, 1), (7, 3), (11, 4), (24, 6)],
)
def test_validation_order_and_coverage(size, batch_size):
    adapter = SyntheticAdapter(validation_size=size)

    batches = list(
        traversal.validation_position_batches(
            adapter=adapter,
            batch_size=batch_size,
        )
    )

    positions = flatten(batches)

    np.testing.assert_array_equal(
        positions,
        np.arange(size, dtype=np.int64),
    )
    assert all(0 < len(batch) <= batch_size for batch in batches)
    assert adapter.requests == ["validation"]


def test_dispatch_rejects_test_split_without_adapter_access():
    adapter = SyntheticAdapter()

    with pytest.raises(ValueError):
        list(
            traversal.position_batches(
                adapter=adapter,
                split="test",
                batch_size=4,
            )
        )

    assert adapter.requests == []


def test_training_dispatch_requires_seed_and_epoch():
    adapter = SyntheticAdapter()

    with pytest.raises(ValueError):
        list(
            traversal.position_batches(
                adapter=adapter,
                split="train",
                batch_size=4,
            )
        )

    assert adapter.requests == []


def test_validation_dispatch_rejects_training_controls():
    adapter = SyntheticAdapter()

    with pytest.raises(ValueError):
        list(
            traversal.position_batches(
                adapter=adapter,
                split="validation",
                batch_size=4,
                experimental_seed=42,
                epoch=0,
            )
        )

    assert adapter.requests == []


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_invalid_batch_size_rejected(batch_size):
    adapter = SyntheticAdapter()

    with pytest.raises((TypeError, ValueError)):
        list(
            traversal.training_position_batches(
                adapter=adapter,
                batch_size=batch_size,
                experimental_seed=42,
                epoch=0,
            )
        )


@pytest.mark.parametrize("seed", [-1, True, 1.5])
def test_invalid_seed_rejected(seed):
    adapter = SyntheticAdapter()

    with pytest.raises((TypeError, ValueError)):
        list(
            traversal.training_position_batches(
                adapter=adapter,
                batch_size=4,
                experimental_seed=seed,
                epoch=0,
            )
        )


@pytest.mark.parametrize("epoch", [-1, True, 1.5])
def test_invalid_epoch_rejected(epoch):
    adapter = SyntheticAdapter()

    with pytest.raises((TypeError, ValueError)):
        list(
            traversal.training_position_batches(
                adapter=adapter,
                batch_size=4,
                experimental_seed=42,
                epoch=epoch,
            )
        )


def test_invalid_adapter_rejected():
    with pytest.raises(TypeError):
        list(
            traversal.validation_position_batches(
                adapter=object(),
                batch_size=4,
            )
        )


def test_zero_size_partition_rejected():
    adapter = SyntheticAdapter(train_size=0)

    with pytest.raises(ValueError):
        list(
            traversal.training_position_batches(
                adapter=adapter,
                batch_size=4,
                experimental_seed=42,
                epoch=0,
            )
        )


def test_final_batch_size_preserved():
    adapter = SyntheticAdapter(train_size=23)

    batches = list(
        traversal.training_position_batches(
            adapter=adapter,
            batch_size=5,
            experimental_seed=42,
            epoch=0,
        )
    )

    assert [len(batch) for batch in batches] == [5, 5, 5, 5, 3]
