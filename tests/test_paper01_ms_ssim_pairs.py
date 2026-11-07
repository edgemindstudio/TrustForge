import itertools
import numpy as np
import pytest
from trustforge.evaluation.paper01_ms_ssim_pairs import (
    _decode_pair_ranks, select_ms_ssim_pairs,
)


def select(labels, classes=2, digest="a" * 64):
    return select_ms_ssim_pairs(
        labels=labels, class_count=classes, manifest_sha256=digest,
    )


def test_rank_decoder_is_exact_bijection():
    for count in range(2, 12):
        expected = list(itertools.combinations(range(count), 2))
        observed = _decode_pair_ranks(
            np.arange(len(expected), dtype=np.int64), count,
        )
        assert observed.tolist() == [list(pair) for pair in expected]


def test_replay_unique_pairs_and_manifest_order():
    labels = np.tile(np.arange(2, dtype=np.int32), 2000)
    result = select(labels)
    assert result == select(labels.copy())
    for c in range(2):
        record = result["classes"][str(c)]
        local = np.array(record["local_pair_indices"])
        global_pairs = np.array(record["manifest_position_pairs"])
        assert local.shape == global_pairs.shape == (200, 2)
        assert np.all(local[:, 0] < local[:, 1])
        assert len({tuple(pair) for pair in local}) == 200
        assert np.all(labels[global_pairs] == c)
        positions = np.flatnonzero(labels == c)
        np.testing.assert_array_equal(global_pairs, positions[local])
        ranks = np.random.default_rng(42 + c).choice(
            2000 * 1999 // 2, size=200, replace=False,
        )
        decoded_ranks = (
            local[:, 0] * (3999 - local[:, 0]) // 2
            + local[:, 1] - local[:, 0] - 1
        )
        np.testing.assert_array_equal(decoded_ranks, ranks)


def test_label_storage_normalized_and_manifest_bound():
    labels = np.repeat(np.arange(2), 2000)
    assert select(labels.astype(np.int32)) == select(labels.astype(np.int64))
    assert select(labels)["manifest_sha256"] != select(
        labels, digest="b" * 64,
    )["manifest_sha256"]


@pytest.mark.parametrize("labels", [
    np.zeros(4000, dtype=np.int64),
    np.repeat(np.arange(2), 1999),
    np.repeat(np.arange(2), 2000).astype(np.float32),
    np.repeat(np.arange(2), 2000).reshape(2, 2000),
    np.repeat(np.arange(1, 3), 2000),
])
def test_invalid_or_incomplete_pool_rejected(labels):
    with pytest.raises(ValueError):
        select(labels)


def test_invalid_manifest_identity_rejected():
    with pytest.raises(ValueError):
        select(np.repeat(np.arange(2), 2000), digest="unverified")
