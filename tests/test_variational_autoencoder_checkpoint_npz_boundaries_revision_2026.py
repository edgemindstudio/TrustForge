"""Adversarial ZIP-boundary tests for the VAE NPZ codec.

These tests intentionally establish the required stricter
checkpoint format. They use synthetic in-memory archives only.
"""

import numpy as np
import pytest

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_checkpoint_npz as codec,
)


def make_archive():
    specification = (
        {
            "key": "encoder_0000",
            "shape": (2, 3),
            "dtype": "float32",
            "trainable": True,
        },
    )

    state = {
        "encoder_0000": np.arange(
            6, dtype=np.float32
        ).reshape(2, 3),
    }

    raw = codec.encode_network_state(
        role="encoder",
        state=state,
        specification=specification,
    )

    return specification, state, raw


def test_canonical_archive_accepted():
    specification, state, raw = make_archive()

    decoded = codec.decode_network_state(
        role="encoder",
        raw=raw,
        specification=specification,
    )

    np.testing.assert_array_equal(
        decoded["encoder_0000"],
        state["encoder_0000"],
    )


@pytest.mark.parametrize(
    "suffix",
    [
        b"UNEXPECTED_TRAILER",
        b"\x00" * 16,
    ],
)
def test_trailing_bytes_rejected(suffix):
    specification, _, raw = make_archive()

    with pytest.raises(ValueError):
        codec.decode_network_state(
            role="encoder",
            raw=raw + suffix,
            specification=specification,
        )


@pytest.mark.parametrize(
    "prefix",
    [
        b"UNEXPECTED_PREFIX",
        b"\x00" * 16,
    ],
)
def test_prefixed_bytes_rejected(prefix):
    specification, _, raw = make_archive()

    with pytest.raises(ValueError):
        codec.decode_network_state(
            role="encoder",
            raw=prefix + raw,
            specification=specification,
        )


def test_truncated_archive_rejected():
    specification, _, raw = make_archive()

    with pytest.raises(ValueError):
        codec.decode_network_state(
            role="encoder",
            raw=raw[:-12],
            specification=specification,
        )
