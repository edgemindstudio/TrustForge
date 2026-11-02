"""Descriptor-anchored VAE checkpoint-bundle regressions.

Uses disposable synthetic artifacts from the existing round-trip fixture.
No real datasets, training, or checkpoint publication.
"""

from contextlib import contextmanager

import optree
import keras
import pytest
import tensorflow as tf

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_checkpoint_anchored_bundle as bundle_reader,
    variational_autoencoder_checkpoint_directory_anchor as anchor,
)

from tests.test_variational_autoencoder_checkpoint_loader_roundtrip_revision_2026 import (
    synthetic_bundle,
    DATASET_ID,
    SEED,
    CLASS_MAPPING,
)


def read_bundle(bundle, *, seed=SEED):
    return bundle_reader.read_anchored_vae_checkpoint_bundle(
        bundle["directory"],
        requested_dataset_id=DATASET_ID,
        requested_experimental_seed=seed,
        requested_class_mapping=CLASS_MAPPING,
    )


def test_anchored_bundle_returns_verified_evidence(synthetic_bundle):
    result = read_bundle(synthetic_bundle)

    assert result["metadata"] == synthetic_bundle["metadata"]
    assert (
        result["dataset_provenance"]
        == synthetic_bundle["provenance"]
    )

    state_bytes = result["verified_state_bytes"]

    assert set(state_bytes) == {
        "vae_encoder_state.npz",
        "vae_decoder_state.npz",
    }

    for name, raw in state_bytes.items():
        assert type(raw) is bytes
        assert raw == (
            synthetic_bundle["directory"] / name
        ).read_bytes()


def test_anchored_bundle_rejects_wrong_seed(synthetic_bundle):
    with pytest.raises(
        ValueError,
        match="experimental seed mismatch",
    ):
        read_bundle(synthetic_bundle, seed=SEED + 1)


def test_anchored_bundle_rejects_symlinked_manifest(
    synthetic_bundle,
):
    root = synthetic_bundle["directory"]
    original = root / bundle_reader.MANIFEST_FILENAME
    saved = root / "manifest-original.json"

    original.rename(saved)
    original.symlink_to(saved.name)

    with pytest.raises(ValueError):
        read_bundle(synthetic_bundle)


def test_directory_replacement_cannot_redirect_bundle(
    synthetic_bundle,
    monkeypatch,
):
    original_open = anchor.open_checkpoint_directory
    root = synthetic_bundle["directory"]
    events = []

    @contextmanager
    def open_then_replace(directory):
        with original_open(directory) as directory_fd:
            moved = root.with_name(root.name + "_original")
            root.rename(moved)

            root.mkdir()
            (root / bundle_reader.MANIFEST_FILENAME).write_bytes(
                b"malicious replacement"
            )

            events.append("path_replaced")
            yield directory_fd

    monkeypatch.setattr(
        bundle_reader,
        "open_checkpoint_directory",
        open_then_replace,
    )

    result = read_bundle(synthetic_bundle)

    assert events == ["path_replaced"]
    assert result["metadata"] == synthetic_bundle["metadata"]

    assert result["verified_state_bytes"]

    assert (
        root / bundle_reader.MANIFEST_FILENAME
    ).read_bytes() == b"malicious replacement"
