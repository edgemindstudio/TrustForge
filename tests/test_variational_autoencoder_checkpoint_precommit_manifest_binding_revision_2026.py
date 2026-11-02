"""Synthetic precommit exact-manifest binding regressions."""

from copy import deepcopy

import pytest

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_checkpoint_precommit_verification as adapter,
)


MANIFEST = "vae_checkpoint_manifest.json"
ENCODER = "vae_encoder_state.npz"
DECODER = "vae_decoder_state.npz"


@pytest.fixture
def evidence(monkeypatch):
    prepared = {
        "metadata": {"dataset_id": "synthetic"},
        "dataset_provenance": {"source": "fixture"},
        "artifacts": {
            MANIFEST: b"expected-canonical-manifest",
            ENCODER: b"synthetic-encoder",
            DECODER: b"synthetic-decoder",
        },
    }

    bundle = {
        "metadata": deepcopy(prepared["metadata"]),
        "dataset_provenance": deepcopy(
            prepared["dataset_provenance"]
        ),
        "verified_state_bytes": {
            ENCODER: prepared["artifacts"][ENCODER],
            DECODER: prepared["artifacts"][DECODER],
        },
    }

    monkeypatch.setattr(
        adapter,
        "validate_vae_publication_input",
        lambda candidate: deepcopy(prepared),
    )

    monkeypatch.setattr(
        adapter,
        "read_anchored_vae_checkpoint_bundle_fd",
        lambda fd, **kwargs: deepcopy(bundle),
    )

    return prepared, monkeypatch


def invoke(prepared):
    return adapter.verify_precommit_vae_generation(
        123,
        prepared=prepared,
        requested_dataset_id="synthetic",
        requested_experimental_seed=42,
        requested_class_mapping={"class0": 0},
    )


def test_different_manifest_bytes_rejected(evidence):
    prepared, monkeypatch = evidence

    monkeypatch.setattr(
        adapter,
        "read_anchored_regular_file",
        lambda *args, **kwargs: b"different-manifest",
        raising=False,
    )

    with pytest.raises(
        ValueError,
        match="manifest bytes mismatch",
    ):
        invoke(prepared)


def test_manifest_read_failure_propagates(evidence):
    prepared, monkeypatch = evidence

    def fail_manifest_read(*args, **kwargs):
        raise ValueError("synthetic manifest read failure")

    monkeypatch.setattr(
        adapter,
        "read_anchored_regular_file",
        fail_manifest_read,
        raising=False,
    )

    with pytest.raises(
        ValueError,
        match="synthetic manifest read failure",
    ):
        invoke(prepared)
