"""Committed VAE generation integrity and mutation regressions.

All filesystem mutations occur in disposable synthetic test fixtures.
Does not train models or publish authoritative checkpoints.
"""

import os

import optree
import keras
import pytest
import tensorflow as tf

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_checkpoint_committed_generation as verifier,
    variational_autoencoder_checkpoint_commit_record as commit,
)

from tests.test_variational_autoencoder_checkpoint_committed_generation_revision_2026 import (
    committed_fixture,
    verify,
)

from tests.test_variational_autoencoder_checkpoint_loader_roundtrip_revision_2026 import (
    synthetic_bundle,
)


@pytest.mark.parametrize(
    "filename",
    [
        "vae_encoder_state.npz",
        "vae_decoder_state.npz",
    ],
)
def test_corrupted_checkpoint_state_rejected(
    committed_fixture,
    filename,
):
    path = committed_fixture["generation"] / filename
    raw = bytearray(path.read_bytes())
    assert raw

    raw[-1] ^= 1
    path.write_bytes(raw)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify(committed_fixture)


def test_commit_record_mutation_during_verification_rejected(
    committed_fixture,
    monkeypatch,
):
    original = verifier.read_anchored_vae_checkpoint_bundle_fd
    events = []

    def verify_then_change_commit(*args, **kwargs):
        bundle = original(*args, **kwargs)
        events.append("checkpoint_verified")

        marker = committed_fixture["marker"]
        replacement = dict(committed_fixture["record"])
        replacement["manifest_sha256"] = "0" * 64

        marker.write_bytes(
            commit.encode_vae_commit_record(replacement)
        )

        events.append("commit_modified")
        return bundle

    monkeypatch.setattr(
        verifier,
        "read_anchored_vae_checkpoint_bundle_fd",
        verify_then_change_commit,
    )

    with pytest.raises(
        RuntimeError,
        match="generation record changed",
    ):
        verify(committed_fixture)

    assert events == [
        "checkpoint_verified",
        "commit_modified",
    ]


def test_manifest_mutation_during_verification_rejected(
    committed_fixture,
    monkeypatch,
):
    original = verifier.read_anchored_vae_checkpoint_bundle_fd
    events = []

    def verify_then_change_manifest(*args, **kwargs):
        bundle = original(*args, **kwargs)
        events.append("checkpoint_verified")

        manifest = (
            committed_fixture["generation"]
            / "vae_checkpoint_manifest.json"
        )

        manifest.write_bytes(
            manifest.read_bytes() + b"modified"
        )

        events.append("manifest_modified")
        return bundle

    monkeypatch.setattr(
        verifier,
        "read_anchored_vae_checkpoint_bundle_fd",
        verify_then_change_manifest,
    )

    with pytest.raises(
        ValueError,
        match="manifest SHA-256 mismatch",
    ):
        verify(committed_fixture)

    assert events == [
        "checkpoint_verified",
        "manifest_modified",
    ]
