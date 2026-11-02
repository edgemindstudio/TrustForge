"""Controlled VAE publication-transition regression.

Injects a publication lock after successful artifact verification.
Uses only disposable synthetic checkpoint artifacts.
"""

import optree
import keras
import pytest
import tensorflow as tf

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_checkpoint_loader as loader,
    variational_autoencoder_checkpoint_anchored_bundle as bundle_reader,
    variational_autoencoder_checkpoint_anchored_artifact as anchored_artifact,
)

from tests.test_variational_autoencoder_checkpoint_loader_roundtrip_revision_2026 import (
    synthetic_bundle,
    load_bundle,
)


def test_publication_transition_after_verification_blocks_restoration(
    synthetic_bundle,
    monkeypatch,
):
    original_verify = (
        anchored_artifact.verify_anchored_vae_checkpoint_state_artifacts
    )
    events = []

    def verify_then_start_publication(directory_fd, metadata):
        verified = original_verify(directory_fd, metadata)
        events.append("artifacts_verified")

        import os
        os.mkdir(
            loader.PUBLICATION_LOCK,
            dir_fd=directory_fd,
        )
        events.append("publication_lock_created")

        return verified

    def forbidden_restoration(**kwargs):
        events.append("restoration_attempted")
        raise AssertionError(
            "restoration reached during publication transition"
        )

    monkeypatch.setattr(
        bundle_reader,
        "verify_anchored_vae_checkpoint_state_artifacts",
        verify_then_start_publication,
    )

    monkeypatch.setattr(
        loader,
        "restore_vae_state_into_isolated_models",
        forbidden_restoration,
    )

    with pytest.raises(
        RuntimeError,
        match="publication incomplete",
    ):
        load_bundle(synthetic_bundle)

    assert events == [
        "artifacts_verified",
        "publication_lock_created",
    ]
