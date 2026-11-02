"""Synthetic VAE pre-commit adapter contract regressions."""

from copy import deepcopy

import pytest

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_checkpoint_precommit_verification as adapter,
)


ENCODER = "vae_encoder_state.npz"
DECODER = "vae_decoder_state.npz"


@pytest.fixture
def evidence(monkeypatch):
    prepared = {
        "metadata": {"dataset_id": "synthetic"},
        "dataset_provenance": {"origin": "fixture"},
        "artifacts": {
            "vae_checkpoint_manifest.json": b"synthetic-manifest",
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

    calls = []

    def validate(candidate):
        calls.append("validate")
        assert candidate is prepared
        return deepcopy(prepared)

    def read(fd, **kwargs):
        calls.append("read")
        assert fd == 123
        assert kwargs == {
            "requested_dataset_id": "synthetic",
            "requested_experimental_seed": 42,
            "requested_class_mapping": {"class0": 0},
        }
        return deepcopy(bundle)

    monkeypatch.setattr(
        adapter,
        "validate_vae_publication_input",
        validate,
    )
    def read_manifest(fd, name, *, max_bytes):
        calls.append("manifest_read")
        assert fd == 123
        assert name == "vae_checkpoint_manifest.json"
        assert max_bytes == adapter.MAX_MANIFEST_BYTES
        return prepared["artifacts"][name]

    monkeypatch.setattr(
        adapter,
        "read_anchored_regular_file",
        read_manifest,
    )

    monkeypatch.setattr(
        adapter,
        "read_anchored_vae_checkpoint_bundle_fd",
        read,
    )

    return prepared, bundle, calls, monkeypatch


def invoke(prepared):
    return adapter.verify_precommit_vae_generation(
        123,
        prepared=prepared,
        requested_dataset_id="synthetic",
        requested_experimental_seed=42,
        requested_class_mapping={"class0": 0},
    )


def test_matching_evidence_accepted(evidence):
    prepared, bundle, calls, _ = evidence
    result = invoke(prepared)

    assert result == bundle
    assert calls == ["validate", "manifest_read", "read"]


def test_metadata_mismatch_rejected(evidence):
    prepared, bundle, _, patch = evidence
    bundle["metadata"]["dataset_id"] = "wrong"

    patch.setattr(
        adapter,
        "read_anchored_vae_checkpoint_bundle_fd",
        lambda *args, **kwargs: deepcopy(bundle),
    )

    with pytest.raises(ValueError, match="metadata mismatch"):
        invoke(prepared)


def test_provenance_mismatch_rejected(evidence):
    prepared, bundle, _, patch = evidence
    bundle["dataset_provenance"]["origin"] = "wrong"

    patch.setattr(
        adapter,
        "read_anchored_vae_checkpoint_bundle_fd",
        lambda *args, **kwargs: deepcopy(bundle),
    )

    with pytest.raises(ValueError, match="provenance mismatch"):
        invoke(prepared)


def test_state_inventory_mismatch_rejected(evidence):
    prepared, bundle, _, patch = evidence
    del bundle["verified_state_bytes"][DECODER]

    patch.setattr(
        adapter,
        "read_anchored_vae_checkpoint_bundle_fd",
        lambda *args, **kwargs: deepcopy(bundle),
    )

    with pytest.raises(ValueError, match="state inventory mismatch"):
        invoke(prepared)


def test_state_bytes_mismatch_rejected(evidence):
    prepared, bundle, _, patch = evidence
    bundle["verified_state_bytes"][ENCODER] = b"wrong"

    patch.setattr(
        adapter,
        "read_anchored_vae_checkpoint_bundle_fd",
        lambda *args, **kwargs: deepcopy(bundle),
    )

    with pytest.raises(ValueError, match="state bytes mismatch"):
        invoke(prepared)


def test_invalid_input_rejected_before_read(evidence):
    prepared, _, calls, patch = evidence

    def reject(_):
        raise ValueError("synthetic invalid package")

    patch.setattr(
        adapter,
        "validate_vae_publication_input",
        reject,
    )

    with pytest.raises(ValueError, match="invalid package"):
        invoke(prepared)

    assert "read" not in calls
