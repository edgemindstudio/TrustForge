"""Negative in-memory VAE checkpoint preparation regressions.

All model objects and scientific evidence are synthetic.
No training, real datasets, or authoritative publication.
"""

from copy import deepcopy

import optree
import keras
import pytest
import tensorflow as tf

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_checkpoint_preparation as preparation,
    variational_autoencoder_checkpoint_fingerprint_registry as registry,
)

from tests.test_variational_autoencoder_checkpoint_preparation_revision_2026 import (
    models_and_evidence,
)

from tests.test_variational_autoencoder_checkpoint_loader_roundtrip_revision_2026 import (
    synthetic_bundle,
)


def evidence_from(bundle):
    return deepcopy({
        key: value
        for key, value in bundle["metadata"].items()
        if key not in preparation.DERIVED_FIELDS
    })


def prepare(bundle, evidence=None, provenance=None):
    return preparation.prepare_vae_checkpoint_artifacts(
        encoder=bundle["encoder"],
        decoder=bundle["decoder"],
        manifest_evidence=(
            evidence_from(bundle)
            if evidence is None else evidence
        ),
        dataset_provenance=(
            bundle["provenance"]
            if provenance is None else provenance
        ),
    )


def test_architecture_fingerprint_rejection_precedes_state_capture(
    models_and_evidence,
    monkeypatch,
):
    events = []

    def reject_fingerprint(**kwargs):
        events.append("fingerprint_rejected")
        raise ValueError("VAE architecture fingerprint mismatch")

    def forbidden_capture(**kwargs):
        events.append("capture_attempted")
        raise AssertionError("STATE CAPTURE MUST NOT OCCUR")

    monkeypatch.setattr(
        preparation,
        "validate_vae_architecture_fingerprint",
        reject_fingerprint,
    )

    monkeypatch.setattr(
        preparation,
        "capture_vae_state",
        forbidden_capture,
    )

    with pytest.raises(
        ValueError,
        match="architecture fingerprint mismatch",
    ):
        prepare(models_and_evidence)

    assert events == ["fingerprint_rejected"]


def test_invalid_provenance_rejected(models_and_evidence):
    provenance = deepcopy(models_and_evidence["provenance"])
    provenance["dataset_id"] = "malex"

    with pytest.raises(ValueError):
        prepare(
            models_and_evidence,
            provenance=provenance,
        )


@pytest.mark.parametrize(
    "role",
    ["candidate", "unknown", ""],
)
def test_invalid_checkpoint_role_rejected(
    models_and_evidence,
    role,
):
    evidence = evidence_from(models_and_evidence)
    evidence["checkpoint_role"] = role

    with pytest.raises(ValueError, match="checkpoint role"):
        prepare(models_and_evidence, evidence=evidence)


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), "not-a-number"],
)
def test_invalid_validation_metric_rejected(
    models_and_evidence,
    value,
):
    evidence = evidence_from(models_and_evidence)
    evidence["validation_criterion"]["value"] = value

    with pytest.raises((ValueError, TypeError)):
        prepare(models_and_evidence, evidence=evidence)


def test_unexpected_scientific_field_rejected(models_and_evidence):
    evidence = evidence_from(models_and_evidence)
    evidence["unauthorized_approval"] = True

    with pytest.raises(ValueError, match="evidence fields mismatch"):
        prepare(models_and_evidence, evidence=evidence)
