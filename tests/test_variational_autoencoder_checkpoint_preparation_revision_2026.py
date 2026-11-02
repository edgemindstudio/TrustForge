"""In-memory VAE checkpoint preparation regressions.

Uses synthetic untrained model objects and disposable test directories.
Does not train models or publish authoritative checkpoints.
"""

import hashlib

import optree
import keras
import pytest
import tensorflow as tf

from trustforge.provenance import canonical_json_bytes

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_checkpoint_preparation as preparation,
    variational_autoencoder_checkpoint_anchored_bundle as bundle_reader,
    variational_autoencoder_checkpoint_compatibility as compatibility,
    variational_autoencoder_checkpoint_manifest as manifest,
)

from tests.test_variational_autoencoder_checkpoint_loader_roundtrip_revision_2026 import (
    synthetic_bundle,
    DATASET_ID,
    SEED,
    CLASS_MAPPING,
)


def prepare_from_fixture(bundle):
    evidence = {
        key: value
        for key, value in bundle["metadata"].items()
        if key not in preparation.DERIVED_FIELDS
    }

    return preparation.prepare_vae_checkpoint_artifacts(
        encoder=bundle["encoder"],
        decoder=bundle["decoder"],
        manifest_evidence=evidence,
        dataset_provenance=bundle["provenance"],
    )


@pytest.fixture
def models_and_evidence(synthetic_bundle):
    from trustforge.models.paper01_revision_2026 import (
        variational_autoencoder_model as model,
    )

    encoder, decoder = model.build_models(
        latent_dim=100,
        num_classes=len(CLASS_MAPPING),
    )

    return {
        **synthetic_bundle,
        "encoder": encoder,
        "decoder": decoder,
    }


def test_preparation_produces_canonical_artifacts(
    models_and_evidence,
):
    result = prepare_from_fixture(models_and_evidence)

    metadata = result["metadata"]
    artifacts = result["artifacts"]

    assert set(artifacts) == {
        manifest.MANIFEST_FILENAME,
        "vae_dataset_provenance.json",
        "vae_encoder_state.npz",
        "vae_decoder_state.npz",
    }

    assert artifacts[manifest.MANIFEST_FILENAME] == (
        canonical_json_bytes(metadata) + b"\n"
    )

    for name, digest in metadata["state_files"].items():
        assert hashlib.sha256(
            artifacts[name]
        ).hexdigest() == digest

    assert hashlib.sha256(
        artifacts["vae_dataset_provenance.json"]
    ).hexdigest() == metadata["dataset_provenance_sha256"]


def test_prepared_bundle_accepted_by_anchored_reader(
    models_and_evidence,
    tmp_path,
):
    result = prepare_from_fixture(models_and_evidence)

    directory = tmp_path / "prepared_bundle"
    directory.mkdir()

    for name, raw in result["artifacts"].items():
        (directory / name).write_bytes(raw)

    verified = bundle_reader.read_anchored_vae_checkpoint_bundle(
        directory,
        requested_dataset_id=DATASET_ID,
        requested_experimental_seed=SEED,
        requested_class_mapping=CLASS_MAPPING,
    )

    assert verified["metadata"] == result["metadata"]
    assert (
        verified["dataset_provenance"]
        == result["dataset_provenance"]
    )


def test_missing_scientific_metadata_rejected(
    models_and_evidence,
):
    evidence = {
        key: value
        for key, value in models_and_evidence["metadata"].items()
        if key not in preparation.DERIVED_FIELDS
    }

    evidence.pop("selection_policy_id")

    with pytest.raises(ValueError, match="evidence fields mismatch"):
        preparation.prepare_vae_checkpoint_artifacts(
            encoder=models_and_evidence["encoder"],
            decoder=models_and_evidence["decoder"],
            manifest_evidence=evidence,
            dataset_provenance=models_and_evidence["provenance"],
        )
