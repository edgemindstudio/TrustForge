"""Exact-state VAE preparation-to-governed-loader regression.

Uses disposable synthetic, untrained models only.
No real data, training, or authoritative checkpoint publication.
"""

import optree
import keras
import numpy as np
import pytest
import tensorflow as tf

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_checkpoint_preparation as preparation,
    variational_autoencoder_checkpoint_loader as loader,
    variational_autoencoder_checkpoint_state as state_module,
)

from tests.test_variational_autoencoder_checkpoint_preparation_revision_2026 import (
    models_and_evidence,
)

from tests.test_variational_autoencoder_checkpoint_loader_roundtrip_revision_2026 import (
    synthetic_bundle,
    DATASET_ID,
    SEED,
    CLASS_MAPPING,
)


def test_prepared_checkpoint_restores_exact_model_state(
    models_and_evidence,
    tmp_path,
):
    encoder = models_and_evidence["encoder"]
    decoder = models_and_evidence["decoder"]

    expected_state = state_module.capture_vae_state(
        encoder=encoder,
        decoder=decoder,
    )

    evidence = {
        key: value
        for key, value in models_and_evidence["metadata"].items()
        if key not in preparation.DERIVED_FIELDS
    }

    prepared = preparation.prepare_vae_checkpoint_artifacts(
        encoder=encoder,
        decoder=decoder,
        manifest_evidence=evidence,
        dataset_provenance=models_and_evidence["provenance"],
    )

    directory = tmp_path / "synthetic_roundtrip"
    directory.mkdir()

    for filename, raw in prepared["artifacts"].items():
        (directory / filename).write_bytes(raw)

    restored = loader.load_governed_vae_checkpoint(
        directory,
        requested_dataset_id=DATASET_ID,
        requested_experimental_seed=SEED,
        requested_class_mapping=CLASS_MAPPING,
    )

    assert restored["metadata"] == prepared["metadata"]
    assert (
        restored["dataset_provenance"]
        == prepared["dataset_provenance"]
    )

    restored_state = state_module.capture_vae_state(
        encoder=restored["encoder"],
        decoder=restored["decoder"],
    )

    assert set(restored_state) == {"encoder", "decoder"}

    for role in ("encoder", "decoder"):
        assert set(restored_state[role]) == set(
            expected_state[role]
        )

        for key, expected in expected_state[role].items():
            np.testing.assert_array_equal(
                restored_state[role][key],
                expected,
            )
