"""Focused VAE best/last checkpoint state-binding regression.

No real datasets, Slurm jobs, filesystem publication, or training
authorization are involved.

The checkpoint preparation call is intercepted so the test can inspect
the exact models supplied by the bridge. Existing independent tests
cover real checkpoint publication and restoration.
"""

from copy import deepcopy
from io import BytesIO

import numpy as np
import pytest
import tensorflow as tf

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_training_checkpoint_bridge as bridge,
)
from trustforge.models.paper01_revision_2026.variational_autoencoder_weight_selection import (
    capture_vae_weights,
)
from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_state import (
    capture_vae_state,
    variable_specification,
)
from trustforge.models.paper01_revision_2026.variational_autoencoder_checkpoint_npz import (
    encode_network_state,
)


CLASS_COUNT = 9
MAPPING = [str(i) for i in range(CLASS_COUNT)]


def fixture_result():
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)

    encoder, decoder = bridge.build_models(num_classes=CLASS_COUNT)

    initial = capture_vae_weights(
        encoder=encoder,
        decoder=decoder,
    )

    # Use distinct finite states for the two roles.
    last = {
        name: tuple(
            np.asarray(array + 0.01, dtype=array.dtype)
            for array in arrays
        )
        for name, arrays in initial.items()
    }

    history = [
        {
            "epoch": 0,
            "completed_updates": 2,
            "validation": {"total_loss": 1.0},
        },
        {
            "epoch": 1,
            "completed_updates": 4,
            "validation": {"total_loss": 2.0},
        },
    ]

    trained = {
        "epochs_completed": 2,
        "completed_updates": 4,
        "history": history,
        "epoch_selection": {"best_epoch": 0},
        "best_weights": {
            "best_epoch": 0,
            "best_validation_total_loss": 1.0,
            "encoder_weights": initial["encoder"],
            "decoder_weights": initial["decoder"],
        },
        "last_weights": last,
    }

    governed_result = {
        "dataset_id": "ustc_tfc2016",
        "experimental_seed": 42,
        "class_count": CLASS_COUNT,
        "dataset_provenance": {"fixture_only": True},
        "training_result": trained,
        "checkpoint_status": "NOT_PREPARED",
        "publication_status": "NOT_PUBLISHED",
    }

    evidence = {}

    for role, epoch, updates, loss in (
        ("best", 0, 2, 1.0),
        ("last", 1, 4, 2.0),
    ):
        evidence[role] = {
            "dataset_id": "ustc_tfc2016",
            "experimental_seed": 42,
            "class_count": CLASS_COUNT,
            "class_mapping": list(MAPPING),
            "checkpoint_role": role,
            "completed_epochs": epoch + 1,
            "completed_updates": updates,
            "validation_criterion": {
                "metric": "validation_total_loss",
                "direction": "minimize",
                "value": loss,
            },
            "selection_policy_id":
                "min_validation_total_loss_earliest_epoch_v1",
            "validation_seed_policy_id":
                "fixed_per_batch_across_epochs_v1",
            "effective_configuration": {
                "epochs": 2,
                "batch_size": 64,
                "learning_rate": 0.001,
            },
        }

    return governed_result, evidence, {"best": initial, "last": last}


def install_capture(monkeypatch):
    observed = {}

    def inspect_preparation(
        *, encoder, decoder, manifest_evidence, dataset_provenance
    ):
        role = manifest_evidence["checkpoint_role"]

        assert role not in observed
        assert dataset_provenance == {"fixture_only": True}

        state = capture_vae_state(
            encoder=encoder,
            decoder=decoder,
        )

        archives = {}

        for name, network in (
            ("encoder", encoder),
            ("decoder", decoder),
        ):
            archives[name] = encode_network_state(
                role=name,
                state=state[name],
                specification=variable_specification(
                    network, role=name
                ),
            )

            # Confirm the existing serializer produces a readable NPZ.
            with np.load(
                BytesIO(archives[name]),
                allow_pickle=False,
            ) as loaded:
                assert loaded.files

        observed[role] = {
            "weights": capture_vae_weights(
                encoder=encoder,
                decoder=decoder,
            ),
            "state": state,
            "archives": archives,
        }

        return {
            "metadata": {"checkpoint_role": role},
            "artifacts": archives,
        }

    monkeypatch.setattr(
        bridge,
        "prepare_vae_checkpoint_artifacts",
        inspect_preparation,
    )

    return observed


def test_best_and_last_have_correct_distinct_serialized_states(
    monkeypatch,
):
    governed_result, evidence, expected = fixture_result()
    observed = install_capture(monkeypatch)

    prepared = bridge.prepare_vae_training_checkpoints(
        governed_result=governed_result,
        class_mapping=MAPPING,
        manifest_evidence_by_role=evidence,
    )

    assert set(prepared) == {"best", "last"}
    assert set(observed) == {"best", "last"}

    for role in ("best", "last"):
        for name in ("encoder", "decoder"):
            actual_weights = observed[role]["weights"][name]
            expected_weights = expected[role][name]

            assert len(actual_weights) == len(expected_weights)

            for actual, reference in zip(
                actual_weights, expected_weights, strict=True
            ):
                np.testing.assert_array_equal(actual, reference)

    # Both network states, not merely checkpoint labels, differ.
    for name in ("encoder", "decoder"):
        best_state = observed["best"]["state"][name]
        last_state = observed["last"]["state"][name]

        assert set(best_state) == set(last_state)
        assert any(
            not np.array_equal(best_state[key], last_state[key])
            for key in best_state
        )

        assert (
            observed["best"]["archives"][name]
            != observed["last"]["archives"][name]
        )


@pytest.mark.parametrize(
    "role,field,value",
    [
        ("best", "checkpoint_role", "last"),
        ("best", "completed_updates", 4),
        ("last", "completed_epochs", 1),
        ("last", "experimental_seed", 43),
    ],
)
def test_mismatched_checkpoint_identity_is_rejected(
    monkeypatch, role, field, value
):
    governed_result, evidence, _ = fixture_result()
    observed = install_capture(monkeypatch)

    evidence[role][field] = value

    with pytest.raises(ValueError, match="identity mismatch"):
        bridge.prepare_vae_training_checkpoints(
            governed_result=governed_result,
            class_mapping=MAPPING,
            manifest_evidence_by_role=evidence,
        )

    assert role not in observed


def test_unapproved_effective_configuration_rejected(monkeypatch):
    governed_result, evidence, _ = fixture_result()
    observed = install_capture(monkeypatch)

    evidence["best"]["effective_configuration"]["epochs"] = 3

    with pytest.raises(ValueError, match="effective configuration"):
        bridge.prepare_vae_training_checkpoints(
            governed_result=governed_result,
            class_mapping=MAPPING,
            manifest_evidence_by_role=evidence,
        )

    assert observed == {}


def test_wrong_selected_epoch_rejected(monkeypatch):
    governed_result, evidence, _ = fixture_result()
    observed = install_capture(monkeypatch)

    governed_result["training_result"]["epoch_selection"][
        "best_epoch"
    ] = 1

    with pytest.raises(ValueError, match="selected best epoch"):
        bridge.prepare_vae_training_checkpoints(
            governed_result=governed_result,
            class_mapping=MAPPING,
            manifest_evidence_by_role=evidence,
        )

    assert observed == {}
