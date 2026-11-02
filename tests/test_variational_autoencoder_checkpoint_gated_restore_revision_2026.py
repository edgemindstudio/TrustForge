"""Candidate fingerprint-gated VAE restoration tests.

Disposable untrained models and synthetic model states only.
No real datasets, checkpoint publication, or training.
"""

import optree
import keras
import numpy as np
import pytest
import tensorflow as tf

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_model as model,
    variational_autoencoder_checkpoint_state as checkpoint_state,
    variational_autoencoder_checkpoint_fingerprint_registry as registry,
    variational_autoencoder_checkpoint_gated_restore as gated,
)


DATASETS = {
    "malex": 2,
    "cicmaldroid2020": 5,
    "ustc_tfc2016": 9,
    "malnet_image": 47,
}


def make_state(class_count):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)

    encoder, decoder = model.build_models(
        latent_dim=100,
        num_classes=class_count,
    )

    return checkpoint_state.capture_vae_state(
        encoder=encoder,
        decoder=decoder,
    )


@pytest.mark.parametrize("dataset_id", sorted(DATASETS))
def test_gated_restoration_round_trip(dataset_id):
    state = make_state(DATASETS[dataset_id])

    restored_encoder, restored_decoder = (
        gated.restore_vae_state_with_fingerprint_gate(
            dataset_id=dataset_id,
            state=state,
        )
    )

    observed = registry.validate_vae_architecture_fingerprint(
        dataset_id=dataset_id,
        encoder=restored_encoder,
        decoder=restored_decoder,
    )

    assert observed == registry.expected_vae_architecture_fingerprint(
        dataset_id=dataset_id,
    )

    restored_state = checkpoint_state.capture_vae_state(
        encoder=restored_encoder,
        decoder=restored_decoder,
    )

    for role in ("encoder", "decoder"):
        assert set(restored_state[role]) == set(state[role])

        for key, expected in state[role].items():
            np.testing.assert_array_equal(
                restored_state[role][key],
                expected,
            )


@pytest.mark.parametrize(
    "dataset_id",
    ["unknown", "", "candidate_v1"],
)
def test_unsupported_dataset_rejected(dataset_id):
    with pytest.raises(ValueError):
        gated.restore_vae_state_with_fingerprint_gate(
            dataset_id=dataset_id,
            state={},
        )


@pytest.mark.parametrize("dataset_id", [None, 9, True])
def test_nonstring_dataset_rejected(dataset_id):
    with pytest.raises(TypeError):
        gated.restore_vae_state_with_fingerprint_gate(
            dataset_id=dataset_id,
            state={},
        )


def test_actual_pair_fingerprint_mismatch_blocks_assignments(
    monkeypatch,
):
    state = make_state(9)
    original_factory = model.build_models
    original_validator = (
        gated.validate_vae_architecture_fingerprint
    )

    constructed = []
    assignments = []
    validations = []

    def tracking_factory(*, latent_dim, num_classes):
        encoder, decoder = original_factory(
            latent_dim=latent_dim,
            num_classes=num_classes,
        )

        constructed.append((encoder, decoder))

        for role, network in (
            ("encoder", encoder),
            ("decoder", decoder),
        ):
            for index, variable in enumerate(network.variables):
                original_assign = variable.assign

                def tracking_assign(
                    value,
                    *,
                    _assign=original_assign,
                    _role=role,
                    _index=index,
                ):
                    assignments.append((_role, _index))
                    return _assign(value)

                monkeypatch.setattr(
                    variable,
                    "assign",
                    tracking_assign,
                )

        return encoder, decoder

    def tracking_validator(*, dataset_id, encoder, decoder):
        validations.append(
            (dataset_id, encoder, decoder)
        )

        return original_validator(
            dataset_id=dataset_id,
            encoder=encoder,
            decoder=decoder,
        )

    monkeypatch.setattr(
        model,
        "build_models",
        tracking_factory,
    )

    monkeypatch.setattr(
        gated,
        "validate_vae_architecture_fingerprint",
        tracking_validator,
    )

    monkeypatch.setitem(
        registry.REFERENCE_FINGERPRINTS,
        "ustc_tfc2016",
        "0" * 64,
    )

    with pytest.raises(
        ValueError,
        match="VAE architecture fingerprint mismatch",
    ):
        gated.restore_vae_state_with_fingerprint_gate(
            dataset_id="ustc_tfc2016",
            state=state,
        )

    assert len(constructed) == 1
    assert len(validations) == 1

    dataset_id, encoder, decoder = validations[0]
    assert dataset_id == "ustc_tfc2016"
    assert encoder is constructed[0][0]
    assert decoder is constructed[0][1]

    assert assignments == []


def test_invalid_state_rejected():
    state = make_state(9)

    damaged = {
        role: {
            key: value.copy()
            for key, value in tensors.items()
        }
        for role, tensors in state.items()
    }

    key = next(iter(damaged["decoder"]))
    damaged["decoder"][key].flat[0] = np.nan

    with pytest.raises(ValueError):
        gated.restore_vae_state_with_fingerprint_gate(
            dataset_id="ustc_tfc2016",
            state=damaged,
        )


def test_restoration_failure_propagates(monkeypatch):
    state = make_state(9)

    class InjectedFailure(RuntimeError):
        pass

    def fail_restore(**kwargs):
        raise InjectedFailure("injected restore failure")

    monkeypatch.setattr(
        gated,
        "restore_vae_state_into_isolated_models",
        fail_restore,
    )

    with pytest.raises(
        InjectedFailure,
        match="injected restore failure",
    ):
        gated.restore_vae_state_with_fingerprint_gate(
            dataset_id="ustc_tfc2016",
            state=state,
        )
