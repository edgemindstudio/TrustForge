"""Synthetic-only isolated VAE checkpoint restoration tests."""

import optree
import keras
import numpy as np
import pytest
import tensorflow as tf

from trustforge.models.paper01_revision_2026 import (
    variational_autoencoder_model as model,
    variational_autoencoder_checkpoint_state as checkpoint_state,
    variational_autoencoder_checkpoint_restore as restore,
)


@pytest.fixture
def source_state():
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)

    encoder, decoder = model.build_models(
        latent_dim=100,
        num_classes=9,
    )

    state = checkpoint_state.capture_vae_state(
        encoder=encoder,
        decoder=decoder,
    )

    return encoder, decoder, state


@pytest.mark.parametrize("class_count", [2, 5, 9, 47])
def test_isolated_restoration_round_trip(class_count):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)

    source_encoder, source_decoder = model.build_models(
        latent_dim=100,
        num_classes=class_count,
    )

    source_state = checkpoint_state.capture_vae_state(
        encoder=source_encoder,
        decoder=source_decoder,
    )

    restored_encoder, restored_decoder = (
        restore.restore_vae_state_into_isolated_models(
            num_classes=class_count,
            state=source_state,
        )
    )

    assert restored_encoder is not source_encoder
    assert restored_decoder is not source_decoder

    restored_state = checkpoint_state.capture_vae_state(
        encoder=restored_encoder,
        decoder=restored_decoder,
    )

    for role in ("encoder", "decoder"):
        assert set(restored_state[role]) == set(source_state[role])

        for key, expected in source_state[role].items():
            np.testing.assert_array_equal(
                restored_state[role][key],
                expected,
            )


@pytest.mark.parametrize("missing_role", ["encoder", "decoder"])
def test_missing_network_rejected(source_state, missing_role):
    _, _, state = source_state

    incomplete = {
        role: values
        for role, values in state.items()
        if role != missing_role
    }

    with pytest.raises(ValueError):
        restore.restore_vae_state_into_isolated_models(
            num_classes=9,
            state=incomplete,
        )


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_nonfinite_decoder_rejected(source_state, bad_value):
    _, _, state = source_state

    damaged = {
        role: {
            key: value.copy()
            for key, value in tensors.items()
        }
        for role, tensors in state.items()
    }

    key = next(iter(damaged["decoder"]))
    damaged["decoder"][key].flat[0] = bad_value

    with pytest.raises(ValueError):
        restore.restore_vae_state_into_isolated_models(
            num_classes=9,
            state=damaged,
        )


def test_wrong_class_count_rejected(source_state):
    _, _, state = source_state

    with pytest.raises((ValueError, TypeError)):
        restore.restore_vae_state_into_isolated_models(
            num_classes=5,
            state=state,
        )


def test_assignment_failure_does_not_return_models(
    source_state,
    monkeypatch,
):
    _, _, state = source_state

    original_build = model.build_models

    class AssignmentFailure(RuntimeError):
        pass

    def failing_build_models(*, latent_dim, num_classes):
        encoder, decoder = original_build(
            latent_dim=latent_dim,
            num_classes=num_classes,
        )

        target = decoder.variables[0]

        def fail_assign(value):
            raise AssignmentFailure("injected assignment failure")

        monkeypatch.setattr(target, "assign", fail_assign)

        return encoder, decoder

    monkeypatch.setattr(
        model,
        "build_models",
        failing_build_models,
    )

    with pytest.raises(
        AssignmentFailure,
        match="injected assignment failure",
    ):
        restore.restore_vae_state_into_isolated_models(
            num_classes=9,
            state=state,
        )
